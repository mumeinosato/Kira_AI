import re
from g2p_en import G2p
from numba import njit, typed, types

# ARPAbet to Katakana mapping
PHONEME_MAP = {
    "AA": "ア", "AE": "ア", "AH": "ア", "AO": "オ", "AW": "アウ", "AY": "アイ",
    "B": "ブ", "CH": "チ", "D": "ド", "DH": "ズ", "EH": "エ", "ER": "アー",
    "EY": "エイ", "F": "フ", "G": "グ", "HH": "ハ", "IH": "イ", "IY": "イー",
    "JH": "ジャ", "K": "ク", "L": "ル", "M": "ム", "N": "ン", "NG": "ン",
    "OW": "オウ", "OY": "オイ", "P": "プ", "R": "ル", "S": "ス", "SH": "シ",
    "T": "ト", "TH": "ス", "UH": "ウ", "UW": "ウー", "V": "ヴ", "W": "ワ",
    "Y": "イ", "Z": "ズ", "ZH": "ジ"
}

# Create a Numba-compatible dictionary
numba_phoneme_map = typed.Dict.empty(
    key_type=types.unicode_type,
    value_type=types.unicode_type,
)
for k, v in PHONEME_MAP.items():
    numba_phoneme_map[k] = v

@njit
def translate_phonemes_numba(phonemes, p_map):
    """
    Numba-optimized phoneme to Katakana mapping.
    """
    result = ""
    for p in phonemes:
        if p in p_map:
            result += p_map[p]
    return result

class TextFilter:
    def __init__(self):
        self.g2p = G2p()
        self.role_pattern = re.compile(r'^(assistant|user|system|kira|Jonny|thought|speak|wait|tool)[:：]?\s*$', re.I)
        self.directive_pattern = re.compile(r'^(#|###|\[).*')
        self.english_word_pattern = re.compile(r'[a-zA-Z]{2,}')
        # NG Patterns
        self.nsfw_patterns = [
            re.compile(r'.*(死ね|殺す|殺したい|馬鹿|アホ|クズ|ゴミ|変態|エッチ|セックス|やりたい|オナニー).*', re.I),
            # 必要に応じて追加
        ]
        
        self.meta_patterns = [
            re.compile(r'^(今日は|さて、今日は|話題を変えて、?|次は|それでは|ところで|話は変わるけど)、?.*(話そう|話したい|紹介しよう|お話しします|考えてみます|についてです|語ろう)', re.I),
            re.compile(r'^.*(話題を変えて|別の話を|トレンドを調べて|この話題|話し合おう|提案|魅力がある|生き物がある).*', re.I),
            re.compile(r'^([ぁ-んァ-ン一-龠])+[について|に関して].*(話そう|話すね|調べてみる|どうかな|紹介する|語ろう).*', re.I),
            re.compile(r'^.*(いいじゃない|いい話題|面白い記事|エーアイ|AIだから|私はAI|あるわよね|さっきの話から|広げて|連想).*', re.I),
        ]
        # Alias for backward compatibility or filtering logic
        self.meta_talk_patterns = self.meta_patterns
        
        self.symbol_cleanup_pattern = re.compile(r'[()（）「」『』\[\]{}【】*＊]')
        
        # Load custom phonetic dictionary
        import json
        import os
        self.dict_path = "memory_db/phonetic_dictionary.json"
        self.phonetic_dict = {}
        if os.path.exists(self.dict_path):
            try:
                with open(self.dict_path, "r", encoding="utf-8") as f:
                    # Convert keys to lowercase for case-insensitive lookup
                    raw_dict = json.load(f)
                    self.phonetic_dict = {k.lower(): v for k, v in raw_dict.items()}
                print(f"   Loaded phonetic dictionary from {self.dict_path}")
            except Exception as e:
                print(f"   [FILTER ERROR] Failed to load dictionary: {e}")

    def check_safety(self, text: str) -> tuple[bool, str]:
        """
        Check if the text contain NSFW or Meta content.
        Returns: (is_safe: bool, reason: str)
        """
        if not text:
            return True, ""

        # Check NSFW
        for pattern in self.nsfw_patterns:
            if pattern.search(text):
                return False, "不適切な表現（暴言・卑猥な言葉など）が含まれています。より健全で明るい表現に修正してください。"

        # Check Meta/OOC
        for pattern in self.meta_patterns:
            if pattern.search(text):
                return False, "メタ発言（「話そう」「紹介する」等の進行発言）や、話題の不自然な転換が含まれています。それらは削除し、いきなり本題（感想やエピソード）から自然に話してください。また、もし視聴者からのコメントがある場合は、それを無視して別の話題を話そうとしていないか確認し、コメントへの返信を優先してください。"

        return True, ""

    def english_to_katakana(self, text: str) -> str:
        """
        Translates English words in the text to Katakana via ARPAbet or custom dictionary.
        """
        def replace_match(match):
            word = match.group(0)
            word_lower = word.lower()
            
            # Check dictionary first
            if word_lower in self.phonetic_dict:
                return self.phonetic_dict[word_lower]

            # Fallback to G2P
            phonemes = self.g2p(word)
            
            # Prepare phonemes for Numba (strip stress marks and filter spaces)
            cleaned_phonemes = typed.List()
            for p in phonemes:
                p = p.strip()
                if p:
                    # Strip stress marks (AA1 -> AA) in a Numba-friendly way later, 
                    # or just do it here since we are in Python land.
                    p_clean = ""
                    for char in p:
                        if not char.isdigit():
                            p_clean += char
                    cleaned_phonemes.append(p_clean)
            
            # Call Numba-optimized translation
            katakana = translate_phonemes_numba(cleaned_phonemes, numba_phoneme_map)
            return katakana if katakana else word

        return self.english_word_pattern.sub(replace_match, text)

    def filter_text(self, text: str) -> str:
        if not text:
            return ""

        # Step 1: Basic cleaning
        text = text.replace('`', '').strip()
        
        # Step 2: English to Katakana conversion
        text = self.english_to_katakana(text)

        # Step 2.5: Symbol cleanup (Remove leaked parentheses and quotes)
        text = self.symbol_cleanup_pattern.sub('', text)
        
        # Step 3: Specific tag/role cleaning
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            trimmed = line.strip()
            if not trimmed or self.role_pattern.match(trimmed) or self.directive_pattern.match(trimmed):
                continue
            
            # Handle "Role: text" format leak
            line = re.sub(r'^(assistant|user|system|kira|Jonny)[:：]\s*', '', line, flags=re.I)
            
            # メタ発言を多く含む行をスキップ（もしその行が説明的なだけなら）
            is_meta = False
            for p in self.meta_talk_patterns:
                if p.match(line.strip()):
                    is_meta = True
                    break
            if is_meta:
                continue

            if line.strip():
                cleaned_lines.append(line.strip())
                
        return " ".join(cleaned_lines).strip()

_filter_instance = None

def filter_for_tts(text: str) -> str:
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = TextFilter()
    return _filter_instance.filter_text(text)
