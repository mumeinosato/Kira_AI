"""
PersonaEnforcer - Ensures AI output matches the desired persona.

This module detects and filters out "assistant-like" responses,
forcing the AI to sound more like a human VTuber.
"""

import re
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class ViolationReport:
    """ペルソナ違反の詳細レポート。"""
    is_valid: bool
    violations: List[str]
    severity: int  # 0-10
    suggestion: str


class PersonaEnforcer:
    """
    AIの出力がペルソナに合っているかをチェックし、
    違反があれば修正を要求する。
    """
    
    # === 禁止フレーズ（これが含まれていたらアウト）===
    BANNED_PHRASES = [
        # 丁寧語・敬語系
        "ありがとうございます",
        "承知しました",
        "かしこまりました",
        "ございます",
        "いただき",
        "させていただ",
        "存じます",
        
        # アシスタント系
        "何かあれば",
        "何かありましたら",
        "お手伝い",
        "お役に立",
        "ご質問",
        "ご不明",
        "サポート",
        "アシスト",
        
        # 受動的すぎる
        "教えてください",
        "聞かせてください",
        "どうぞ",
        "遠慮なく",
        "何でも聞いて",
        
        # メタ的すぎる
        "話題を変え",
        "提案します",
        "アドバイス",
        "おすすめ",  # VTuberは自然に勧めるが「おすすめします」とは言わない
        
        # AI感が出る
        "AIとして",
        "プログラム",
        "設定",
        "キャラクター",
    ]
    
    # === 禁止パターン（正規表現）===
    BANNED_PATTERNS = [
        r"何か.*ありますか[？?]?",
        r"お役に立てれば",
        r"〜して(あげ|さしあげ)(ます|る)",
        r"(ご|お).*ください",
        r"いかがでしょうか",
        r"よろしければ",
        r"(何|なに)でも(言|い)って",
        r"力になれ(れば|たら)",
        r"お(手伝い|力)",
        r"(する|やる)ことができます",
        r"私(は|が)AI",
    ]
    
    # === 警告フレーズ（頻出するなら問題）===
    WARNING_PHRASES = [
        "！！！",  # 感嘆符連続は不自然
        "ですね！",
        "ますね！",
        "嬉しいです",
        "楽しいです",
    ]
    
    # === 推奨される話し方の例 ===
    GOOD_EXAMPLES = [
        "あー暇",
        "マジで？",
        "嘘でしょ",
        "知らんけど",
        "ところでさ",
        "やばくない？",
        "それな",
        "草",
        "わかるわー",
        "ていうかさ",
    ]
    
    def __init__(self):
        # コンパイル済み正規表現
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.BANNED_PATTERNS
        ]
    
    def check(self, text: str) -> Tuple[bool, str]:
        """
        テキストがペルソナに合っているかチェック。
        
        Args:
            text: チェックするテキスト
        
        Returns:
            (is_valid, reason) - 合格ならTrue、違反なら理由を返す
        """
        report = self.analyze(text)
        
        if report.is_valid:
            return True, ""
        else:
            return False, "; ".join(report.violations)
    
    def analyze(self, text: str) -> ViolationReport:
        """
        テキストを詳細に分析してレポートを生成。
        """
        violations = []
        severity = 0
        
        # 禁止フレーズチェック
        for phrase in self.BANNED_PHRASES:
            if phrase in text:
                violations.append(f"禁止フレーズ「{phrase}」")
                severity += 3
        
        # 禁止パターンチェック
        for i, pattern in enumerate(self._compiled_patterns):
            if pattern.search(text):
                violations.append(f"禁止パターン「{self.BANNED_PATTERNS[i]}」")
                severity += 2
        
        # 警告フレーズチェック
        warning_count = 0
        for phrase in self.WARNING_PHRASES:
            if phrase in text:
                warning_count += text.count(phrase)
        
        if warning_count >= 2:
            violations.append(f"警告フレーズ多用({warning_count}回)")
            severity += 1
        
        # 丁寧語チェック（「です」「ます」の連続は不自然）
        polite_endings = len(re.findall(r'(です|ます)[。！!？?\n]', text))
        if polite_endings >= 3:
            violations.append(f"丁寧語多用({polite_endings}回)")
            severity += 2
        
        # 質問で終わりすぎ
        questions = len(re.findall(r'[？?]', text))
        if questions >= 3:
            violations.append(f"質問過多({questions}個)")
            severity += 1
        
        is_valid = len(violations) == 0
        
        suggestion = ""
        if not is_valid:
            suggestion = self._generate_suggestion(violations)
        
        return ViolationReport(
            is_valid=is_valid,
            violations=violations,
            severity=min(severity, 10),
            suggestion=suggestion
        )
    
    def _generate_suggestion(self, violations: List[str]) -> str:
        """違反に基づいてリトライ用の提案を生成。"""
        
        if any("禁止フレーズ" in v for v in violations):
            return "もっとフランクに。敬語禁止。"
        
        if any("丁寧語" in v for v in violations):
            return "「です」「ます」を減らして。タメ口で。"
        
        if any("質問過多" in v for v in violations):
            return "質問じゃなくて自分の意見を言って。"
        
        return "もっと自然に、友達に話すみたいに。"
    
    def get_retry_prompt(self, original_text: str, violations: List[str]) -> str:
        """リトライ用のプロンプトを生成。"""
        
        suggestion = self._generate_suggestion(violations)
        
        return (
            f"[System] 前の発言はNGでした。理由: {', '.join(violations[:2])}\n"
            f"言い直して: {suggestion}\n"
            f"絶対に「です」「ます」「ありがとう」は使わないで。"
        )
    
    def quick_fix(self, text: str) -> str:
        """
        軽微な違反を自動修正する（LLMを使わない簡易修正）。
        重大な違反は修正できないのでそのまま返す。
        """
        result = text
        
        # 末尾の丁寧語を置換
        replacements = [
            (r'ですね([。！!])', r'だね\1'),
            (r'ますね([。！!])', r'るね\1'),
            (r'ですよ([。！!])', r'だよ\1'),
            (r'ますよ([。！!])', r'るよ\1'),
            (r'です([。！!])', r'だよ\1'),
            (r'ます([。！!])', r'るよ\1'),
            (r'でしょうか([。？?])', r'かな\1'),
            (r'ありがとうございます', r'ありがとね'),
        ]
        
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def score_naturalness(self, text: str) -> float:
        """
        テキストの自然さスコアを0-1で返す。
        1に近いほど良い（VTuberらしい）。
        """
        report = self.analyze(text)
        
        # 違反があるとスコアが下がる
        violation_penalty = report.severity * 0.1
        
        # 良い例が含まれているとボーナス
        good_count = sum(1 for phrase in self.GOOD_EXAMPLES if phrase in text)
        good_bonus = min(good_count * 0.05, 0.2)
        
        score = 1.0 - violation_penalty + good_bonus
        return max(0.0, min(1.0, score))
