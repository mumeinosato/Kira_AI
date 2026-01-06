import asyncio
import io
import torch
import soundfile as sf
from config import (
    TTS_ENGINE, STYLE_BERT_VITS2_MODEL_PATH,
    STYLE_BERT_VITS2_CONFIG_PATH, STYLE_BERT_VITS2_STYLE_PATH
)

try:
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModel
except ImportError:
    TTSModel = None

class TTSManager:
    def __init__(self):
        self.style_bert_model = None

    async def initialize(self):
        print(f"-> Initializing TTS engine: {TTS_ENGINE}...")
        if TTS_ENGINE == "edge":
            if not TTSModel:
                raise ImportError("Run 'pip install style-bert-vits2'")

            bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
            bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")

            '''
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                device = "xpu"
            '''
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.style_bert_model = TTSModel(
                model_path=STYLE_BERT_VITS2_MODEL_PATH,
                config_path=STYLE_BERT_VITS2_CONFIG_PATH,
                style_vec_path=STYLE_BERT_VITS2_STYLE_PATH,
                device=device
            )
        else:
            raise ValueError(f"Unsupported TTS_ENGINE: {TTS_ENGINE}")
        print(f"   {TTS_ENGINE.capitalize()} TTS ready.")

    async def generate_speech(self, text: str):
        if TTS_ENGINE == "edge":
            import re
            chunks = re.split(r'([。、!?！？])', text)

            temp_chunks = []
            for i in range(0, len(chunks) - 1, 2):
                temp_chunks.append(chunks[i] + chunks[i + 1])
            if len(chunks) % 2 == 1:
                last_chunk = chunks[-1]
                if last_chunk.strip():
                    temp_chunks.append(last_chunk)

            result_chunks = []
            i = 0
            while i < len(temp_chunks):
                current_chunk = temp_chunks[i]
                while (len(current_chunk) <= 4 or len(current_chunk) <= 10) and i + 1 < len(temp_chunks):
                    i += 1
                    current_chunk += temp_chunks[i]
                    if len(current_chunk) > 4 and len(current_chunk) > 10:
                        break
                result_chunks.append(current_chunk)
                i += 1

            for chunk_text in result_chunks:
                if not chunk_text.strip():
                    continue

                sr, audio = await asyncio.to_thread(
                    self.style_bert_model.infer,
                    text=chunk_text,
                    length=0.85
                )

                buf = io.BytesIO()
                sf.write(buf, audio, sr, format="WAV")
                buf.seek(0)

                yield buf.getvalue()