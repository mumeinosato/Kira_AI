import asyncio
import numpy as np
import torch
from transformers import pipeline
from config import WHISPER_MODEL_SIZE

class WhisperManager:
    def __init__(self):
        self.whisper = None

    def initialize(self):
        print("-> Loading Whisper STT model...")
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
        print(f"   Whisper STT will use device: {device}")
        self.whisper = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{WHISPER_MODEL_SIZE}",
            device=device
        )
        print("   Whisper STT model loaded.")

    async def transcribe(self, audio_data: bytes) -> str:
        arr = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = await asyncio.to_thread(self.whisper, arr,generate_kwargs={"language": "ja", "task": "transcribe"})
        return result.get("text", "").strip()
