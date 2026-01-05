
# Kira AI - Config loads from .env for secrets and sensitive info
import os
from dotenv import load_dotenv
load_dotenv()

# Model and runtime config (safe to share)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", -1))
N_CTX = int(os.getenv("N_CTX", 4096))
LLM_MAX_RESPONSE_TOKENS = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", 512))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
TTS_ENGINE = os.getenv("TTS_ENGINE", "edge")
AI_NAME = os.getenv("AI_NAME", "Kira")
PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", 1.0))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", 3))
MEMORY_PATH = os.getenv("MEMORY_PATH", "memory_db/")

# Secrets and API keys (must be in .env, never commit real values)
TWITCH_OAUTH_TOKEN = os.getenv("TWITCH_OAUTH_TOKEN", "")
TWITCH_BOT_USERNAME = os.getenv("TWITCH_BOT_USERNAME", "")
TWITCH_CHANNEL_TO_JOIN = os.getenv("TWITCH_CHANNEL_TO_JOIN", "")
#VIRTUAL_AUDIO_DEVICE = os.getenv("VIRTUAL_AUDIO_DEVICE", None)
VIRTUAL_AUDIO_DEVICE = None

STYLE_BERT_VITS2_MODEL_PATH = os.getenv("STYLE_BERT_VITS2_MODEL_PATH", "")
STYLE_BERT_VITS2_CONFIG_PATH = os.getenv("STYLE_BERT_VITS2_CONFIG_PATH", "")
STYLE_BERT_VITS2_STYLE_PATH = os.getenv("STYLE_BERT_VITS2_STYLE_PATH", "")

VTUBESTUDIO = os.getenv("VTUBESTUDIO", "false")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")