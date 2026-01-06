import asyncio
from src.audio.audio_player import AudioPlayer
from src.audio.lip_sync import generate_lip_sync, split_text_for_streaming
from src.audio.tts_manager import TTSManager
from src.audio.whisper_manager import WhisperManager
from src.llm.llm_manager import LLMManager
from src.vtube.vtube_client import VtubeStudioClient
from config import AI_NAME, VTUBESTUDIO
from persona import EmotionalState

class AI_Core:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        self.is_initialized = False

        self.llm_manager = LLMManager()
        self.whisper_manager = WhisperManager()
        self.tts_manager = TTSManager()
        self.audio_player = AudioPlayer(interruption_event)
        self.vtube_client = VtubeStudioClient() if VTUBESTUDIO == "true" else None

    async def initialize(self):
        print("-> Initializing AI Core components...")
        try:
            await asyncio.to_thread(self.llm_manager.initialize)
            await asyncio.to_thread(self.whisper_manager.initialize)
            await self.tts_manager.initialize()

            if self.vtube_client:
                self.vtube_client.connect()
                print("   VTube Studio client connected.")

            self.is_initialized = True
            print("   AI Core initialized successfully!")
        except Exception as e:
            print(f"FATAL: AI Core failed to initialize: {e}")
            self.is_initialized = False
            raise

    async def llm_inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "") -> str:
        return await self.llm_manager.inference(messages, current_emotion, memory_context)

    async def analyze_emotion_of_turn(self, user_text: str, ai_response: str) -> EmotionalState | None:
        return await self.llm_manager.analyze_emotion(user_text, ai_response)

    async def transcribe_audio(self, audio_data: bytes) -> str:
        return await self.whisper_manager.transcribe(audio_data)

    async def speak_text(self, text: str):
        if not text:
            return
        print(f"<<< {AI_NAME} says: {text}")
        self.interruption_event.clear()

        try:
            chunks = split_text_for_streaming(text)
            audio_queue = asyncio.Queue(maxsize=2)

            async def generate_audio():
                for chunk in chunks:
                    if self.interruption_event.is_set():
                        break

                    #audio_bytes = await self.tts_manager.generate_speech(chunk)
                    async for audio_chunk in self.tts_manager.generate_speech(chunk):
                        if self.interruption_event.is_set():
                            break
                        lip_sync_data = generate_lip_sync(chunk) if VTUBESTUDIO == "true" else None
                        await audio_queue.put((audio_chunk, lip_sync_data))

                await audio_queue.put(None)

            await asyncio.gather(
                generate_audio(),
                self.audio_player.stream_audio(audio_queue, self.vtube_client)
            )

        except Exception as e:
            print(f"   ERROR during TTS generation: {e}")
