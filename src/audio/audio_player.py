import asyncio
import io
import time
import pygame
from config import VIRTUAL_AUDIO_DEVICE, VTUBESTUDIO

class AudioPlayer:
    def __init__(self, interruption_event):
        self.interruption_event = interruption_event
        pygame.mixer.pre_init(44100, -16, 2, 2048)
        pygame.init()
        try:
            pygame.mixer.init(devicename=VIRTUAL_AUDIO_DEVICE)
        except Exception as e:
            print(f"   [WARNING] Failed to init mixer with device {VIRTUAL_AUDIO_DEVICE}: {e}")
            pygame.mixer.init()

    async def play_audio_with_lip_sync(self, audio_bytes: bytes, lip_sync_data=None, vtube_client=None):
        if self.interruption_event.is_set() or not audio_bytes:
            return

        try:
            if pygame.mixer.get_busy():
                pygame.mixer.stop()
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            channel = sound.play()

            if lip_sync_data and vtube_client:
                start_time = time.time()
                for phoneme in lip_sync_data:
                    if self.interruption_event.is_set():
                        channel.stop()
                        break

                    elapsed = time.time() - start_time
                    if elapsed >= phoneme["time"]:
                        vtube_client.send_lip_sync({
                            "jaw_open": phoneme["mouth_open"]
                        })

                    await asyncio.sleep(0.01)

                vtube_client.send_lip_sync({"jaw_open": 0})

            while channel.get_busy():
                if self.interruption_event.is_set():
                    channel.stop()
                    break
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"   [AudioPlayer ERROR]: {e}")

    def stop(self):
        if pygame.mixer.get_init() and pygame.mixer.get_busy():
            pygame.mixer.stop()

    async def stream_audio(self, audio_queue, vtube_client=None):
        while True:
            if self.interruption_event.is_set():
                # Consume remaining items and clear
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except: break
                break

            item = await audio_queue.get()
            if item is None:
                break
            audio_bytes, lip_sync_data = item

            await self.play_audio_with_lip_sync(audio_bytes, lip_sync_data, vtube_client)
