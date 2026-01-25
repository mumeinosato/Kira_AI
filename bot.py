# bot.py - Main application file with advanced memory and web search.

import os
import sys
import logging

# Aggressively silence Style-Bert-VITS2 logs and loguru BEFORE any other imports
os.environ["LOGURU_LEVEL"] = "ERROR"
logging.disable(logging.WARNING)
try:
    from loguru import logger
    logger.remove()
except ImportError:
    pass

import asyncio
import webrtcvad
import collections
import pyaudio
import time

from ai_core import AI_Core
from src.memory.memory import MemoryManager
from src.memory.summarizer import SummarizationManager
from config import (
    AI_NAME, PAUSE_THRESHOLD, VAD_AGGRESSIVENESS, ENABLE_YOUTUBE_COMMENTS, YOUTUBE_API_KEY, LIVE_ID
)

ENABLE_PROACTIVE_THOUGHTS = True
ENABLE_WEB_SEARCH = False
ENABLE_VAD = False


class VTubeBot:
    def __init__(self):
        self.interruption_event = asyncio.Event()
        self.processing_lock = asyncio.Lock()
        self.memory = MemoryManager()
        self.ai_core = AI_Core(self.interruption_event, memory_manager=self.memory)
        self.summarizer = SummarizationManager(self.ai_core, self.memory)
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        
        # Brain Components
        from src.brain.ai_state import AIState
        from src.brain.director import Director
        self.ai_state = AIState()
        self.director = Director()
        
        self.last_interaction_time = time.time()
        self.pyaudio_instance = None
        self.stream = None
        self.frames_per_buffer = int(16000 * 30 / 1000)
        
        self.bg_tasks = set() # Use a set for easier task management
        self.conversation_history = []
        self.conversation_segment = []
        
        # YouTube comment manager
        self.youtube_comment_manager = None
        if ENABLE_YOUTUBE_COMMENTS == "true" and YOUTUBE_API_KEY and LIVE_ID:
            from src.tools.tool.youtube import YoutubeCommentManager, YoutubeCommentTool
            self.youtube_comment_manager = YoutubeCommentManager(YOUTUBE_API_KEY, LIVE_ID)
            self.ai_core.tool_registry.register(YoutubeCommentTool(self.youtube_comment_manager))
            print("-> YouTube comment feature enabled.")

        #register_default_tools(self.ai_core.tool_registry)

    def reset_idle_timer(self):
        self.last_interaction_time = time.time()

    async def run(self):
        await self._main_loop()

    async def _main_loop(self):
        """Contains the primary startup and listening logic."""
        try:
            await self.ai_core.initialize()
            if not self.ai_core.is_initialized: return

            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=self.frames_per_buffer
            )

            print(f"\n--- {AI_NAME} is now running. Press Ctrl+C to exit. ---\n")
            
            background_task = asyncio.create_task(self.background_loop())
            self.bg_tasks.add(background_task)
 
            conversation_task = asyncio.create_task(self.conversation_loop())
            self.bg_tasks.add(conversation_task)
            
            # Start YouTube comment polling if enabled
            if self.youtube_comment_manager:
                youtube_task = asyncio.create_task(self.youtube_comment_manager.start_polling())
                self.bg_tasks.add(youtube_task)

            if ENABLE_VAD:
                await self.vad_loop()
            else:
                # Keep active even without VAD to maintain conversation and youtube polling
                while True:
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            print("Main loop cancelled.")
        finally:
            print("--- Cleaning up resources... ---")
            if self.stream: self.stream.stop_stream(); self.stream.close()
            if self.pyaudio_instance: self.pyaudio_instance.terminate()
            print("--- Cleanup complete. ---")


    ##マイクからの音声入力処理
    async def vad_loop(self):
        # This function's logic remains the same
        frames = collections.deque()
        triggered = False
        silent_chunks = 0
        max_silent_chunks = int(PAUSE_THRESHOLD * 1000 / 30)

        while True:
            data = await asyncio.to_thread(self.stream.read, self.frames_per_buffer, exception_on_overflow=False)
            is_speech = self.vad.is_speech(data, 16000)

            if self.processing_lock.locked() and is_speech:
                self.interruption_event.set()
                continue
            
            if not self.processing_lock.locked():
                if is_speech:
                    if not triggered:
                        print("🎤 Recording...")
                        triggered = True
                    frames.append(data)
                    silent_chunks = 0
                elif triggered:
                    frames.append(data)
                    silent_chunks += 1
                    if silent_chunks > max_silent_chunks:
                        audio_data = b"".join(frames)
                        frames.clear()
                        triggered = False
                        self.reset_idle_timer()
                        task = asyncio.create_task(self.handle_audio(audio_data))
                        self.bg_tasks.add(task)
                        task.add_done_callback(self.bg_tasks.discard)

    ##音声データをテキストに変換し、応答を生成
    async def handle_audio(self, audio_data: bytes):
        async with self.processing_lock:
            user_text = await self.ai_core.transcribe_audio(audio_data)
            if not user_text or len(user_text) < 3: return
            
            print(f">>> You said: {user_text}")

            # --- NEW: Ignore duplicate inputs ---
            if any(h["content"] == user_text for h in self.conversation_history):
                print(f"(Duplicate input ignored: {user_text})")
                return
            
            # User input triggers state update
            self.ai_state.update(0, event="got_reaction")
            
            contextual_prompt = f"Jonny says: \"{user_text}\""
            # Process as a user reaction action
            await self.process_and_respond(user_text, contextual_prompt, "user", temperature=0.7)

    async def process_and_respond(self, original_text: str, contextual_prompt: str, role: str, system_directive: str = "", temperature: float = 0.7, mode: str = "monologue"):

        if original_text:
            self.conversation_history.append({"role": role, "content": original_text})
            self.conversation_segment.append({"role": role, "content": original_text})

        # Keep history concise to prevent obsessive repetition or getting stuck in the past
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

        # Search memory more broadly if there is no user text
        search_query = original_text if original_text else (contextual_prompt if contextual_prompt else "Kiraの趣味や最近の出来事")
        mem_ctx = self.memory.search_memories(search_query, n_results=5) 
        # Prepare messages for LLM
        messages = list(self.conversation_history)
        
        # Add YouTube comment hint if comments are available
        # AND Override system_directive to prioritize comment response
        if self.youtube_comment_manager and self.youtube_comment_manager.has_comments():
            comment_count = self.youtube_comment_manager.get_comment_count()
            comment_hint = f"[視聴者のコメント]: 現在{comment_count}件のコメントがあります。`<tool name=\"youtube_comment\"/>`を使ってコメントを取得し、視聴者と対話してください。"
            messages.append({"role": "system", "content": comment_hint})
            
            # FORCE override directive to ensure AI focuses on comments
            system_directive = "視聴者からコメントが来ています。自分の話はいったん止めて、必ず `youtube_comment` ツールを使ってコメントを読み、返事をしてあげてください。"
        
        if system_directive:
            # Check if last action was a tool call
            if self.conversation_history and "(Called a tool" in self.conversation_history[-1]["content"]:
                # If we just used a tool, FORCE the AI to speak about it, instead of calling another tool
                system_directive = "ツールで得た情報について、あなた自身の感想や驚きをリスナーに話してください。再度ツールを使う必要はありません。"
            
            # Shift from "Instruction" to a more persona-friendly "Situation"
            messages.append({"role": "system", "content": f"[Current Situation/Idea]: {system_directive}"})

        # Use streaming inference with GBNF tags
        from src.utils.exceptions import SafetyViolationError
        try:
            response = await self.ai_core.generate_and_process_stream(
                messages, 
                mem_ctx,
                temperature=temperature
            )
            
            if response:
                # Store spoken text for display/TTL
                spoken_text = self.ai_core.extract_speak_text(response)
                
                # Store the internal state (especially tools) as a system-like hint or assistant history
                # But we need the LLM to know it just used a tool.
                if spoken_text:
                    self.conversation_history.append({"role": "assistant", "content": spoken_text})
                    self.conversation_segment.append({"role": "assistant", "content": spoken_text})
                elif "<tool" in response:
                    # If it only used a tool, we still need to record that it responded with SOMETHING
                    # so the history moves forward.
                    self.conversation_history.append({"role": "assistant", "content": "(Called a tool to research something)"})
                
                if role == "user" and original_text:
                     self.memory.add_memory(user_text=original_text, ai_text=spoken_text if spoken_text else response)

        except SafetyViolationError as e:
            print(f"   [GENERATION STOPPED]: {e.reason}")
            # Do NOT retry. Just stop this turn.
            # Optionally add a system note to history so LLM knows why it stopped?
            # self.conversation_history.append({"role": "system", "content": f"[System]: 前回の発言はフィルターにより中断されました（理由: {e.reason}）。"})
            return
        
        # --- NEW: Correction Loop for Missing Speech ---
        # If we had a thought but NO speech (and no tool call), force a short follow-up.
        if response and "<thought>" in response and not self.ai_core.extract_speak_text(response) and "<tool" not in response:
             print("   [Correction]: Detected thought without speech. Triggering follow-up...")
             
             # Add the thought to history so the AI knows it just thought that.
             # We want to persist this specific internal monologue only for this immediate correction context.
             # Actually, simpler: Just ask it to speak what it thought.
             
             correction_directive = "直前の思考（thought）を声に出して（speak）言ってください。"
             
             # Append a temporary system message to force output
             messages.append({"role": "assistant", "content": response}) # Assume it 'happened' internally
             messages.append({"role": "system", "content": correction_directive})
             
             try:
                 print("   [Correction]: meaningful silence recovery...")
                 retry_response = await self.ai_core.generate_and_process_stream(
                    messages, 
                    mem_ctx, 
                    temperature=0.7
                 )
                 # Update history if successful
                 if retry_response:
                     spoken = self.ai_core.extract_speak_text(retry_response)
                     if spoken:
                         self.conversation_history.append({"role": "assistant", "content": spoken})
                         self.conversation_segment.append({"role": "assistant", "content": spoken})
             except Exception as e:
                 print(f"   [Correction Failed]: {e}")
        
        self.reset_idle_timer()



    async def conversation_loop(self):
        """
        Main autonomous loop. The AI should speak sequentially.
        """
        print("-> Conversation loop started.")
        while True:
            if self.processing_lock.locked():
                await asyncio.sleep(0.1)
                continue
 
            # Sequential Turn logic:
            # If we are here, nothing else is talking.
            async with self.processing_lock:
                # 1. Update State
                idle_time = time.time() - self.last_interaction_time
                self.ai_state.update(idle_time)
                
                # 2. Gather Context
                context = {
                    "has_comments": self.youtube_comment_manager and self.youtube_comment_manager.has_comments(),
                    "idle_time": idle_time,
                    "recent_topics": self.ai_state.topics_discussed
                }
                
                # 3. Decide Action
                action = self.director.decide_action(self.ai_state, context)
                
                if action.mode.value == "wait":
                    await asyncio.sleep(5.0)
                    continue
                
                print(f"-> Starting organic turn (Mode: {action.mode.value}, Temp: {action.temperature:.2f})")
                
                # 4. Execute
                await self.process_and_respond(
                    "", 
                    action.directive, 
                    "assistant", 
                    system_directive=action.directive,
                    temperature=action.temperature,
                    mode=action.mode.value
                )
                
                # 5. Feedback
                if action.mode.value == "boke":
                    self.ai_state.update(0, event="boke")
                elif action.mode.value == "monologue":
                    self.ai_state.update(0, event="monologue")
                    if action.topic_hint:
                        self.ai_state.change_topic(action.topic_hint)
                else:
                    self.ai_state.update(0, event="spoke")
                
                self.director.record_action(action)
   
            # Loop delay: give more breath between turns
            await asyncio.sleep(10.0)
  
            # Loop delay: give more breath between turns
            await asyncio.sleep(10.0)

    async def background_loop(self):
        # We can keep this for summary tasks or other non-conversation background work.
        while True:
            await asyncio.sleep(10)
            
            if self.processing_lock.locked():
                continue

            # Task 3: Summarize conversation
            if len(self.conversation_segment) >= 8:
                async with self.processing_lock:
                    print("\n--- Summarizing conversation segment... ---")
                    await self.summarizer.consolidate_and_store(self.conversation_segment)
                    self.conversation_segment.clear()


# --- UPDATED: Graceful Shutdown Logic ---
async def main():
    bot = VTubeBot()
    try:
        await bot.run()
    except asyncio.CancelledError:
        print("Main task cancelled.")
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("\nApplication shutting down...")
    finally:
        # Gracefully cancel all running tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        # Gather all canceled tasks to let them finish
        group = asyncio.gather(*tasks, return_exceptions=True)
        loop.run_until_complete(group)
        loop.close()