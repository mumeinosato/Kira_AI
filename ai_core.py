import asyncio
import os
import re
from src.audio.audio_player import AudioPlayer
from src.audio.lip_sync import generate_lip_sync
from src.audio.text_filter import filter_for_tts
from src.audio.tts_manager import TTSManager
from src.audio.whisper_manager import WhisperManager
from src.llm.llm_manager import LLMManager
from src.vtube.vtube_client import VtubeStudioClient
from config import AI_NAME, VTUBESTUDIO
from persona import EmotionalState
from src.tools import ToolRegistry, register_default_tools
from src.utils.exceptions import SafetyViolationError
from src.brain.persona_enforcer import PersonaEnforcer

class AI_Core:
    def __init__(self, interruption_event, memory_manager=None):
        self.interruption_event = interruption_event
        self.is_initialized = False

        self.llm_manager = LLMManager()
        self.whisper_manager = WhisperManager()
        self.tts_manager = TTSManager()
        self.audio_player = AudioPlayer(interruption_event)
        self.vtube_client = VtubeStudioClient() if VTUBESTUDIO == "true" else None
        
        self.tool_registry = ToolRegistry()
        self.tool_results = [] # Store results to be injected into next prompt
        self.speak_log_path = "memory_db/speak.txt"
        self.memory_manager = memory_manager
        
        # Initialize TextFilter once
        from src.audio.text_filter import TextFilter
        self.text_filter = TextFilter()
        self.persona_enforcer = PersonaEnforcer()

    async def initialize(self):
        print("-> Initializing AI Core components...")
        try:
            await asyncio.to_thread(self.llm_manager.initialize)
            await asyncio.to_thread(self.whisper_manager.initialize)
            await self.tts_manager.initialize()

            if self.vtube_client:
                self.vtube_client.connect()
                print("   VTube Studio client connected.")

            register_default_tools(self.tool_registry)
            print(f"   Tools registered: {list(self.tool_registry.list_tools().keys())}")

            self.is_initialized = True
            print("   AI Core initialized successfully!")
        except Exception as e:
            print(f"FATAL: AI Core failed to initialize: {e}")
            self.is_initialized = False
            raise

    async def llm_inference(self, messages: list, memory_context: str = "") -> str:
        # Inject tool results if any
        if self.tool_results:
            combined_results = "\n".join(self.tool_results)
            messages.append({"role": "system", "content": f"[Tool Results]:\n{combined_results}"})
            self.tool_results = []
        return await self.llm_manager.inference(messages, None, memory_context)

    async def transcribe_audio(self, audio_data: bytes) -> str:
        return await self.whisper_manager.transcribe(audio_data)

    async def generate_and_process_stream(self, messages: list, memory_context: str = "", temperature: float = 0.7):
        """
        Generates LLM output stream and parses GBNF tags in real-time.
        Sentences are voiced in parallel with ongoing LLM streaming.
        """
        print("-> Starting streaming inference...")
        
        buffer = ""
        current_tag = None
        speak_buffer = ""
        complete_response = ""
        
        # Tracking state
        tag_pattern = re.compile(r'<(speak|thought|wait|tool)(.*?)>')
        end_tag_pattern = re.compile(r'</(speak|thought|tool)>')
        self_closing_pattern = re.compile(r'<(wait|tool)(.*?)\s*/>')

        # Using improved splitting logic
        from src.audio.lip_sync import split_text_for_streaming

        # TTS Parallelization structures
        sentence_queue = asyncio.Queue()
        audio_stream_queue = asyncio.Queue(maxsize=5)

        async def tts_worker():
            """Generates audio for each sentence in the background."""
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    await audio_stream_queue.put(None)
                    break
                
                try:
                    async for audio_chunk in self.tts_manager.generate_speech(sentence):
                        if self.interruption_event.is_set():
                            break
                        lip_sync_data = generate_lip_sync(sentence) if VTUBESTUDIO == "true" else None
                        await audio_stream_queue.put((audio_chunk, lip_sync_data))
                except Exception as e:
                    print(f"   [TTS Worker ERROR]: {e}")
                finally:
                    sentence_queue.task_done()

        # Start background workers
        tts_task = asyncio.create_task(tts_worker())
        audio_task = asyncio.create_task(self.audio_player.stream_audio(audio_stream_queue, self.vtube_client))

        try:
            print("-> Waiting for first LLM token...")
            async for chunk in self.llm_manager.inference_stream(messages, None, memory_context, temperature=temperature):
                # print(f"[DEBUG CLUNCK]: {chunk}")
                buffer += chunk
                complete_response += chunk

                # Process tags in buffer
                while buffer:
                    if not current_tag:
                        # Look for a starting tag
                        match = tag_pattern.search(buffer)
                        if match:
                            tag_name = match.group(1)
                            # Check if it's self-closing
                            sc_match = self_closing_pattern.match(buffer[match.start():])
                            if sc_match:
                                tag_name = sc_match.group(1)
                                attrs = sc_match.group(2)
                                await self._handle_tag(tag_name, attrs, "")
                                buffer = buffer[match.start() + len(sc_match.group(0)):]
                                continue
                            
                            current_tag = tag_name
                            # print(f"   [Processing Tag: <{current_tag}>]")
                            buffer = buffer[match.end():]
                        else:
                            break
                    
                    else:
                        # We are inside a tag (thought or speak)
                        end_match = end_tag_pattern.search(buffer)
                        if end_match:
                            end_tag_name = end_match.group(1)
                            if end_tag_name == current_tag:
                                content = buffer[:end_match.start()]
                                if current_tag == "speak":
                                    speak_buffer += content
                                    if speak_buffer.strip():
                                        cleaned_text = filter_for_tts(speak_buffer.strip())
                                        if cleaned_text:
                                            # --- REALTIME SAFETY & PERSONA CHECK ---
                                            # 1. Safety Filter
                                            is_safe, reason = self.text_filter.check_safety(cleaned_text)
                                            if not is_safe:
                                                print(f"   [FILTER INTERCEPT]: {reason}")
                                                await sentence_queue.put(None)
                                                raise SafetyViolationError(reason, cleaned_text)
                                            
                                            # 2. Persona Enforcer (Soft check - log warning, maybe automated fix later)
                                            is_valid, p_reason = self.persona_enforcer.check(cleaned_text)
                                            if not is_valid:
                                                print(f"   [PERSONA WARNING]: {p_reason}")
                                                # Optional: Apply quick fix if simple violation
                                                # cleaned_text = self.persona_enforcer.quick_fix(cleaned_text)

                                            print(f"   [SPEAKING]: {cleaned_text}")
                                            self._log_speak(cleaned_text)
                                            await sentence_queue.put(cleaned_text)
                                    speak_buffer = ""
                                else:
                                    await self._handle_tag(current_tag, "", content)
                                
                                # print(f"   [Finished Tag: <{current_tag}>]")
                                buffer = buffer[end_match.end():]
                                current_tag = None
                            else:
                                # Mismatching end tag? Or nested? GBNF should prevent this mostly.
                                buffer = buffer[end_match.end():]
                        else:
                            # No end tag yet. For 'speak', we can extract sentences.
                            if current_tag == "speak":
                                # Enhanced splitting: Split on sentence marks or long commas
                                split_pattern = r'([。！？!?\n])'
                                if len(buffer) > 30: # If buffer is getting long, split on commas too
                                    split_pattern = r'([。！？!?,、…\n])'
                                
                                s_match = re.search(split_pattern, buffer)
                                if s_match:
                                    sentence = buffer[:s_match.end()]
                                    speak_buffer += sentence
                                    # print(f"   [Buffer Debug]: Found sentence '{sentence}', speak_buffer now: '{speak_buffer}'")
                                    if len(speak_buffer.strip()) >= 5:
                                        sent = speak_buffer.strip()
                                        cleaned_sent = filter_for_tts(sent)
                                        if cleaned_sent:
                                            # --- REALTIME SAFETY & PERSONA CHECK ---
                                            is_safe, reason = self.text_filter.check_safety(cleaned_sent)
                                            if not is_safe:
                                                print(f"   [FILTER INTERCEPT]: {reason}")
                                                await sentence_queue.put(None)
                                                raise SafetyViolationError(reason, cleaned_sent)

                                            # Persona Check
                                            is_valid, p_reason = self.persona_enforcer.check(cleaned_sent)
                                            if not is_valid:
                                                print(f"   [PERSONA WARNING]: {p_reason}")

                                            print(f"   [SPEAKING]: {cleaned_sent}")
                                            self._log_speak(cleaned_sent)
                                            await sentence_queue.put(cleaned_sent)
                                        speak_buffer = ""
                                    buffer = buffer[s_match.end():]
                                else:
                                    break
                            else:
                                break

            # Final flush for any remaining speak_buffer if the stream ended unexpectedly
            if speak_buffer.strip():
                # print(f"   [Final Flush]: {speak_buffer.strip()}")
                sent = speak_buffer.strip()
                cleaned_sent = filter_for_tts(sent)
                if cleaned_sent:
                    print(f"   [SPEAKING]: {cleaned_sent}")
                    self._log_speak(cleaned_sent)
                    await sentence_queue.put(cleaned_sent)
                speak_buffer = ""
        finally:
            # Signal workers to finish
            await sentence_queue.put(None)
            # Wait for tasks with a larger timeout to ensure long audio finishes playing
            try:
                await asyncio.wait_for(asyncio.gather(tts_task, audio_task, return_exceptions=True), timeout=30)
            except asyncio.TimeoutError:
                print("   [WARNING] Parallel TTS tasks timed out during cleanup. Audio might have been truncated.")

        return complete_response

    async def _handle_tag(self, tag_name, attrs, content):
        if tag_name == "thought":
            print(f"   [THOUGHT]: {content.strip()}")
        elif tag_name == "wait":
            match = re.search(r'time="(\d+)"', attrs)
            if match:
                seconds = int(match.group(1))
                print(f"   [WAITING]: {seconds}s")
                await asyncio.sleep(seconds)
        elif tag_name == "speak":
             # This part handles single-sentence triggers if needed
             pass
        elif tag_name == "tool":
            # attrs might contain name="...", args="..."
            name_match = re.search(r'name="([^"]+)"', attrs)
            args_match = re.search(r'args="([^"]+)"', attrs)
            
            tool_name = name_match.group(1) if name_match else None
            tool_args_str = args_match.group(1) if args_match else content.strip()
            
            if tool_name:
                print(f"   [TOOL CALL]: {tool_name}({tool_args_str})")
                tool = self.tool_registry.get_tool(tool_name)
                if tool:
                    try:
                        # Assuming tools take a 'query' or generic kwargs
                        # For web_search, it takes 'query'
                        result = await tool.execute(query=tool_args_str, memory_manager=self.memory_manager)
                        print(f"   [TOOL RESULT]: {result[:100]}...")
                        self.tool_results.append(f"Tool '{tool_name}' result: {result}")
                    except Exception as e:
                        error_msg = f"Error executing tool {tool_name}: {e}"
                        print(f"   [TOOL ERROR]: {error_msg}")
                        self.tool_results.append(error_msg)
                else:
                    msg = f"Tool '{tool_name}' not found."
                    print(f"   [TOOL ERROR]: {msg}")
                    self.tool_results.append(msg)

    def extract_speak_text(self, full_response: str) -> str:
        """Extracts and joins all content within <speak> tags, cleaning up hallucinations."""
        speak_parts = re.findall(r'<speak>(.*?)</speak>', full_response, re.DOTALL)
        raw_text = " ".join(part.strip() for part in speak_parts if part.strip())
        return filter_for_tts(raw_text)

    def extract_thought_text(self, full_response: str) -> str:
        """Extracts content within <thought> tags."""
        thought_parts = re.findall(r'<thought>(.*?)</thought>', full_response, re.DOTALL)
        return " ".join(part.strip() for part in thought_parts if part.strip())

    async def _speak_sentence(self, text):
        """Voicing a sentence using the persistent audio player."""
        if not text: return
        # print(f"   [SPEAKING]: {text}")
        self.interruption_event.clear()
        
        audio_queue = asyncio.Queue(maxsize=1)
        
        async def generate_audio():
            async for audio_chunk in self.tts_manager.generate_speech(text):
                if self.interruption_event.is_set():
                    break
                lip_sync_data = generate_lip_sync(text) if VTUBESTUDIO == "true" else None
                await audio_queue.put((audio_chunk, lip_sync_data))
            await audio_queue.put(None)

        try:
            await asyncio.gather(
                generate_audio(),
                self.audio_player.stream_audio(audio_queue, self.vtube_client)
            )
        except Exception as e:
            print(f"   [TTS ERROR]: {e}")

    async def speak_text(self, text: str):
        await self._speak_sentence(text)

    def _log_speak(self, text: str):
        """Logs the spoken text to memory_db/speak.txt."""
        try:
            os.makedirs(os.path.dirname(self.speak_log_path), exist_ok=True)
            with open(self.speak_log_path, "a", encoding="utf-8") as f:
                f.write(text + "\n")
        except Exception as e:
            print(f"   [LOG ERROR]: Failed to log speak text: {e}")
