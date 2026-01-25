import asyncio
import os
import re
from llama_cpp import Llama, LlamaGrammar
from config import LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, LLM_MAX_RESPONSE_TOKENS
from persona import AI_PERSONALITY_PROMPT, EmotionalState

GBNF_PATH = "tool.gbnf"

class LLMManager:
    def __init__(self):
        self.llm = None

    def initialize(self):
        print("-> Loading LLM model...")
        if not os.path.exists(LLM_MODEL_PATH):
            raise FileNotFoundError(f"LLM model not found at {LLM_MODEL_PATH}")
        self.llm = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=N_CTX,
            n_gpu_layers=N_GPU_LAYERS,
            n_batch=512,
            kv_type="q4_0",
            flash_attn=True,
            logits_all=True,
            verbose=False
        )
        self.grammar = None
        if os.path.exists(GBNF_PATH):
            print(f"   Loading GBNF grammar from {GBNF_PATH}...")
            self.grammar = LlamaGrammar.from_file(GBNF_PATH)
        print("   LLM model loaded.")

    async def inference_stream(self, messages: list, current_emotion=None, memory_context: str = "", temperature: float = 0.7):
        # Static system prompt
        system_content = AI_PERSONALITY_PROMPT
        
        # Add memory context to the system content if available
        if memory_context and "No memories" not in memory_context:
            system_content += f"\n\n[Memory Context]:\n{memory_context}"

        # Consolidate any system messages from the 'messages' list into the main system prompt
        # to keep the conversation history clean for KV caching and model understanding.
        cleaned_history = []
        instructions = []
        for m in messages:
            if m["role"] == "system":
                instructions.append(m["content"])
            else:
                cleaned_history.append(m)
        
        if instructions:
            system_content += "\n\n### Internal System Directive (Do not repeat in output):\n" + "\n".join(instructions)

        system_tokens = self.llm.tokenize(system_content.encode("utf-8"), add_bos=True)
        max_response_tokens = LLM_MAX_RESPONSE_TOKENS
        token_limit = N_CTX - len(system_tokens) - max_response_tokens - 100 # safety buffer

        history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"), add_bos=False)) for m in cleaned_history)
        while history_tokens > token_limit and len(cleaned_history) > 1:
            print("   (Trimming conversation history to fit context window...)")
            cleaned_history.pop(0)
            history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"), add_bos=False)) for m in cleaned_history)

        full_prompt = [{"role": "system", "content": system_content}] + cleaned_history

        # --- Prefix & Rolling KV Cache Strategy ---
        # 1. Prefix Cache: The 'system_content' (persona) is static. 
        #    llama-cpp-python handles this automatically if the prompt prefix matches the cached tokens.
        # 2. Rolling Cache: We keep the conversation history in the context.
        #    By reusing the same 'clean_history' list (which we limit in size above),
        #    and passing it to the same LLM instance, we maximize cache hits.
        #
        # Note: We must ensure 'messages' structure is consistent for cache to work.
        
        try:
            stream = self.llm.create_chat_completion(
                messages=full_prompt,
                max_tokens=LLM_MAX_RESPONSE_TOKENS,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=True,
                grammar=self.grammar,
                stop=["\nJonny:", "\nKira:", "</s>"] 
            )
            for chunk in stream:
                if 'content' in chunk['choices'][0]['delta']:
                    content = chunk['choices'][0]['delta']['content']
                    # print(content, end="", flush=True) # DEBUG
                    yield content
                    await asyncio.sleep(0) # Yield control back to event loop
        except Exception as e:
            print(f"   ERROR during LLM inference: {e}")
            yield "Oops, my brain just short-circuited."

    async def inference(self, messages: list, current_emotion=None, memory_context: str = "") -> str:
        # Backward compatibility or for non-streaming needs
        text = ""
        async for chunk in self.inference_stream(messages, current_emotion, memory_context):
            text += chunk
        return text

    async def analyze_emotion(self, last_user_text: str, last_ai_response: str):
        return None

    def _clean_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.replace('</s>', '').strip()
        text = text.replace('*', '')
        return text
