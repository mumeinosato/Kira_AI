import asyncio
import os
import re
from llama_cpp import Llama
from config import LLM_MODEL_PATH, N_CTX, N_GPU_LAYERS, LLM_MAX_RESPONSE_TOKENS
from persona import AI_PERSONALITY_PROMPT, EmotionalState

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
            verbose=False
        )
        print("   LLM model loaded.")

    async def inference(self, messages: list, current_emotion: EmotionalState, memory_context: str = "") -> str:
        system_prompt = AI_PERSONALITY_PROMPT
        system_prompt += f"\n\n[Your current emotional state is: {current_emotion.name}. Let this state subtly influence your response style and word choice.]"
        if memory_context and "No memories" not in memory_context:
            system_prompt += f"\n[Memory Context]:\n{memory_context}"

        system_tokens = self.llm.tokenize(system_prompt.encode("utf-8"))
        max_response_tokens = LLM_MAX_RESPONSE_TOKENS
        token_limit = N_CTX - len(system_tokens) - max_response_tokens

        history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"))) for m in messages)
        while history_tokens > token_limit and len(messages) > 1:
            print("   (Trimming conversation history to fit context window...)")
            messages.pop(0)
            history_tokens = sum(len(self.llm.tokenize(m["content"].encode("utf-8"))) for m in messages)

        full_prompt = [{"role": "system", "content": system_prompt}] + messages

        try:
            response = await asyncio.to_thread(
                self.llm.create_chat_completion,
                messages=full_prompt,
                max_tokens=LLM_MAX_RESPONSE_TOKENS,
                temperature=0.8,
                top_p=0.9,
                stop=["\nJonny:", "\nKira:", "</s>"]
            )
            raw_text = response['choices'][0]['message']['content']
            return self._clean_response(raw_text)
        except Exception as e:
            print(f"   ERROR during LLM inference: {e}")
            return "Oops, my brain just short-circuited. What were we talking about?"

    async def analyze_emotion(self, last_user_text: str, last_ai_response: str) -> EmotionalState | None:
        if not self.llm:
            return None
        emotion_names = [e.name for e in EmotionalState]
        prompt = (
            f"mumeinosato: \"{last_user_text}\"\nKira: \"{last_ai_response}\"\n\n"
            f"Based on this, which emotional state is most appropriate for Kira's next turn? "
            f"Options: {', '.join(emotion_names)}.\n"
            f"Respond ONLY with the single best state name (e.g., 'SASSY')."
        )
        try:
            response = await asyncio.to_thread(
                self.llm,
                prompt=prompt,
                max_tokens=10,
                temperature=0.2,
                stop=["\n", ".", ","]
            )
            text_response = response['choices'][0]['text'].strip().upper()
            for emotion in EmotionalState:
                if emotion.name in text_response:
                    return emotion
            return None
        except Exception as e:
            print(f"   ERROR during emotion analysis: {e}")
            return None

    def _clean_response(self, text: str) -> str:
        text = re.sub(r'^\s*Kira:\s*', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = text.replace('</s>', '').strip()
        text = text.replace('*', '')
        return text
