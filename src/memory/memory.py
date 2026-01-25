import chromadb
import torch
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time
import uuid
from config import MEMORY_PATH

class MemoryManager:
    def __init__(self, collection_name="conversation_memory"):
        print("-> Initializing Memory Manager...")
        if not os.path.exists(MEMORY_PATH):
            os.makedirs(MEMORY_PATH)

        self.client = chromadb.PersistentClient(
            path=MEMORY_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = 'xpu'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        #self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        print("   Memory Manager initialized.")

    def add_memory(self, user_text: str, ai_text: str):
        """Adds a new raw conversation turn to the memory."""
        try:
            self.collection.add(
                embeddings=[
                    self.embedding_model.encode(user_text).tolist(),
                    self.embedding_model.encode(ai_text).tolist()
                ],
                documents=[user_text, ai_text],
                metadatas=[
                    {"role": "user", "timestamp": time.time(), "type": "turn"},
                    {"role": "assistant", "timestamp": time.time(), "type": "turn"}
                ],
                ids=[str(uuid.uuid4()), str(uuid.uuid4())]
            )
        except Exception as e:
            print(f"   ERROR: Failed to add raw memory turn: {e}")

    def add_summarized_memory(self, summary_text: str):
        """Adds a new high-level, summarized memory to the database."""
        try:
            self.collection.add(
                embeddings=[self.embedding_model.encode(summary_text).tolist()],
                documents=[summary_text],
                metadatas=[{"role": "summary", "timestamp": time.time(), "type": "summary"}],
                ids=[str(uuid.uuid4())]
            )
            print(f"   ✅ Consolidated Memory Added: '{summary_text}'")
        except Exception as e:
            print(f"   ERROR: Failed to add summarized memory: {e}")

    def add_knowledge(self, content: str, source: str = "web_search"):
        """Adds external knowledge to the memory."""
        try:
            # Simple chunking if the content is very long
            max_chunk_size = 1000
            chunks = [content[i:i+max_chunk_size] for i in range(0, len(content), max_chunk_size)]
            
            for chunk in chunks:
                self.collection.add(
                    embeddings=[self.embedding_model.encode(chunk).tolist()],
                    documents=[chunk],
                    metadatas=[{"role": "knowledge", "timestamp": time.time(), "type": "knowledge", "source": source}],
                    ids=[str(uuid.uuid4())]
                )
            print(f"   ✅ Knowledge Added from {source} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"   ERROR: Failed to add knowledge: {e}")

    def search_memories(self, query_text: str, n_results: int = 5) -> str:
        """Searches for memories semantically similar to the query text."""
        if self.collection.count() == 0:
            return "No memories yet."
        try:
            results = self.collection.query(
                query_embeddings=[self.embedding_model.encode(query_text).tolist()],
                n_results=min(n_results, self.collection.count()),
                include=['documents', 'metadatas']
            )

            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    mem_type = meta.get('type', 'turn')
                    
                    if mem_type == 'summary':
                        formatted_results.append(f"[過去の要約]: {doc}")
                    elif mem_type == 'knowledge':
                        formatted_results.append(f"[外部知識]: {doc}")
                    else:
                        role = meta.get('role', 'unknown').capitalize()
                        formatted_results.append(f"[{role}の過去の発言]: {doc}")

            return "\n- ".join(formatted_results) if formatted_results else "No highly relevant memories found."
        except Exception as e:
            print(f"   ERROR: Failed to search memories: {e}")
            return "I had a problem searching my memory."
