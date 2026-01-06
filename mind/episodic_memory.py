"""
Path: mind/episodic_memory.py
Role: Manages the Digital Journal (User History).
"""

from .vector_store import AetherVectorStore
import datetime

class EpisodicMemory:
    def __init__(self, vector_store: AetherVectorStore):
        self.store = vector_store

    def record_interaction(self, user_id: str, role: str, content: str):
        """
        Saves a single turn of conversation into the user's private namespace.
        """
        metadata = {
            "user_id": user_id,
            "role": role,
            "timestamp": str(datetime.datetime.now())
        }
        namespace = f"user_{user_id}_episodic"
        self.store.upsert_knowledge(content, namespace, metadata)

    def get_recent_context(self, user_id: str, current_query: str):
        """
        Retrieves relevant past interactions to provide context for the Brain.
        """
        namespace = f"user_{user_id}_episodic"
        matches, _ = self.store.query_context(current_query, namespace, top_k=3, include_metadata=True)

        formatted_contexts = []
        for m in matches:
            ts = m.get('timestamp', 'Unknown Time')
            text = m.get('text', '')
            formatted_contexts.append(f"[{ts}] {text}")

        return formatted_contexts