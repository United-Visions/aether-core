"""
AetherMind DCLA - Universal Body Component
Path: body/adapters/chat_ui.py
"""

from ..adapter_base import BodyAdapter
from loguru import logger

class ChatAdapter(BodyAdapter):
    def execute(self, intent: str) -> str:
        """
        Processes the raw text intent from the brain for chat display.
        For now, it simply returns the intent directly.
        """
        logger.info(f"ChatAdapter executing with intent: {intent[:50]}...")
        return intent