# services/history/__init__.py
"""
Conversation history management system.

This package provides a comprehensive system for managing conversation
history with large language models, supporting various storage backends,
multiple model types, and advanced features like streaming.
"""

from .formatter import HistoryFormatter
from .manager import HistoryManager
from .prompt_handler import HistoryPromptHandler
from .storage import HistoryStorage
from .streaming_handler import StreamingHistoryHandler
from .vllm_handler import VLLMHistoryHandler


class HistoryService:
    """
    Entry point for conversation history functionality.

    Provides a simplified interface to the history management system
    with factories for creating appropriate handlers.
    """

    @classmethod
    async def create_manager(cls, llm_model, request, max_history_turns=10):
        """
        Create a history manager instance.

        Args:
            llm_model: The LLM model to use
            request: The chat request
            max_history_turns: Maximum number of conversation turns to maintain

        Returns:
            HistoryManager: The history manager instance
        """
        return HistoryManager(llm_model, request, max_history_turns)

    @classmethod
    async def create_vllm_handler(cls, history_manager):
        """
        Create a vLLM history handler.

        Args:
            history_manager: The history manager instance

        Returns:
            VLLMHistoryHandler: The vLLM history handler
        """
        return VLLMHistoryHandler(history_manager)

    @classmethod
    async def create_streaming_handler(cls, vllm_handler):
        """
        Create a streaming history handler.

        Args:
            vllm_handler: The vLLM history handler instance

        Returns:
            StreamingHistoryHandler: The streaming history handler
        """
        return StreamingHistoryHandler(vllm_handler)

    @classmethod
    async def save_history(cls, session_id, rag_sys_info, user_message, ai_response):
        """
        Save a conversation exchange to history storage.

        Args:
            session_id: The session identifier
            rag_sys_info: System information
            user_message: The user's message
            ai_response: The AI's response

        Returns:
            bool: True if successful, False otherwise
        """
        return await HistoryStorage.save_chat_history(
            session_id,
            rag_sys_info,
            user_message,
            ai_response
        )
