# services/history/storage.py
"""
Conversation history storage utilities.

This module provides functionality for storing and retrieving
conversation history from various storage backends.
"""

import asyncio
import logging
from typing import Dict, List, Any

from src.common.config_loader import ConfigLoader
from src.utils.redis_utils import RedisUtils

# Load settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class HistoryStorage:
    """
    Manages storage and retrieval of conversation history.

    Provides an abstraction layer over storage backends (currently Redis)
    with performance optimizations and error handling.
    """

    @staticmethod
    async def save_chat_history(session_id: str, rag_sys_info: str, user_message: str, ai_response: str) -> bool:
        """
        Save a chat exchange to history storage.

        Args:
            session_id: The session identifier
            rag_sys_info: System information for the conversation
            user_message: The user's message
            ai_response: The AI's response

        Returns:
            bool: True if successful, False otherwise
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                logger.debug(f"[{session_id}] Saving chat history to Redis (attempt {retry_count + 1})")

                # Create chat data format
                from src.services.chat_message_handler import create_chat_data, create_message
                chat_data = create_chat_data(session_id, [
                    create_message("HumanMessage", user_message),
                    create_message("AIMessage", ai_response)
                ])

                # Save to Redis
                await RedisUtils.async_save_message_to_redis(
                    system_info=rag_sys_info,
                    session_id=session_id,
                    message=chat_data
                )

                logger.debug(f"[{session_id}] Chat history saved successfully")
                return True

            except Exception as e:
                retry_count += 1
                level = "warning" if retry_count < max_retries else "error"
                logger.log(
                    logging.WARNING if level == "warning" else logging.ERROR,
                    f"[{session_id}] Failed to save chat history (attempt {retry_count}): {str(e)}",
                    exc_info=True
                )

                # Progressive delay before retry
                if retry_count < max_retries:
                    await asyncio.sleep(0.5 * retry_count)

        # All retries failed
        return False

    @staticmethod
    def get_messages_from_storage(rag_sys_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve messages from storage for a specific session.

        Args:
            rag_sys_info: System information for the conversation
            session_id: The session identifier

        Returns:
            List[Dict[str, Any]]: The retrieved messages or empty list on failure
        """
        try:
            # Fetch messages from Redis
            return RedisUtils.get_messages_from_redis(rag_sys_info, session_id)
        except Exception as e:
            logger.error(f"[{session_id}] Failed to fetch messages from Redis: {e}")
            return []
