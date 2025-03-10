from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any


@lru_cache
def default_timestamp() -> str:
    """
    Generate a default timestamp in ISO 8601 format.

    Returns:
        str: The current datetime in ISO 8601 format.
    """
    return datetime.now().isoformat()


def generate_redis_key(*parts: str) -> str:
    """
    Generate a Redis key by concatenating parts with a colon (':').

    Args:
        parts (str): Components of the Redis key.

    Returns:
        str: The generated Redis key.
    """
    return ":".join(parts)


def create_chat_data(chat_id: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create chat data in JSON format for storage or transmission.

    Args:
        chat_id (str): Unique identifier for the chat session.
        messages (List[Dict[str, Any]]): List of messages in the chat session.

    Returns:
        Dict[str, Any]: JSON object representing the chat session, including messages.
    """
    return {
        "chat_id": chat_id,
        "messages": [
            {
                "role": message.get("role"),
                "content": message.get("content"),
                "timestamp": message.get("timestamp", default_timestamp())
            }
            for message in messages
        ]
    }


def create_message(role: str, content: str, timestamp: str = None) -> Dict[str, str]:
    """
    Create a single message dictionary for use in chat data.

    Args:
        role (str): The role of the message sender (e.g., "HumanMessage" or "AIMessage").
        content (str): The content of the message.
        timestamp (str, optional): The timestamp of the message. Defaults to the current time.

    Returns:
        Dict[str, str]: A dictionary representing the message.
    """
    return {
        "role": role,
        "content": content,
        "timestamp": timestamp or default_timestamp()
    }
