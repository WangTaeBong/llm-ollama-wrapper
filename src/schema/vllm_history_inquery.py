"""
Chat Message Models for LLM Conversation History

This module defines Pydantic models for handling chat message structures and
conversation history in a large language model (LLM) system. These models
facilitate the passing of conversation context between client and server.

Classes:
    ChatMessage: Individual message model with role and content
    VllmHistoryInquery: Request model for processing messages with conversation history
"""

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """
    Represents a single message in a conversation.

    This model captures both the role of the sender and the message content,
    allowing for structured conversation representation in chat-based systems.

    Attributes:
        role (str): The role of the entity sending the message.
                   Common values include "user", "assistant", "system".
        message (str): The actual text content of the message.
    """
    role: str = Field(
        ...,
        description="Role of the message sender (e.g., 'user', 'assistant', 'system')"
    )
    message: str = Field(
        ...,
        description="Text content of the message"
    )

    @model_validator(mode='after')
    def validate_role(self) -> 'ChatMessage':
        """
        Validates that the role field contains an acceptable value.

        Returns:
            ChatMessage: The validated model instance

        Raises:
            ValueError: If role is not one of the standard values
        """
        standard_roles = ["user", "assistant", "system", "function", "tool"]
        if self.role.lower() not in standard_roles:
            raise ValueError(
                f"Role '{self.role}' is not one of the standard roles: {', '.join(standard_roles)}. "
                f"If this is intentional, you may ignore this validation."
            )
        return self


class VllmHistoryInquery(BaseModel):
    """
    Request model for processing chat messages with conversation history.

    This model represents a client request to the LLM system, containing the current
    user message and optional conversation history for context preservation.

    Attributes:
        session_id (str): Unique identifier for the conversation session.
        message (str): Current user input message to be processed.
        history (Optional[List[ChatMessage]]): Previous messages in the conversation.
                                              Defaults to an empty list if None.
    """
    session_id: str = Field(
        ...,
        description="Unique identifier for the conversation session"
    )
    message: str = Field(
        ...,
        description="Current user input message to be processed"
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages in the conversation"
    )

    @classmethod
    @model_validator(mode='before')
    def ensure_history_list(cls, data: dict) -> dict:
        """
        Ensures that history is always a list, converting None to an empty list.

        Args:
            data (dict): The raw input data

        Returns:
            dict: The processed input data with history as a list
        """
        if isinstance(data, dict) and data.get('history') is None:
            data['history'] = []
        return data

    def get_full_history(self) -> List[ChatMessage]:
        """
        Returns the complete conversation history including the current message.

        This method creates a new list with all previous messages plus the
        current message formatted as a user message.

        Returns:
            List[ChatMessage]: Complete conversation history with current message
        """
        # Create a copy of the existing history
        full_history = list(self.history)

        # Add the current message as a user message
        full_history.append(ChatMessage(role="user", message=self.message))

        return full_history

    def get_message_count(self) -> int:
        """
        Gets the total number of messages in the conversation, including the current message.

        Returns:
            int: Total message count
        """
        return len(self.history) + 1  # +1 for the current message
