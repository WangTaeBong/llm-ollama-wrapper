"""
VLLM Inquiry Model for LLM Query Processing

This module defines a Pydantic model for handling basic inquiry requests to a
VLLM (Very Large Language Model) system. It provides a structured schema for
client applications to submit text processing requests to the model.

Classes:
    VllmInquery: Basic inquiry request model with session tracking
"""

from pydantic import BaseModel, Field, model_validator
from typing import List, Optional


class VllmInquery(BaseModel):
    """
    Represents a basic inquiry request to the VLLM system.

    This model defines the minimum required information for processing
    a text query, including session identification for tracking and 
    the text content to be processed by the language model.

    Attributes:
        session_id (str): Unique identifier for tracking the client session.
        vllm_inquery (str): The text query to be processed by the language model.
    """
    request_id: str
    prompt: str
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    stream: bool = Field(default=False)
    stop: Optional[List[str]] = Field(default=None)

    @model_validator(mode='after')
    def validate_fields(self) -> 'VllmInquery':
        """
        Validates that the fields contain meaningful values.

        Ensures that both session_id and vllm_inquery contain non-empty
        and non-whitespace-only strings.

        Returns:
            VllmInquery: The validated model instance

        Raises:
            ValueError: If any field fails validation
        """
        # Validate session_id is not just whitespace
        if not self.request_id.strip():
            raise ValueError("session_id cannot be empty or contain only whitespace")

        # Validate inquiry is not just whitespace
        if not self.prompt.strip():
            raise ValueError("vllm_inquery cannot be empty or contain only whitespace")

        return self

    def get_inquery_length(self) -> int:
        """
        Gets the character count of the inquiry text.

        This helper method is useful for logging or determining 
        if the inquiry meets length requirements.

        Returns:
            int: Number of characters in the inquiry text
        """
        return len(self.prompt)

    def get_truncated_inquery(self, max_length: int = 50) -> str:
        """
        Gets a truncated version of the inquiry text for logging or display.

        Args:
            max_length (int): Maximum length of the returned string.
                             Defaults to 50 characters.

        Returns:
            str: Truncated inquiry text with ellipsis if necessary
        """
        if len(self.prompt) <= max_length:
            return self.prompt

        return f"{self.prompt[:max_length]}..."

    def to_log_string(self) -> str:
        """
        Creates a formatted string for logging this inquiry.

        Returns:
            str: Formatted string containing session ID and truncated inquiry
        """
        truncated = self.get_truncated_inquery()
        return f"Session[{self.request_id}]: {truncated}"
