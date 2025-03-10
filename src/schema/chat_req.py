"""
Data Model Definitions for Document Management System

This module defines Pydantic models for handling document-related requests in a 
Retrieval-Augmented Generation (RAG) system. It provides validation and type checking
for requests containing document metadata, content, and user information.

Classes:
    PayloadReq: Document payload information model
    ChatReq: Chat request information model
    MetaReq: Document metadata model
    ChatRequest: Complete chat request model combining metadata and chat details
"""

import re
from typing import List, Optional

from pydantic import BaseModel, field_validator, Field


class PayloadReq(BaseModel):
    """
    Represents the payload for a document in the system.

    This model validates document information including name, page numbers, and content.
    It ensures that page references follow proper formatting conventions.

    Attributes:
        doc_name (str): The name of the document.
        doc_page (str): The page reference of the document. 
                        Valid formats include:
                        - Single page: "1"
                        - Page range: "5-6"
                        - Other identifiers (URLs, text references, etc.)
        content (str): The textual content of the document payload.
    """
    doc_name: str = Field(..., description="Document name identifier")
    doc_page: str = Field(..., description="Page reference (single number, range, or other identifier)")
    content: str = Field(..., description="Document content text")

    @classmethod
    @field_validator("doc_page", mode="before")
    def validate_doc_page(cls, value: str) -> str:
        """
        Validates the document page reference format.

        Ensures the doc_page field is properly formatted as either:
        - A single number (e.g., "1")
        - A range of numbers (e.g., "5-6") with start < end
        - Any other non-empty string reference

        Args:
            value (str): The page reference value to validate.

        Returns:
            str: The validated page reference value.

        Raises:
            ValueError: If value is empty or if a page range has invalid logic (start >= end).
        """
        # Ensure value is a non-empty string
        if not isinstance(value, str) or not value.strip():
            raise ValueError("doc_page must be a non-empty string.")

        # Define regex patterns for validation
        single_number_pattern = r"^\d+$"  # Matches "1", "42", etc.
        range_pattern = r"^\d+-\d+$"  # Matches "5-6", "10-20", etc.

        # Case 1: Single number validation
        if re.match(single_number_pattern, value):
            return value

        # Case 2: Range validation (including logical check)
        if re.match(range_pattern, value):
            start, end = map(int, value.split("-"))
            if start >= end:
                raise ValueError("In doc_page range format, start must be less than end.")
            return value

        # Case 3: Any other non-empty string reference
        return value.strip()


class ChatReq(BaseModel):
    """
    Represents a chat request within the document management system.

    This model captures user information, categorization data, and optional
    document payloads associated with the chat interaction.

    Attributes:
        lang (str): The language code for the chat. Defaults to "ko" (Korean).
        user (str): The user identifier making the request.
        category1 (Optional[str]): Primary categorization label (optional).
        category2 (Optional[str]): Secondary categorization label (optional).
        category3 (Optional[str]): Tertiary categorization label (optional).
        payload (List[PayloadReq]): List of document payloads. Defaults to empty list.
    """
    lang: str = Field(default="ko", description="Language code, defaults to Korean")
    user: str = Field(..., description="User identifier")
    category1: Optional[str] = Field(default=None, description="Primary category")
    category2: Optional[str] = Field(default=None, description="Secondary category")
    category3: Optional[str] = Field(default=None, description="Tertiary category")
    payload: List[PayloadReq] = Field(default_factory=list, description="Document payloads")

    @classmethod
    @field_validator("payload", mode="before")
    def set_default_payload(cls, value: Optional[List[PayloadReq]]) -> List[PayloadReq]:
        """
        Ensures the payload field is always a list.

        Converts None values to an empty list to prevent null reference errors
        during processing of the payload field.

        Args:
            value (Optional[List[PayloadReq]]): The input payload value or None.

        Returns:
            List[PayloadReq]: Either the original list or an empty list if None was provided.
        """
        return value or []


class MetaReq(BaseModel):
    """
    Represents metadata for a document request.

    This model captures organizational and system-level information about the 
    document request, including identification and session tracking details.

    Attributes:
        company_id (str): The organizational identifier for the company.
        dept_class (str): The department or division classification.
        rag_sys_info (str): Information about the RAG (Retrieval-Augmented Generation) system.
        session_id (Optional[str]): The session tracking identifier (optional).
    """
    company_id: str = Field(..., description="Company identifier")
    dept_class: str = Field(..., description="Department classification")
    rag_sys_info: str = Field(..., description="RAG system information")
    session_id: Optional[str] = Field(default=None, description="Session tracking identifier")


class ChatRequest(BaseModel):
    """
    Master model representing a complete chat request in the document system.

    This model combines metadata and chat information into a single request object,
    enforcing validation across all components while preventing extraneous fields.

    Attributes:
        meta (MetaReq): System and organizational metadata.
        chat (ChatReq): Chat request details including user information and document payloads.

    Configuration:
        The model is configured to forbid extra fields, ensuring strict schema compliance.
    """
    meta: MetaReq = Field(..., description="System and organizational metadata")
    chat: ChatReq = Field(..., description="Chat request details and payloads")

    class Config:
        """
        Pydantic configuration settings for the ChatRequest model.

        Controls model behavior regarding extra fields, validation, and serialization.
        """
        extra = "forbid"  # Reject any fields not explicitly defined in the model
        validate_assignment = True  # Validate values during attribute assignment
        arbitrary_types_allowed = False  # Disallow arbitrary types for strict type checking
