"""
Response Model Definitions for Document Management System

This module defines Pydantic models for handling document-related responses in a
Retrieval-Augmented Generation (RAG) system. It provides validation, logging capabilities,
and type checking for responses containing document metadata, content, and system information.

Classes:
    PayloadRes: Document payload information model for responses
    ChatRes: Chat response information model
    MetaRes: Document metadata model for responses
    ChatResponse: Complete chat response model with logging capabilities
"""

import json
import logging
import re
from typing import List, Optional

from pydantic import BaseModel, field_validator, Field


class PayloadRes(BaseModel):
    """
    Represents the payload of a document in system responses.

    This model validates document information including name, page numbers, and content
    that will be sent back to the client in responses.

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


class ChatRes(BaseModel):
    """
    Represents a chat response from the document management system.

    This model captures user information, system response data, categorization,
    and optional document payloads included in the response.

    Attributes:
        user (str): The identifier for the user who made the request.
        system (str): The system's response identifier or message.
                      Defaults to "system_default".
        category1 (Optional[str]): Primary categorization label (optional).
        category2 (Optional[str]): Secondary categorization label (optional).
        category3 (Optional[str]): Tertiary categorization label (optional).
        info (Optional[List[PayloadRes]]): List of document payloads providing 
                                           additional information (optional).
    """
    user: str = Field(..., description="User identifier")
    system: str = Field(default="system_default", description="System response identifier")
    category1: Optional[str] = Field(default=None, description="Primary category")
    category2: Optional[str] = Field(default=None, description="Secondary category")
    category3: Optional[str] = Field(default=None, description="Tertiary category")
    info: Optional[List[PayloadRes]] = Field(default=None, description="Document payloads with additional information")


class MetaRes(BaseModel):
    """
    Represents metadata about the document in responses.

    This model captures organizational and system-level information to be
    included in responses, including identification and session tracking details.

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


class ChatResponse(BaseModel):
    """
    Master model representing a complete chat response from the document system.

    This model combines result status, metadata, and chat information into a single
    response object, enforcing validation across all components while providing
    logging capabilities for debugging and auditing purposes.

    Attributes:
        result_cd (int): The result code indicating the status of the response.
                         Must be a non-negative integer.
        result_desc (str): A description of the result, typically used for 
                           error messages or status information.
        meta (MetaRes): System and organizational metadata.
        chat (ChatRes): Chat response details including user information and document payloads.

    Methods:
        log_response(): Logs response details at appropriate logging levels.

    Configuration:
        The model is configured to forbid extra fields, ensuring strict schema compliance.
    """
    result_cd: int = Field(..., description="Result code (non-negative integer)")
    result_desc: str = Field(..., description="Result description or error message")
    meta: MetaRes = Field(..., description="System and organizational metadata")
    chat: ChatRes = Field(..., description="Chat response details and payloads")
    performance_data: Optional[dict] = Field(default=None, description="Performance metrics and timing data")

    @classmethod
    @field_validator("result_cd")
    def validate_result_cd(cls, value: int) -> int:
        """
        Validates that the result code is a non-negative integer.

        Args:
            value (int): The result code to validate.

        Returns:
            int: The validated result code.

        Raises:
            ValueError: If the result code is negative.
        """
        if value < 0:
            raise ValueError("result_cd must be a non-negative integer.")
        return value

    def add_performance_data(self, data: dict) -> 'ChatResponse':
        """
        Adds or updates performance metrics and timing data to the response.

        This method allows for tracking and analyzing the performance of various
        processing stages in the chat response generation pipeline.

        Args:
            data (dict): Dictionary containing performance metrics and timing data.
                         May include keys such as:
                         - total_processing_time: Total time taken to process the request
                         - processing_stages: Breakdown of time spent in each processing stage
                         - llm_metrics: Metrics related to the language model processing

        Returns:
            ChatResponse: Returns self for method chaining.

        Example:
            response.add_performance_data({
                "total_processing_time": 0.856,
                "processing_stages": {
                    "document_retrieval": 0.234,
                    "llm_processing": 0.512,
                    "post_processing": 0.110
                },
                "llm_metrics": {
                    "token_count": 156,
                    "model_name": "gpt-3.5-turbo"
                }
            })
        """
        if self.performance_data is None:
            self.performance_data = {}

        # Update with new performance data
        self.performance_data.update(data)
        return self

    def log_response(self, max_content_length: int = 200) -> None:
        """
        Logs the response details for debugging and auditing purposes in an optimized manner.

        This method selectively logs response data based on logging level to minimize
        performance impact and log file size while ensuring critical information is captured.

        Args:
            max_content_length (int, optional): Maximum length for content strings in logs.
                Defaults to 200 characters.

        Features:
        - Only serializes detailed JSON when DEBUG level is enabled
        - Truncates long content to prevent log bloat
        - Masks potentially sensitive information
        - Uses structured logging pattern for easier log parsing
        """
        # Extract key identifiers for log context
        session_id = self.meta.session_id or "no_session"
        rag_sys_info = self.meta.rag_sys_info

        # Always log basic response information at INFO level
        logging.info(f"Response[{rag_sys_info}/{session_id}] Code: {self.result_cd}, Description: {self.result_desc}")

        # Log performance data if available
        if self.performance_data and logging.getLogger().isEnabledFor(logging.INFO):
            total_time = self.performance_data.get("total_processing_time", 0)
            logging.info(f"Performance[{session_id}]: Total processing time: {total_time:.4f}s")

        # Only proceed with detailed logging if DEBUG level is enabled
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            # For metadata, include only non-sensitive fields
            safe_meta = {
                "company_id": self.meta.company_id,
                "dept_class": self.meta.dept_class,
                "rag_sys_info": self.meta.rag_sys_info
            }

            # For chat data, truncate long content fields
            safe_chat = self.chat.model_dump(exclude={"info"})

            # Process payload info items if they exist
            if self.chat.info:
                truncated_info = []
                for item in self.chat.info:
                    # Truncate long content with indicator if needed
                    content = item.content
                    if len(content) > max_content_length:
                        content = f"{content[:max_content_length]}... [truncated, {len(content)} chars total]"

                    truncated_info.append({
                        "doc_name": item.doc_name,
                        "doc_page": item.doc_page,
                        "content": content
                    })
                safe_chat["info"] = truncated_info

            # Log metadata and chat info as compact JSON to save space
            meta_json = json.dumps(safe_meta, ensure_ascii=False)
            chat_json = json.dumps(safe_chat, ensure_ascii=False)

            logging.debug(f"Metadata[{session_id}]: {meta_json}")
            logging.debug(f"ChatData[{session_id}]: {chat_json}")

            # Log document count and total content size at TRACE level if available
            if self.chat.info:
                total_size = sum(len(item.content) for item in self.chat.info)
                logging.debug(f"Documents[{session_id}]: count={len(self.chat.info)}, total_size={total_size} chars")

            # Log detailed performance data if available
            if self.performance_data:
                perf_json = json.dumps(self.performance_data, ensure_ascii=False)
                logging.debug(f"PerformanceDetails[{session_id}]: {perf_json}")

    class Config:
        """
        Pydantic configuration settings for the ChatResponse model.

        Controls model behavior regarding extra fields, validation, and serialization.
        """
        extra = "forbid"  # Reject any fields not explicitly defined in the model
        validate_assignment = True  # Validate values during attribute assignment
        arbitrary_types_allowed = False  # Disallow arbitrary types for strict type checking
