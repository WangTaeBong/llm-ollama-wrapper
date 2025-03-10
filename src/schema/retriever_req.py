"""
Retriever Model Definitions for RAG System

This module defines Pydantic models used for document retrieval requests in a
Retrieval-Augmented Generation (RAG) system. These models provide validation,
type checking, and example schemas for requests containing metadata and query information.

Classes:
    RetrieverMeta: Request metadata model with company and system information
    RetrieverQuery: User query model with categorization capabilities
    RetrieverRequest: Complete retrieval request model combining metadata and query
"""

from typing import Optional

from pydantic import BaseModel, field_validator, Field, model_validator


class RetrieverMeta(BaseModel):
    """
    Metadata for document retrieval requests providing organizational and system context.

    This model captures the necessary context information for processing retrieval
    requests, including company identification, department structure, and session tracking.

    Attributes:
        company_id (str): Organizational identifier for the company initiating the request.
        dept_class (Optional[str]): Hierarchical department classification string.
                                   Format: "dept1_dept2_dept3" for nested departments.
        rag_sys_info (str): Information about the RAG system backend (e.g., VectorDB metadata).
                           Limited to 100 characters maximum.
        session_id (Optional[str]): User session tracking identifier if available.
    """
    company_id: str = Field(..., description="Organizational identifier")
    dept_class: Optional[str] = Field(default=None, description="Department hierarchy (dept1_dept2_dept3)")
    rag_sys_info: str = Field(..., description="RAG system backend information (max 100 chars)")
    session_id: Optional[str] = Field(default=None, description="Session tracking identifier")

    @classmethod
    @field_validator('rag_sys_info')
    def validate_rag_sys_info(cls, value: str) -> str:
        """
        Validates the RAG system information field.

        Ensures that the rag_sys_info field:
        1. Is not empty or whitespace-only
        2. Does not exceed the maximum length of 100 characters

        Args:
            value (str): The RAG system information string to validate

        Returns:
            str: The validated RAG system information

        Raises:
            ValueError: If the value is empty/whitespace or exceeds length limit
        """
        # Check for empty or whitespace-only strings
        if not value or not value.strip():
            raise ValueError("rag_sys_info cannot be empty or contain only whitespace")

        # Check length constraint
        if len(value) > 100:
            raise ValueError(f"rag_sys_info exceeds maximum length (100 chars): got {len(value)} chars")

        return value.strip()

    @model_validator(mode='after')
    def validate_dept_class_format(self) -> 'RetrieverMeta':
        """
        Validates the format of the department classification if provided.

        Ensures that if dept_class is provided, it follows the expected
        underscore-separated hierarchical format.

        Returns:
            RetrieverMeta: The validated model instance

        Raises:
            ValueError: If dept_class has invalid format
        """
        dept_class = self.dept_class
        if dept_class:
            # Verify format follows dept1_dept2_dept3 pattern
            if not all(part.strip() for part in dept_class.split('_')):
                raise ValueError("dept_class must contain non-empty department names separated by underscores")
        return self


class RetrieverQuery(BaseModel):
    """
    User query model for document retrieval requests.

    This model captures the user's query text and optional categorization
    information to help with retrieval context and filtering.

    Attributes:
        user (str): The query text provided by the user for document retrieval.
        category1 (Optional[str]): Primary category classification for the query.
        category2 (Optional[str]): Secondary category classification for the query.
        category3 (Optional[str]): Tertiary category classification for the query.
    """
    user: str = Field(..., description="User query text")
    category1: Optional[str] = Field(default=None, description="Primary category")
    category2: Optional[str] = Field(default=None, description="Secondary category")
    category3: Optional[str] = Field(default=None, description="Tertiary category")

    @model_validator(mode='after')
    def validate_categories_hierarchy(self) -> 'RetrieverQuery':
        """
        Validates that categories follow a proper hierarchical structure.

        Ensures that if a lower-level category is specified, the higher-level
        categories must also be specified (e.g., if category2 is present,
        category1 must also be present).

        Returns:
            RetrieverQuery: The validated model instance

        Raises:
            ValueError: If categories do not follow proper hierarchy
        """
        # Check that category hierarchy is properly maintained
        if self.category2 and not self.category1:
            raise ValueError("category1 must be specified when category2 is present")
        if self.category3 and not self.category2:
            raise ValueError("category2 must be specified when category3 is present")

        return self


class RetrieverRequest(BaseModel):
    """
    Complete model for document retrieval requests.

    This model combines metadata and query information into a unified
    request schema with examples to guide API users.

    Attributes:
        meta (RetrieverMeta): Request metadata with company and system information.
        chat (RetrieverQuery): User query details with optional categorization.

    Configuration:
        Model schema includes comprehensive examples showing different
        valid request structures.
    """
    meta: RetrieverMeta = Field(..., description="Request metadata with system context")
    chat: RetrieverQuery = Field(..., description="User query with optional categorization")

    model_config = {
        """
        Configuration for the Pydantic model with schema examples.

        Provides example JSON structures to help API consumers understand
        the expected request format and variations.
        """
        "json_schema_extra": {
            "examples": [
                {
                    "meta": {
                        "company_id": "mico",
                        "dept_class": "dept1_dept2_dept3",
                        "rag_sys_info": "vectorDB info",
                        "session_id": "session_id"
                    },
                    "chat": {
                        "user": "user query",
                        "category1": "category1",
                        "category2": "category2",
                        "category3": "category3"
                    }
                },
                {
                    "meta": {
                        "company_id": "test_company",
                        "rag_sys_info": "otherDB info",
                        "session_id": None
                    },
                    "chat": {
                        "user": "another query",
                        "category1": None,
                        "category2": None,
                        "category3": None
                    }
                },
                {
                    "meta": {
                        "company_id": "example_company",
                        "rag_sys_info": "exampleDB info",
                        "session_id": "example_session"
                    },
                    "chat": {
                        "user": "query without categories",
                        "category1": None,
                        "category2": None,
                        "category3": None
                    }
                }
            ]
        }
    }
