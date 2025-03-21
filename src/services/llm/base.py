"""
Base interfaces and classes for LLM services.

This module defines the core abstractions for working with large language
models, including model initialization, chain management, and query processing.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.common.config_loader import ConfigLoader

# Load application settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class LLMServiceBase(ABC):
    """
    Base abstract class for LLM service implementations.

    Defines the interface that all LLM service implementations must follow,
    regardless of the underlying model or backend.
    """

    @abstractmethod
    async def initialize(self):
        """
        Initialize the LLM service.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        pass

    @abstractmethod
    async def ask(self, query: str, documents: List[Any], language: str) -> str:
        """
        Send a query to the LLM with the given context.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Returns:
            str: Generated response.
        """
        pass

    @abstractmethod
    async def stream_response(self, query: str, documents: List[Any], language: str):
        """
        Stream a response from the LLM.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Returns:
            AsyncGenerator: Generator yielding response chunks.
        """
        pass

    @classmethod
    def is_gemma_model(cls) -> bool:
        """
        Check if the current model is a Gemma model.

        Returns:
            bool: True if the model is Gemma, False otherwise.
        """
        # Check LLM backend
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # Check for Ollama backend
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # Check for vLLM backend
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # Default to False
        return False

    @classmethod
    def _format_image_data(cls, image_data: Dict[str, str]) -> str:
        """
        Format image data for inclusion in prompts.

        Args:
            image_data (Dict[str, str]): Image data (base64, URL, etc.)

        Returns:
            str: Formatted image description
        """
        if 'base64' in image_data:
            return "[Image data provided in base64 format. Please analyze the image and provide relevant information.]"
        elif 'url' in image_data:
            return f"[Image URL: {image_data.get('url')}]"
        elif 'description' in image_data:
            return f"[Image description: {image_data.get('description')}]"
        else:
            return "[Image data provided. Please analyze the image and provide relevant information.]"
