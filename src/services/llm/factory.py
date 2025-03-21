"""
LLM service factory implementation.

This module provides a factory for creating appropriate LLM service
instances based on configuration settings.
"""

import logging
from typing import Optional

from src.common.config_loader import ConfigLoader
from src.services.llm.base import LLMServiceBase
from src.services.llm.ollama import OllamaService
from src.services.llm.vllm import VLLMService

# Load application settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """
    Factory for creating LLM service instances.

    Creates and initializes the appropriate LLM service implementation
    based on configuration settings.
    """

    @staticmethod
    async def create_llm_service(response_generator) -> Optional[LLMServiceBase]:
        """
        Create and initialize an LLM service instance.

        Args:
            response_generator: Generator for responses based on LLM outputs

        Returns:
            LLMServiceBase: The initialized LLM service or None if initialization fails
        """
        try:
            # Validate configuration
            if not hasattr(settings, 'llm') or not hasattr(settings.llm, 'llm_backend'):
                logger.error("LLM backend not configured")
                return None

            backend = settings.llm.llm_backend.lower()

            # Create appropriate service based on backend
            if backend == 'ollama':
                logger.info("Creating Ollama LLM service")
                service = OllamaService(response_generator)
            elif backend == 'vllm':
                logger.info("Creating vLLM service")
                service = VLLMService(response_generator)
            else:
                logger.error(f"Unsupported LLM backend: {backend}")
                return None

            # Initialize service
            success = await service.initialize()
            if not success:
                logger.error(f"Failed to initialize {backend} service")
                return None

            return service

        except Exception as e:
            logger.error(f"Error creating LLM service: {e}", exc_info=True)
            return None
