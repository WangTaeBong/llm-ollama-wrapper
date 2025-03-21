"""
Ollama LLM service implementation.

This module provides the implementation of the LLM service interface
for the Ollama backend, supporting both local and remote Ollama instances.
"""

import asyncio
import inspect
import logging
import time
from typing import Any, List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from src.common.config_loader import ConfigLoader
from src.services.llm.base import LLMServiceBase
from src.services.utils.async_helpers import limited_task_semaphore

# Load application settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)

# Semaphore to limit concurrent tasks
_semaphore = asyncio.Semaphore(settings.cache.max_concurrent_tasks)


class OllamaService(LLMServiceBase):
    """
    LLM service implementation for Ollama backend.

    Handles interactions with Ollama LLM models, including initialization,
    chain management, and query processing.
    """

    def __init__(self, response_generator):
        """
        Initialize the Ollama service.

        Args:
            response_generator: Generator for responses based on LLM outputs
        """
        self.llm_model = None
        self.response_generator = response_generator
        self.timeout = getattr(settings.llm, 'timeout', 60)

        # For performance monitoring
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0
        }

    async def initialize(self):
        """
        Initialize the Ollama LLM model.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        try:
            # Validate settings
            if not settings.ollama.model_name:
                raise ValueError("Ollama model name not configured")

            if settings.ollama.access_type.lower() == 'url' and not settings.ollama.ollama_url:
                raise ValueError("Ollama URL not configured for URL access method")

            # Initialize Ollama with URL or local access
            if settings.ollama.access_type.lower() == 'url':
                logger.info(f"Initializing Ollama LLM with URL: {settings.ollama.ollama_url}")
                self.llm_model = OllamaLLM(
                    base_url=settings.ollama.ollama_url,
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )
            else:  # Local access
                logger.info(f"Initializing Ollama LLM with local access: {settings.ollama.model_name}")
                self.llm_model = OllamaLLM(
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )

            logger.info("Ollama LLM initialization complete")
            return True

        except Exception as err:
            logger.error(f"Ollama LLM initialization failed: {err}", exc_info=True)
            return False

    async def initialize_chain(self, settings_key, prompt_template):
        """
        Initialize or retrieve a chain for the Ollama model.

        Args:
            settings_key (str): Unique key for the chain.
            prompt_template: Prompt template for the chain.

        Returns:
            chain: The initialized chain.
        """
        try:
            chat_prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = create_stuff_documents_chain(self.llm_model, chat_prompt)
            return chain
        except Exception as e:
            logger.error(f"Chain initialization failed: {str(e)}", exc_info=True)
            raise

    async def ask(self, query: str, documents: List[Any], language: str) -> str:
        """
        Send a query to the Ollama LLM with the given context.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Returns:
            str: Generated response.
        """
        start_time = time.time()

        try:
            # Get prompt template
            prompt_template = self.response_generator.get_rag_qa_prompt("default")

            # Initialize chain
            chain = await self.initialize_chain("default", prompt_template)

            # Prepare context
            context = {
                "input": query,
                "context": documents,
                "language": language,
                "today": self.response_generator.get_today(),
            }

            # Invoke chain with concurrency control
            result = await limited_task_semaphore(_semaphore, chain.invoke, context, timeout=self.timeout)

            # Handle coroutine result if needed
            if inspect.iscoroutine(result):
                result = await result

            elapsed = time.time() - start_time
            logger.info(f"LLM query complete: {elapsed:.4f}s elapsed [backend=ollama]")

            # Update metrics
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"LLM query failed: {elapsed:.4f}s after: {type(e).__name__}: {str(e)}", exc_info=True)

            self.metrics["error_count"] += 1
            raise

    async def stream_response(self, query: str, documents: List[Any], language: str):
        """
        Stream a response from the Ollama LLM.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Returns:
            AsyncGenerator: Generator yielding response chunks.
        """
        # Ollama doesn't natively support streaming through LangChain
        # Return an error message
        yield {
            "error": True,
            "message": "Streaming is only supported with vLLM backend"
        }

    def get_metrics(self):
        """
        Get performance metrics for the Ollama service.

        Returns:
            dict: Service metrics.
        """
        avg_time = 0
        if self.metrics["request_count"] > 0:
            avg_time = self.metrics["total_time"] / self.metrics["request_count"]

        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "avg_response_time": avg_time,
            "total_time": self.metrics["total_time"]
        }
