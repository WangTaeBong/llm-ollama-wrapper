"""
Document retrieval service implementation.

This module provides the document retrieval service for obtaining
context documents for LLM queries, supporting various document sources.
"""

import logging
import time

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest

# Load application settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class RetrieverService:
    """
    Service to retrieve documents from various sources.

    Handles document retrieval and processing for chat context.
    """

    def __init__(self, request: ChatRequest, document_processor):
        """
        Initialize the retrieval service.

        Args:
            request (ChatRequest): Chat request instance.
            document_processor: Processor for document conversion.
        """
        self.request = request
        self.documents = []
        self.document_processor = document_processor
        self.start_time = time.time()  # For performance monitoring

    def retrieve_documents(self):
        """
        Convert payload to documents for retrieval.

        Returns:
            list: Retrieved documents.

        Raises:
            Exception: If document retrieval fails.
        """
        start_time = time.time()
        session_id = self.request.meta.session_id

        try:
            logger.debug(f"[{session_id}] Starting document retrieval")
            self.documents = self.document_processor.convert_payload_to_document(self.request)

            elapsed = time.time() - start_time
            logger.debug(
                f"[{session_id}] Retrieved {len(self.documents)} documents - {elapsed:.4f}s elapsed"
            )
            return self.documents
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{session_id}] Error during document retrieval: {str(e)} - {elapsed:.4f}s elapsed",
                exc_info=True
            )
            raise

    async def add_web_search_results(self):
        """
        Add web search results to documents if configured.

        Returns:
            list: Documents with web search results added.
        """
        if not getattr(settings, 'web_search', {}).get('enabled', False):
            return self.documents

        start_time = time.time()
        session_id = self.request.meta.session_id

        try:
            # Implementation for web search integration
            # Placeholder for future web search implementation
            elapsed = time.time() - start_time
            logger.debug(f"[{session_id}] Added web search results - {elapsed:.4f}s elapsed")

            return self.documents
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{session_id}] Error adding web search results: {str(e)} - {elapsed:.4f}s elapsed",
                exc_info=True
            )
            # Return existing documents even if web search fails
            return self.documents

    def get_performance_metrics(self):
        """
        Get performance metrics for the retrieval service.

        Returns:
            dict: Performance metrics.
        """
        end_time = time.time()
        return {
            "total_time": end_time - self.start_time,
            "document_count": len(self.documents) if self.documents else 0
        }
