# services/history/streaming_handler.py
"""
Streaming support for conversation history.

This module provides streaming response functionality with conversation
history integration, optimized for real-time character-by-character delivery.
"""

import asyncio
import logging
import time
from typing import Tuple

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.history.formatter import HistoryFormatter
from src.services.history.vllm_handler import VLLMHistoryHandler
from src.utils.prompt import PromptManager

# Load settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class StreamingHistoryHandler:
    """
    Handles streaming responses with conversation history integration.

    Provides optimized streaming capabilities that integrate with
    conversation history and support various model formats.
    """

    def __init__(self, vllm_handler: VLLMHistoryHandler):
        """
        Initialize the streaming history handler.

        Args:
            vllm_handler: The vLLM history handler instance
        """
        self.vllm_handler = vllm_handler

    async def handle_streaming_with_history(self, request: ChatRequest, language: str) -> Tuple[VllmInquery, list]:
        """
        Process a streaming chat request with conversation history.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (vllm_request, retrieval_document) for streaming
        """
        # Determine appropriate handler based on model type
        if self.vllm_handler.history_manager.is_gemma_model():
            logger.info(
                f"[{self.vllm_handler.history_manager.current_session_id}] Detected Gemma model, using Gemma streaming handler")
            return await self._handle_streaming_with_history_gemma(request, language)
        else:
            # Use improved streaming handler if enabled
            if getattr(settings.llm, 'use_improved_history', False):
                return await self._handle_streaming_with_history_improved(request, language)
            else:
                return await self._handle_streaming_with_history_original(request, language)

    async def _handle_streaming_with_history_original(self,
                                                      request: ChatRequest,
                                                      language: str) -> Tuple[VllmInquery, list]:
        """
        Original implementation for streaming with history.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (vllm_request, retrieval_document) for streaming
        """
        session_id = self.vllm_handler.history_manager.current_session_id

        # Get prompt template
        rag_prompt_template = self.vllm_handler.history_manager.response_generator.get_rag_qa_prompt(
            self.vllm_handler.history_manager.current_rag_sys_info
        )

        # Get retrieval documents
        logger.debug(f"[{session_id}] Retrieving documents for streaming...")
        retrieval_document = await self.vllm_handler.history_manager.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{session_id}] Retrieved {len(retrieval_document)} documents for streaming")

        # Get chat history in structured format
        session_history = self.vllm_handler.history_manager.get_session_history()
        formatted_history = HistoryFormatter.format_for_prompt(
            session_history,
            settings.llm.max_history_turns
        )
        logger.debug(f"[{session_id}] Processed {len(session_history.messages)} history messages for streaming")

        # Prepare context for prompt
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,
            "context": retrieval_document,
            "language": language,
            "today": self.vllm_handler.history_manager.response_generator.get_today(),
        }

        # Add VOC-specific context if needed
        if self.vllm_handler.history_manager.current_rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Handle image data if present
        if hasattr(request.chat, 'image') and request.chat.image:
            common_input["image_description"] = self.vllm_handler._format_image_data(request.chat.image)

        # Build system prompt
        if self.vllm_handler.history_manager.is_gemma_model():
            vllm_inquery_context = self.vllm_handler._build_system_prompt_gemma(rag_prompt_template, common_input)
        else:
            vllm_inquery_context = self.vllm_handler._build_system_prompt(rag_prompt_template, common_input)

        # Create streaming vLLM request
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # Enable streaming
        )

        logger.debug(f"[{session_id}] Streaming request prepared")
        return vllm_request, retrieval_document

    async def _handle_streaming_with_history_improved(self,
                                                      request: ChatRequest,
                                                      language: str) -> Tuple[VllmInquery, list]:
        """
        Improved two-stage approach for streaming with history.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (vllm_request, retrieval_document) for streaming
        """
        session_id = self.vllm_handler.history_manager.current_session_id
        logger.debug(f"[{session_id}] Starting improved streaming history processing")
        start_time = time.time()

        # 1. Use conversation history to create a standalone question
        rewrite_start_time = time.time()

        # Get conversation history
        session_history = self.vllm_handler.history_manager.get_session_history()
        formatted_history = HistoryFormatter.format_for_prompt(
            session_history,
            settings.llm.max_history_turns
        )

        # If no history, use original question
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] No conversation history, using original question")
            rewritten_question = request.chat.user
        else:
            # Get question rewriting prompt template
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # Create question rewriting prompt
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            rewrite_prompt = rewrite_prompt_template.format(**rewrite_context)

            # Request question rewriting from vLLM with timeout
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{session_id}] Sending question rewriting request to vLLM")
            try:
                # Apply timeout to question rewriting
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)
                rewrite_response = await asyncio.wait_for(
                    self.vllm_handler._call_vllm_endpoint(rewrite_request),
                    timeout=rewrite_timeout
                )
                rewritten_question = rewrite_response.get("generated_text", "").strip()

                # Validate rewritten question
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] Question rewriting failed, using original question")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] Final rewritten question: '{rewritten_question}'")
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] Question rewriting timed out, using original question")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] Error during question rewriting: {str(e)}")
                rewritten_question = request.chat.user

        rewrite_time = time.time() - rewrite_start_time
        self.vllm_handler.response_stats["rewrite_time"] = rewrite_time

        # 2. Use rewritten question to retrieve documents
        retrieval_start_time = time.time()
        logger.debug(f"[{session_id}] Retrieving documents using rewritten question")

        retrieval_document = []  # Default to empty list
        try:
            # Check if retriever is initialized
            if self.vllm_handler.history_manager.retriever is not None:
                retrieval_document = await self.vllm_handler.history_manager.retriever.ainvoke(rewritten_question)
                logger.debug(f"[{session_id}] Retrieved {len(retrieval_document)} documents")
            else:
                logger.warning(f"[{session_id}] Retriever not initialized, using empty document list")
        except Exception as e:
            logger.error(f"[{session_id}] Error during document retrieval: {str(e)}")

        retrieval_time = time.time() - retrieval_start_time
        self.vllm_handler.response_stats["retrieval_time"] = retrieval_time

        # 3. Prepare streaming response
        # Get RAG prompt template
        rag_prompt_template = self.vllm_handler.history_manager.response_generator.get_rag_qa_prompt(
            self.vllm_handler.history_manager.current_rag_sys_info
        )

        # Prepare context for final response
        final_prompt_context = {
            "input": request.chat.user,  # Original question
            "rewritten_question": rewritten_question,  # Rewritten question
            "history": formatted_history,  # Formatted conversation history
            "context": retrieval_document,  # Retrieved documents
            "language": language,
            "today": self.vllm_handler.history_manager.response_generator.get_today(),
        }

        # Add VOC-specific context if needed
        if self.vllm_handler.history_manager.current_rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Handle image data if present
        if hasattr(request.chat, 'image') and request.chat.image:
            final_prompt_context["image_description"] = self.vllm_handler._format_image_data(request.chat.image)

        # Build system prompt based on model
        if self.vllm_handler.history_manager.is_gemma_model():
            vllm_inquery_context = self.vllm_handler._build_system_prompt_gemma(rag_prompt_template,
                                                                                final_prompt_context)
        else:
            vllm_inquery_context = self.vllm_handler._build_improved_system_prompt(rag_prompt_template,
                                                                                   final_prompt_context)

        # Create streaming vLLM request
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # Enable streaming
        )

        # Track total preparation time
        total_prep_time = time.time() - start_time
        self.vllm_handler.response_stats["total_prep_time"] = total_prep_time

        logger.debug(
            f"[{session_id}] Streaming request prepared - "
            f"Original question: '{request.chat.user}', "
            f"Rewritten question: '{rewritten_question}', "
            f"Documents: {len(retrieval_document)}, "
            f"Prep time: {total_prep_time:.4f}s"
        )

        return vllm_request, retrieval_document

    async def _handle_streaming_with_history_gemma(self, request: ChatRequest, language: str) -> Tuple[
        VllmInquery, list]:
        """
        Gemma-specific implementation for streaming with history.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (vllm_request, retrieval_document) for streaming
        """
        session_id = self.vllm_handler.history_manager.current_session_id
        logger.debug(f"[{session_id}] Starting Gemma streaming history processing")
        start_time = time.time()

        # 1. Use conversation history to create a standalone question
        # Get conversation history in Gemma format
        session_history = self.vllm_handler.history_manager.get_session_history()
        formatted_history = HistoryFormatter.format_for_gemma(
            session_history,
            settings.llm.max_history_turns
        )

        # If no history, use original question
        if not formatted_history or len(session_history.messages) == 0:
            logger.debug(f"[{session_id}] No conversation history, using original question")
            rewritten_question = request.chat.user
        else:
            # Get question rewriting prompt
            rewrite_prompt_template = PromptManager.get_rewrite_prompt_template()

            # Create question rewriting context
            rewrite_context = {
                "history": formatted_history,
                "input": request.chat.user,
            }

            # Create Gemma-formatted prompt
            rewrite_prompt = self.vllm_handler._build_system_prompt_gemma(rewrite_prompt_template, rewrite_context)

            # Request question rewriting with timeout
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{session_id}] Sending question rewriting request to Gemma")
            try:
                # Apply timeout to question rewriting
                rewrite_timeout = getattr(settings.llm, 'rewrite_timeout', 3.0)
                rewrite_response = await asyncio.wait_for(
                    self.vllm_handler._call_vllm_endpoint(rewrite_request),
                    timeout=rewrite_timeout
                )
                rewritten_question = rewrite_response.get("generated_text", "").strip()

                # Validate rewritten question
                if not rewritten_question or len(rewritten_question) < 5:
                    logger.warning(f"[{session_id}] Question rewriting failed, using original question")
                    rewritten_question = request.chat.user
                else:
                    logger.debug(f"[{session_id}] Final rewritten question: '{rewritten_question}'")
            except asyncio.TimeoutError:
                logger.warning(f"[{session_id}] Question rewriting timed out, using original question")
                rewritten_question = request.chat.user
            except Exception as e:
                logger.error(f"[{session_id}] Error during question rewriting: {str(e)}")
                rewritten_question = request.chat.user

        # 2. Use rewritten question to retrieve documents
        logger.debug(f"[{session_id}] Retrieving documents using rewritten question")

        retrieval_document = []  # Default to empty list
        try:
            # Check if retriever is initialized
            if self.vllm_handler.history_manager.retriever is not None:
                retrieval_document = await self.vllm_handler.history_manager.retriever.ainvoke(rewritten_question)
                logger.debug(f"[{session_id}] Retrieved {len(retrieval_document)} documents")
            else:
                logger.warning(f"[{session_id}] Retriever not initialized, using empty document list")
        except Exception as e:
            logger.error(f"[{session_id}] Error during document retrieval: {str(e)}")

        # 3. Prepare streaming response
        # Get RAG prompt template
        rag_prompt_template = self.vllm_handler.history_manager.response_generator.get_rag_qa_prompt(
            self.vllm_handler.history_manager.current_rag_sys_info
        )

        # Prepare context for final response
        final_prompt_context = {
            "input": request.chat.user,  # Original question
            "rewritten_question": rewritten_question,  # Rewritten question
            "history": formatted_history,  # Formatted conversation history
            "context": retrieval_document,  # Retrieved documents
            "language": language,
            "today": self.vllm_handler.history_manager.response_generator.get_today(),
        }

        # Add VOC-specific context if needed
        if self.vllm_handler.history_manager.current_rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Handle image data if present
        if hasattr(request.chat, 'image') and request.chat.image:
            final_prompt_context["image_description"] = self.vllm_handler._format_image_data(request.chat.image)

        # Build Gemma-formatted system prompt
        vllm_inquery_context = self.vllm_handler._build_system_prompt_gemma(rag_prompt_template,
                                                                            final_prompt_context)

        # Create streaming vLLM request
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context,
            stream=True  # Enable streaming
        )

        total_prep_time = time.time() - start_time
        self.vllm_handler.response_stats["total_prep_time"] = total_prep_time

        logger.debug(f"[{session_id}] Gemma streaming request prepared")

        return vllm_request, retrieval_document
