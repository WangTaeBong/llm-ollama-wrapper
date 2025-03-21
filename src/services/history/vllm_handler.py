# services/history/vllm_handler.py
"""
vLLM integration for conversation history handling.

This module provides specialized history handling for vLLM backend,
including streaming support and model-specific optimizations.
"""

import logging

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.history.formatter import HistoryFormatter
from src.services.history.manager import HistoryManager
from src.utils.prompt import PromptManager

# Load settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class VLLMHistoryHandler:
    """
    Specialized history handler for vLLM backend.

    Provides optimized conversation history handling for vLLM models,
    including streaming support and two-stage context processing.
    """

    def __init__(self, history_manager: HistoryManager):
        """
        Initialize the vLLM history handler.

        Args:
            history_manager: The main history manager instance
        """
        self.history_manager = history_manager
        self.response_stats = {
            "rewrite_time": 0,
            "retrieval_time": 0,
            "total_prep_time": 0
        }

    async def handle_chat_with_history(self, request: ChatRequest, language: str):
        """
        Process a chat request with conversation history for vLLM backend.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (answer, retrieval_document)
        """
        # Determine appropriate handler based on model type
        if self.history_manager.is_gemma_model():
            logger.info(f"[{self.history_manager.current_session_id}] Detected Gemma model, using Gemma handler")
            return await self._handle_chat_with_history_gemma(request, language)
        else:
            # Use the improved history handler if enabled
            if getattr(settings.llm, 'use_improved_history', False):
                return await self._handle_chat_with_history_improved(request, language)
            else:
                return await self._handle_chat_with_history_original(request, language)

    async def _handle_chat_with_history_original(self, request: ChatRequest, language: str):
        """
        Original implementation for handling chat with history for vLLM.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (answer, retrieval_document)
        """
        # Get prompt template
        rag_prompt_template = self.history_manager.response_generator.get_rag_qa_prompt(
            self.history_manager.current_rag_sys_info
        )

        # Get retrieval documents
        logger.debug(f"[{self.history_manager.current_session_id}] Retrieving documents...")
        retrieval_document = await self.history_manager.retriever.ainvoke(request.chat.user)
        logger.debug(f"[{self.history_manager.current_session_id}] Retrieved {len(retrieval_document)} documents")

        # Get chat history in structured format
        session_history = self.history_manager.get_session_history()
        formatted_history = HistoryFormatter.format_for_prompt(
            session_history,
            settings.llm.max_history_turns
        )
        logger.debug(
            f"[{self.history_manager.current_session_id}] Processed {len(session_history.messages)} history messages for vLLM")

        # Prepare context for prompt
        common_input = {
            "input": request.chat.user,
            "history": formatted_history,  # Using optimized format
            "context": retrieval_document,
            "language": language,
            "today": self.history_manager.response_generator.get_today(),
        }

        # Add VOC-specific context if needed
        if self.history_manager.current_rag_sys_info == "komico_voc":
            common_input.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Handle image data if present
        if hasattr(request.chat, 'image') and request.chat.image:
            common_input["image_description"] = self._format_image_data(request.chat.image)

        # Build prompt and create request
        if self.history_manager.is_gemma_model():
            vllm_inquery_context = self._build_system_prompt_gemma(rag_prompt_template, common_input)
        else:
            vllm_inquery_context = self._build_system_prompt(rag_prompt_template, common_input)

        vllm_request = VllmInquery(
            request_id=self.history_manager.current_session_id,
            prompt=vllm_inquery_context
        )

        # Call vLLM endpoint
        logger.debug(f"[{self.history_manager.current_session_id}] Calling vLLM endpoint")
        response = await self._call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{self.history_manager.current_session_id}] vLLM response received, length: {len(answer)}")

        return answer, retrieval_document

    async def _handle_chat_with_history_improved(self, request: ChatRequest, language: str):
        """
        Improved two-stage approach for handling chat with history for vLLM.

        Args:
            request: The chat request
            language: The response language

        Returns:
            tuple: (answer, retrieval_document)
        """
        session_id = self.history_manager.current_session_id
        logger.debug(f"[{session_id}] Starting improved vLLM history processing")

        # 1. Use conversation history to create a standalone question
        # Get conversation history
        session_history = self.history_manager.get_session_history()
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

            # Request question rewriting from vLLM
            rewrite_request = VllmInquery(
                request_id=f"{session_id}_rewrite",
                prompt=rewrite_prompt
            )

            logger.debug(f"[{session_id}] Sending question rewriting request to vLLM")
            rewrite_response = await self._call_vllm_endpoint(rewrite_request)
            rewritten_question = rewrite_response.get("generated_text", "").strip()

            # Validate rewritten question
            if not rewritten_question or len(rewritten_question) < 5:
                logger.warning(f"[{session_id}] Question rewriting failed, using original question")
                rewritten_question = request.chat.user
            else:
                logger.debug(f"[{session_id}] Final rewritten question: '{rewritten_question}'")

        # 2. Use rewritten question to retrieve documents
        logger.debug(f"[{session_id}] Retrieving documents using rewritten question")
        retrieval_document = await self.history_manager.retriever.ainvoke(rewritten_question)
        logger.debug(f"[{session_id}] Retrieved {len(retrieval_document)} documents")

        # 3. Generate final response
        # Get RAG prompt template
        rag_prompt_template = self.history_manager.response_generator.get_rag_qa_prompt(
            self.history_manager.current_rag_sys_info
        )

        # Prepare context for final response
        final_prompt_context = {
            "input": request.chat.user,  # Original question
            "rewritten_question": rewritten_question,  # Rewritten question
            "history": formatted_history,  # Formatted conversation history
            "context": retrieval_document,  # Retrieved documents
            "language": language,
            "today": self.history_manager.response_generator.get_today(),
        }

        # Add VOC-specific context if needed
        if self.history_manager.current_rag_sys_info == "komico_voc":
            final_prompt_context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        # Handle image data if present
        if hasattr(request.chat, 'image') and request.chat.image:
            final_prompt_context["image_description"] = self._format_image_data(request.chat.image)

        # Build system prompt
        if self.history_manager.is_gemma_model():
            vllm_inquery_context = self._build_system_prompt_gemma(rag_prompt_template, final_prompt_context)
        else:
            vllm_inquery_context = self._build_improved_system_prompt(rag_prompt_template, final_prompt_context)

        # Create vLLM request
        vllm_request = VllmInquery(
            request_id=session_id,
            prompt=vllm_inquery_context
        )

        # Call vLLM endpoint
        logger.debug(f"[{session_id}] Sending final response request to vLLM")
        response = await self._call_vllm_endpoint(vllm_request)
        answer = response.get("generated_text", "") or response.get("answer", "")
        logger.debug(f"[{session_id}] Final response received, length: {len(answer)}")

        return answer, retrieval_document
