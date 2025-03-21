# services/chat/streaming.py
"""
Streaming service implementation for chat responses.

This module provides the streaming processor for real-time character-by-character
response handling to provide faster user experience.
"""

import asyncio
import logging
import re
import time
from typing import Tuple

# Load configuration
from src.common.config_loader import ConfigLoader

settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)


class StreamResponsePostProcessor:
    """
    Processes streaming responses character by character for faster user experience.

    Optimization points:
    1. Process text character by character instead of waiting for sentence completion
    2. Set minimum display units for natural flow
    3. Ensure text consistency with special character handling
    4. Process special patterns like URLs only after completion
    5. Send the complete response after the final chunk for client-side reconstruction
    """

    def __init__(self, response_generator, voc_processor, search_engine, request, documents):
        """Initialize processor with required components"""
        self.response_generator = response_generator
        self.voc_processor = voc_processor
        self.search_engine = search_engine
        self.request = request
        self.documents = documents
        self.logger = logging.getLogger(__name__)

        # Track full response
        self.full_text = ""
        self.processed_chunks = []

        # Processing settings
        self.min_chars = 2  # Consider at least 2 chars (for Korean character composition)
        self.force_interval = 100  # Force send after 100ms
        self.last_send_time = time.time()

        # URL and special pattern detection
        self.url_pattern = re.compile(r'https?://\S+')
        self.url_buffer = ""  # Buffer for URL completion
        self.in_url = False  # URL processing state

    def process_partial(self, text: str) -> Tuple[str, str]:
        """
        Process text character by character without waiting for sentence completion.

        Args:
            text: Text to process

        Returns:
            tuple: (processed_text, remaining_buffer)
        """
        current_time = time.time()
        force_send = (current_time - self.last_send_time) > (self.force_interval / 1000)

        # Skip processing if text is empty
        if not text:
            return "", ""

        # Check for URL pattern - buffer until URL is complete
        if self.in_url:
            # Check for URL termination (whitespace, newline, etc.)
            end_idx = -1
            for i, char in enumerate(text):
                if char.isspace():
                    end_idx = i
                    break

            if end_idx >= 0:
                # URL is complete
                self.url_buffer += text[:end_idx]
                processed_url = self._quick_process_urls(self.url_buffer)

                # Return processed result and remaining text
                self.in_url = False
                self.full_text += self.url_buffer + text[end_idx:end_idx + 1]
                remaining = text[end_idx + 1:]
                self.url_buffer = ""

                self.last_send_time = current_time
                return processed_url + text[end_idx:end_idx + 1], remaining
            else:
                # Continue accumulating URL
                self.url_buffer += text
                self.full_text += text
                return "", ""  # Defer output until URL is complete

        # Detect URL start
        url_match = self.url_pattern.search(text)
        if url_match:
            start_idx = url_match.start()
            if start_idx > 0:
                # Process text before URL
                prefix = text[:start_idx]
                self.full_text += prefix

                # Start buffering URL
                self.in_url = True
                self.url_buffer = text[start_idx:]

                self.last_send_time = current_time
                return prefix, ""
            else:
                # Text starts with URL
                self.in_url = True
                self.url_buffer = text
                self.full_text += text
                return "", ""

        # Regular text processing (not URL)
        # Send if there's enough text or force send condition is met
        if len(text) >= self.min_chars or force_send:
            self.full_text += text
            self.last_send_time = current_time
            return text, ""

        # Keep in buffer if below minimum length
        self.full_text += text
        return "", ""

    def _quick_process_urls(self, text: str) -> str:
        """Quickly transform URLs into links"""
        return self.url_pattern.sub(lambda m: f'<a href="{m.group(0)}" target="_blank">{m.group(0)}</a>', text)

    async def finalize(self, remaining_text: str) -> str:
        """
        Perform final processing - add references and format the response.

        Args:
            remaining_text: Any remaining text to process

        Returns:
            str: The complete processed response
        """
        session_id = self.request.meta.session_id
        self.logger.debug(f"[{session_id}] Starting final response processing")

        # Process remaining text and URL buffer
        final_text = remaining_text
        if self.url_buffer:
            final_text = self.url_buffer + final_text
            self.url_buffer = ""
            self.in_url = False

        if final_text:
            self.full_text += final_text

        # Return empty string if there's nothing to process
        if not final_text and not self.full_text:
            return ""

        try:
            # Get language settings
            _, _, reference_word = self.response_generator.get_translation_language_word(
                self.request.chat.lang
            )

            # Perform final processing on the complete text
            processed_text = self.full_text

            # 1. Add references
            if settings.prompt.source_count:
                processed_text = await asyncio.to_thread(
                    self.response_generator.make_answer_reference,
                    processed_text,
                    self.request.meta.rag_sys_info,
                    reference_word,
                    self.documents,
                    self.request
                )

            # 2. Process VOC
            if "komico_voc" in settings.voc.voc_type.split(',') and self.request.meta.rag_sys_info == "komico_voc":
                processed_text = await asyncio.to_thread(
                    self.voc_processor.make_komico_voc_groupware_docid_url,
                    processed_text
                )

            # 3. Process URLs
            final_text = await asyncio.to_thread(
                self.search_engine.replace_urls_with_links,
                processed_text
            )

            self.logger.debug(f"[{session_id}] Final response processing complete")
            return final_text

        except Exception as e:
            self.logger.error(f"[{session_id}] Error during final response processing: {str(e)}", exc_info=True)
            # Return original text if processing fails
            return self.full_text

    def get_full_text(self) -> str:
        """Return the complete response text"""
        # Include any content remaining in the URL buffer
        if self.url_buffer:
            return self.full_text + self.url_buffer
        return self.full_text
