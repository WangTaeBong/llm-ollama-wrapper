# services/llm/vllm.py
"""
vLLM service implementation.

This module provides the implementation of the LLM service interface
for the vLLM backend, supporting both regular and streaming responses.
"""

import logging
import time
from typing import Any, Dict, List

from src.common.config_loader import ConfigLoader
from src.common.restclient import rc
from src.schema.vllm_inquery import VllmInquery
from src.services.llm.base import LLMServiceBase
from src.services.utils.async_helpers import async_retry
from src.services.utils.circuit_breaker import CircuitBreaker

# Load application settings
settings = ConfigLoader().get_settings()
logger = logging.getLogger(__name__)

# Create circuit breaker for vLLM service
_vllm_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.circuit_breaker.failure_threshold if hasattr(settings.circuit_breaker,
                                                                            'failure_threshold') else 3,
    recovery_timeout=settings.circuit_breaker.recovery_timeout if hasattr(settings.circuit_breaker,
                                                                          'recovery_timeout') else 60,
    reset_timeout=settings.circuit_breaker.reset_timeout if hasattr(settings.circuit_breaker, 'reset_timeout') else 300
)


class VLLMService(LLMServiceBase):
    """
    LLM service implementation for vLLM backend.

    Handles interactions with vLLM models, including initialization,
    prompt management, and query processing with streaming support.
    """

    def __init__(self, response_generator):
        """
        Initialize the vLLM service.

        Args:
            response_generator: Generator for responses based on LLM outputs
        """
        self.response_generator = response_generator
        self.system_prompt_template = None
        self.timeout = getattr(settings.llm, 'timeout', 60)

        # For performance monitoring
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0
        }

    async def initialize(self):
        """
        Initialize the vLLM service.

        Returns:
            bool: True if initialization is successful, False otherwise.
        """
        try:
            # Validate settings
            if not settings.vllm.endpoint_url:
                raise ValueError("vLLM endpoint URL not configured")

            # Initialize system prompt template
            self.system_prompt_template = self.response_generator.get_rag_qa_prompt("default")

            logger.info(f"vLLM service will connect to: {settings.vllm.endpoint_url}")
            return True

        except Exception as err:
            logger.error(f"vLLM service initialization failed: {err}", exc_info=True)
            return False

    def build_system_prompt(self, context):
        """
        Build system prompt with dynamic variables and chat history.

        Args:
            context (dict): Context with keys like {input}, {context}, {language}, {today}.

        Returns:
            str: Formatted system prompt.
        """
        try:
            # Handle image data if present
            if (hasattr(context, 'image') and context.get('image')):
                # Add image description to context
                if 'image_description' not in context:
                    context['image_description'] = self._format_image_data(context.get('image'))

                    # Add image description token to template if not present
                    if '{image_description}' not in self.system_prompt_template:
                        insert_point = self.system_prompt_template.find('{input}')
                        if insert_point > 0:
                            image_instruction = "\n\n# Image Information\nThe following is information about an image provided by the user:\n{image_description}\n\n# Question\n"
                            self.system_prompt_template = (
                                    self.system_prompt_template[:insert_point] +
                                    image_instruction +
                                    self.system_prompt_template[insert_point:]
                            )

            prompt = self.system_prompt_template.format(**context)
            return prompt
        except KeyError as e:
            # Set missing key to empty string and log as warning
            missing_key = str(e).strip("'")
            logger.warning(f"Key missing in system prompt build: {missing_key}")
            context[missing_key] = ""
            return self.system_prompt_template.format(**context)
        except Exception as e:
            logger.error(f"System prompt build failed: {str(e)}")
            raise

    def build_system_prompt_gemma(self, context):
        """
        Build system prompt in Gemma format.

        Args:
            context (dict): Context with variables for the template.

        Returns:
            str: System prompt in Gemma format.
        """
        try:
            # Handle image data if present
            if 'image' in context and context['image']:
                if 'image_description' not in context:
                    context['image_description'] = self._format_image_data(context['image'])

                    if '{image_description}' not in self.system_prompt_template:
                        insert_point = self.system_prompt_template.find('{input}')
                        if insert_point > 0:
                            image_instruction = "\n\n# Image Information\nThe following is information about an image provided by the user:\n{image_description}\n\n# Question\n"
                            self.system_prompt_template = (
                                    self.system_prompt_template[:insert_point] +
                                    image_instruction +
                                    self.system_prompt_template[insert_point:]
                            )

            # Generate regular prompt first
            raw_prompt = self.build_system_prompt(context)

            # Convert to Gemma format
            formatted_prompt = "<start_of_turn>user\n"
            formatted_prompt += raw_prompt
            formatted_prompt += "\n<end_of_turn>\n<start_of_turn>model\n"

            return formatted_prompt

        except KeyError as e:
            # Handle missing key
            missing_key = str(e).strip("'")
            logger.warning(f"Missing key in system prompt template: {missing_key}, substituting with empty string")
            context[missing_key] = ""
            return self.build_system_prompt_gemma(context)
        except Exception as e:
            # Handle other exceptions
            logger.error(f"Error formatting Gemma system prompt: {e}")
            # Fallback to basic Gemma prompt
            basic_prompt = f"<start_of_turn>user\nPlease answer the following question: {context.get('input', 'No question provided')}\n<end_of_turn>\n<start_of_turn>model\n"
            return basic_prompt

    @async_retry(max_retries=2, backoff_factor=2, circuit_breaker=_vllm_circuit_breaker)
    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        Call vLLM endpoint with retry and circuit breaker.

        Args:
            data (VllmInquery): vLLM request data.

        Returns:
            Dict: Response from vLLM
        """
        start_time = time.time()
        session_id = data.request_id
        logger.debug(f"[{session_id}] Calling vLLM endpoint (stream={data.stream})")

        # Check circuit breaker
        if _vllm_circuit_breaker.is_open():
            logger.warning(f"[{session_id}] Circuit breaker is open, skipping request")
            raise RuntimeError("vLLM service unavailable: circuit breaker is open")

        vllm_url = settings.vllm.endpoint_url

        try:
            response = await rc.restapi_post_async(vllm_url, data)
            elapsed = time.time() - start_time
            logger.debug(f"[{session_id}] vLLM response received: {elapsed:.4f}s elapsed")

            # Update metrics
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            # Update circuit breaker
            _vllm_circuit_breaker.record_success()

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM endpoint error: {elapsed:.4f}s after: {str(e)}")

            self.metrics["error_count"] += 1

            # Update circuit breaker
            _vllm_circuit_breaker.record_failure()

            raise

    async def _stream_vllm_response(self, session_id: str, url: str, data: VllmInquery):
        """
        Stream response from vLLM endpoint.

        Args:
            session_id: Session ID
            url: vLLM endpoint URL
            data: vLLM request data

        Yields:
            Dict: Response chunks
        """
        start_time = time.time()

        try:
            logger.debug(f"[{session_id}] Starting vLLM streaming")

            # Stream chunks from vLLM endpoint
            async for chunk in rc.restapi_stream_async(session_id, url, data):
                if chunk is None:
                    continue

                # Process chunk
                processed_chunk = self._process_vllm_chunk(chunk)

                # Log chunk (only length if it contains text)
                log_chunk = processed_chunk.copy()
                if 'new_text' in log_chunk:
                    log_chunk['new_text'] = f"<Text of length {len(log_chunk['new_text'])}>"
                logger.debug(f"[{session_id}] Processed chunk: {log_chunk}")

                yield processed_chunk

                # Handle final chunk
                if processed_chunk.get('finished', False) or processed_chunk.get('error', False):
                    # Record success to circuit breaker
                    _vllm_circuit_breaker.record_success()

                    # Update metrics
                    elapsed = time.time() - start_time
                    self.metrics["request_count"] += 1
                    self.metrics["total_time"] += elapsed

                    logger.debug(f"[{session_id}] vLLM streaming complete: {elapsed:.4f}s elapsed")
                    break

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM streaming error: {elapsed:.4f}s after: {str(e)}")

            # Update metrics
            self.metrics["error_count"] += 1

            # Record failure to circuit breaker
            _vllm_circuit_breaker.record_failure()

            # Return error chunk
            yield {
                "error": True,
                "message": f"Streaming error: {str(e)}",
                "finished": True
            }

    @classmethod
    def _process_vllm_chunk(cls, chunk):
        """
        Process a vLLM response chunk to standardized format.

        Args:
            chunk: Raw vLLM response chunk

        Returns:
            Dict: Processed chunk
        """
        # Check for error
        if 'error' in chunk:
            return {
                'error': True,
                'message': chunk.get('message', 'Unknown error'),
                'finished': True
            }

        # Check for completion marker
        if chunk == '[DONE]':
            return {
                'new_text': '',
                'finished': True
            }

        # Process different chunk formats
        if isinstance(chunk, dict):
            # Text chunk (standard streaming)
            if 'new_text' in chunk:
                return {
                    'new_text': chunk['new_text'],
                    'finished': chunk.get('finished', False)
                }
            # Completion signal
            elif 'finished' in chunk and chunk['finished']:
                return {
                    'new_text': '',
                    'finished': True
                }
            # Full text response (non-streaming format)
            elif 'generated_text' in chunk:
                return {
                    'new_text': chunk['generated_text'],
                    'finished': True
                }
            # OpenAI-compatible format
            elif 'delta' in chunk:
                return {
                    'new_text': chunk['delta'].get('content', ''),
                    'finished': chunk.get('finished', False)
                }
            # Unknown format
            else:
                return chunk

        # String response (rare case)
        elif isinstance(chunk, str):
            return {
                'new_text': chunk,
                'finished': False
            }

        # Handle other types
        return {
            'new_text': str(chunk),
            'finished': False
        }

    async def ask(self, query: str, documents: List[Any], language: str) -> str:
        """
        Send a query to the vLLM with the given context.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Returns:
            str: Generated response.
        """
        start_time = time.time()

        # Prepare context
        context = {
            "input": query,
            "context": documents,
            "language": language,
            "today": self.response_generator.get_today(),
        }

        try:
            # Determine if Gemma model should be used
            use_gemma_format = self.is_gemma_model()

            if use_gemma_format:
                logger.debug(f"Detected Gemma model, using Gemma format")
                vllm_inquery_context = self.build_system_prompt_gemma(context)
            else:
                vllm_inquery_context = self.build_system_prompt(context)

            # Create vLLM request
            vllm_request = VllmInquery(
                request_id=f"ask_{int(time.time())}",
                prompt=vllm_inquery_context,
                stream=False
            )

            # Call vLLM endpoint
            response = await self.call_vllm_endpoint(vllm_request)
            result = response.get("generated_text", "")

            elapsed = time.time() - start_time
            logger.info(f"vLLM query complete: {elapsed:.4f}s elapsed")

            # Update metrics
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"vLLM query failed: {elapsed:.4f}s after: {type(e).__name__}: {str(e)}", exc_info=True)

            # Update metrics
            self.metrics["error_count"] += 1
            raise

    async def stream_response(self, query: str, documents: List[Any], language: str):
        """
        Stream a response from the vLLM.

        Args:
            query (str): The user query.
            documents (List[Any]): List of documents for context.
            language (str): Response language.

        Yields:
            Dict: Response chunks
        """
        session_id = f"stream_{int(time.time())}"
        logger.debug(f"[{session_id}] Starting streaming response")

        # Prepare context
        context = {
            "input": query,
            "context": documents,
            "language": language,
            "today": self.response_generator.get_today(),
        }

        try:
            # Determine if Gemma model should be used
            use_gemma_format = self.is_gemma_model()

            if use_gemma_format:
                logger.debug(f"[{session_id}] Detected Gemma model, using Gemma format")
                vllm_inquery_context = self.build_system_prompt_gemma(context)
            else:
                vllm_inquery_context = self.build_system_prompt(context)

            # Create vLLM request for streaming
            vllm_request = VllmInquery(
                request_id=session_id,
                prompt=vllm_inquery_context,
                stream=True
            )

            # Stream from vLLM endpoint
            vllm_url = settings.vllm.endpoint_url

            async for chunk in self._stream_vllm_generate(session_id, vllm_url, vllm_request):
                yield chunk

        except Exception as e:
            logger.error(f"[{session_id}] Streaming response error: {str(e)}", exc_info=True)
            yield {"error": True, "message": str(e)}

    @classmethod
    async def _stream_vllm_generate(cls, session_id: str, url: str, data: VllmInquery):
        """
        Stream generator for vLLM responses.

        Args:
            session_id: Session ID
            url: vLLM endpoint URL
            data: vLLM request data

        Yields:
            dict: Streaming chunks
        """
        try:
            # Generate streaming response
            async for chunk in rc.restapi_stream_async(session_id, url, data):
                if chunk is None:
                    continue

                # Format chunk correctly
                if isinstance(chunk, dict):
                    # Chunk with new text
                    if 'new_text' in chunk:
                        yield {
                            "text": chunk['new_text'],
                            "finished": chunk.get('finished', False)
                        }
                    # Completion signal only
                    elif chunk.get('finished', False):
                        yield {"text": "", "finished": True}
                elif chunk == '[DONE]':
                    # Completion marker
                    yield {"text": "", "finished": True}
                else:
                    # Other chunk formats
                    yield {"text": str(chunk), "finished": False}

        except Exception as e:
            logger.error(f"[{session_id}] vLLM streaming error: {str(e)}")
            yield {"error": True, "message": str(e)}

    def get_metrics(self):
        """
        Get performance metrics for the vLLM service.

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
