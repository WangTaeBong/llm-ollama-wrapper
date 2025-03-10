"""
Enterprise Chat Service with RAG (Retrieval-Augmented Generation) System
===================================================================

This module implements a comprehensive chat service that integrates various LLM backends
with document retrieval and processing capabilities. It provides robust handling for
concurrent requests, caching, error management, and performance monitoring.

Key components:
- LLM interaction (Ollama and vLLM backends)
- Document retrieval and processing
- Query preprocessing and optimization
- Chat history management
- Response generation with context enhancement
- Async processing with concurrency control
- Circuit breaker pattern for external service protection
- Comprehensive logging and performance tracking

Dependencies:
- FastAPI for the web framework
- LangChain for chain operations
- Redis for caching chat history
- Various utility modules for specific processing tasks
"""

import asyncio
import inspect
import logging
import random
import re
import time
from asyncio import Semaphore, create_task, wait_for, TimeoutError
from contextlib import asynccontextmanager
from functools import lru_cache, wraps
from threading import Lock

from cachetools import TTLCache
from fastapi import FastAPI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from src.common.config_loader import ConfigLoader
from src.common.error_cd import ErrorCd
from src.common.query_check_dict import QueryCheckDict
from src.common.restclient import rc
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse, MetaRes, PayloadRes, ChatRes
from src.schema.vllm_inquery import VllmInquery
from src.services.chat_message_handler import create_chat_data, create_message
from src.services.document_processor import DocumentProcessor
from src.services.llm_history_handler import LlmHistoryHandler
from src.services.query_processor import QueryProcessor
from src.services.response_generator import ResponseGenerator
from src.services.search_engine import SearchEngine
from src.services.voc_processor import VOCLinkProcessor
from src.utils.redis_utils import RedisUtils

# Load application settings from configuration
settings = ConfigLoader().get_settings()

# Configure module-level logger
logger = logging.getLogger(__name__)

# Initialize TTL cache for storing chains to improve performance
# The cache has a maximum size and time-to-live defined in settings
_chain_registry = TTLCache(maxsize=settings.cache.max_size, ttl=settings.cache.chain_ttl)

# Thread-safe lock for chain registry access
_chain_lock = Lock()

# Semaphore to limit concurrent tasks for resource management
_semaphore = Semaphore(settings.cache.max_concurrent_tasks)

# Global LLM instance to be initialized during startup
mai_chat_llm = None


@lru_cache(maxsize=100)
def get_or_create_chain(settings_key, model, prompt_template):
    """
    Retrieve a cached chain or create a new one based on a settings key.
    Uses LRU cache for efficient caching.

    Args:
        settings_key (str): Unique key for the chain configuration.
        model: LLM model instance.
        prompt_template: Prompt template for the chain.

    Returns:
        chain: The chain instance (either cached or newly created).
    """
    with _chain_lock:
        if settings_key in _chain_registry:
            cached_chain = _chain_registry[settings_key]
            # Ensure the cached chain has the required 'invoke' method
            if not hasattr(cached_chain, 'invoke'):
                _chain_registry[settings_key] = create_stuff_documents_chain(model, prompt_template)
            return _chain_registry[settings_key]

        # Create and cache a new chain
        _chain_registry[settings_key] = create_stuff_documents_chain(model, prompt_template)
        return _chain_registry[settings_key]


async def limited_chain_invoke(chain, context, timeout=60):
    """
    Invoke a chain with concurrency and timeout controls.

    Manages resource utilization by using a semaphore and applies timeout
    to prevent hanging operations.

    Args:
        chain: The chain to invoke.
        context (dict): Context for the chain.
        timeout (int): Timeout in seconds.

    Returns:
        Any: Result of the chain.

    Raises:
        TimeoutError: If the operation exceeds the timeout period.
    """
    start_time = time.time()

    async with _semaphore:
        try:
            if inspect.iscoroutinefunction(chain.invoke):
                # Apply timeout to async call
                return await wait_for(chain.invoke(context), timeout=timeout)
            else:
                # For sync calls, use wait_for with a task wrapper
                return await wait_for(create_task(chain.invoke(context)), timeout=timeout)
        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"Chain invocation timed out: {elapsed:.2f}s (limit: {timeout}s)")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Chain invocation failed: {elapsed:.2f}s after: {type(e).__name__}: {str(e)}")
            raise


@asynccontextmanager
async def model_llm_init(app: FastAPI):
    """
    Async context manager for LLM model initialization.
    Initializes either ollama or vLLM based on configuration.

    Args:
        app (FastAPI): FastAPI app instance.

    Yields:
        dict: Dictionary of initialized models.
    """
    try:
        global mai_chat_llm

        # Validate configuration settings
        validate_settings()

        backend = settings.llm.llm_backend.lower()
        initialized_models = {}

        logger.info(f"Starting LLM initialization: {backend} backend")

        if backend == 'ollama':
            logger.info(f"Initializing Ollama LLM: {settings.ollama.access_type} access method")
            if settings.ollama.access_type.lower() == 'url':
                # Initialize Ollama with URL access
                mai_chat_llm = OllamaLLM(
                    base_url=settings.ollama.ollama_url,
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )
                initialized_models[backend] = mai_chat_llm
            elif settings.ollama.access_type.lower() == 'local':
                # Initialize Ollama with local access
                mai_chat_llm = OllamaLLM(
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )
                initialized_models[backend] = None
            else:
                raise ValueError(f"Unsupported ollama access type: {settings.ollama.access_type}")
        elif backend == 'vllm':
            # vLLM doesn't need initialization here, just mark as initialized
            initialized_models[backend] = None
            logger.info(f"vLLM backend will connect to: {settings.vllm.endpoint_url}")
        else:
            raise ValueError(f"Unsupported LLM backend: {backend}")

        logger.info(f"LLM initialization complete: {backend} backend")

        # Register health check if supported
        if hasattr(app, 'add_healthcheck') and callable(app.add_healthcheck):
            app.add_healthcheck("llm", lambda: mai_chat_llm is not None if backend == 'ollama' else True)

        yield initialized_models

    except Exception as err:
        logger.error(f"LLM initialization failed: {err}", exc_info=True)
        raise RuntimeError(f"LLM initialization failed: {err}")
    finally:
        logger.info("LLM context exited")


def validate_settings():
    """
    Validate the Ollama or vLLM configuration settings.

    Raises:
        ValueError: If required configuration parameters are missing.
    """
    backend = settings.llm.llm_backend.lower()
    if backend == 'ollama':
        if not settings.ollama.model_name:
            raise ValueError("Ollama model name not configured")
        if settings.ollama.access_type.lower() == 'url' and not settings.ollama.ollama_url:
            raise ValueError("Ollama URL not configured for URL access method")
    elif backend == "vllm":
        if not settings.vllm.endpoint_url:
            raise ValueError("vLLM endpoint URL not configured")
    else:
        raise ValueError("LLM backend must be either 'ollama' or 'vllm'")


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls.

    Protects system stability by opening the circuit after consecutive failures,
    preventing additional requests, and automatically attempting recovery.

    The circuit has three states:
    - CLOSED: Normal operation, requests are passed through
    - OPEN: Circuit is open, all requests are rejected
    - HALF-OPEN: Test state, limited requests are allowed to check if service is recovered
    """

    def __init__(self, failure_threshold=3, recovery_timeout=60, reset_timeout=300):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold (int): Number of consecutive failures before opening the circuit.
            recovery_timeout (int): Seconds to wait before attempting recovery.
            reset_timeout (int): Seconds to wait before fully resetting the circuit.
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.open_time = None
        self.state = "CLOSED"
        self.half_open_calls = 0
        self.lock = Lock()

    def is_open(self):
        """
        Check if the circuit is open and handle auto-recovery.

        Returns:
            bool: True if requests should be blocked, False if allowed.
        """
        with self.lock:
            if self.state == "CLOSED":
                return False

            if self.state == "OPEN":
                # Check if recovery timeout has elapsed
                if time.time() - self.open_time > self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to half-open state")
                    self.state = "HALF-OPEN"
                    self.half_open_calls = 0
                    return False
                return True

            # In HALF-OPEN state, allow limited calls
            if self.half_open_calls < 1:
                self.half_open_calls += 1
                return False
            return True

    def record_success(self):
        """
        Record a successful call to the service.

        In HALF-OPEN state, this will close the circuit.
        In CLOSED state, this resets the failure counter.
        """
        with self.lock:
            if self.state == "HALF-OPEN":
                logger.info("Circuit breaker closed - service has recovered")
                self.state = "CLOSED"
                self.failure_count = 0
                self.open_time = None
                self.half_open_calls = 0
            elif self.state == "CLOSED":
                self.failure_count = 0

    def record_failure(self):
        """
        Record a failed call to the service.

        In HALF-OPEN state, this will re-open the circuit.
        In CLOSED state, this increases the failure counter and may open the circuit.
        """
        with self.lock:
            if self.state == "HALF-OPEN":
                logger.warning("Circuit breaker keeping open state - service still failing")
                self.state = "OPEN"
                self.open_time = time.time()
            elif self.state == "CLOSED":
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    logger.warning("Circuit breaker opening - failure threshold reached")
                    self.state = "OPEN"
                    self.open_time = time.time()


# Create circuit breaker for external services with configuration from settings
_vllm_circuit_breaker = CircuitBreaker(
    failure_threshold=settings.circuit_breaker.failure_threshold if hasattr(settings.circuit_breaker,
                                                                            'failure_threshold') else 3,
    recovery_timeout=settings.circuit_breaker.recovery_timeout if hasattr(settings.circuit_breaker,
                                                                          'recovery_timeout') else 60,
    reset_timeout=settings.circuit_breaker.reset_timeout if hasattr(settings.circuit_breaker, 'reset_timeout') else 300
)


def async_retry(max_retries=3, backoff_factor=1.5, circuit_breaker=None):
    """
    Decorator to retry async functions with exponential backoff.
    Integrates with circuit breaker for fault tolerance.

    Args:
        max_retries (int): Maximum number of retry attempts.
        backoff_factor (float): Factor to increase wait time between retries.
        circuit_breaker (CircuitBreaker): Optional circuit breaker instance.

    Returns:
        callable: Decorated function with retry logic.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_exception = None

            while retry_count < max_retries:
                # Check circuit breaker
                if circuit_breaker and circuit_breaker.is_open():
                    logger.warning(f"Circuit open, skipping call to {func.__name__}")
                    raise RuntimeError(f"Service unavailable: circuit is open for {func.__name__}")

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # Log execution time for monitoring
                    logger.debug(f"Function {func.__name__} completed: {execution_time:.4f}s")

                    # Record success to circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result
                except (TimeoutError, ConnectionError) as e:
                    # Record failure to circuit breaker for specific errors
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    last_exception = e

                    logger.warning(
                        f"Retry {retry_count}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.2f}s - Cause: {type(e).__name__}: {str(e)}"
                    )

                    # Wait before retrying
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.warning(f"async_retry exception: {e}")
                    # For other exceptions, record failure but don't retry
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    raise

            # If we get here, all retries failed
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise last_exception or RuntimeError(f"All retries failed for {func.__name__}")

        return wrapper

    return decorator


class RetrieverService:
    """
    Service to retrieve documents from various sources.

    Handles document retrieval and processing for chat context.
    """

    def __init__(self, request: ChatRequest):
        """
        Initialize the retrieval service.

        Args:
            request (ChatRequest): Chat request instance.
        """
        self.request = request
        self.documents = []
        self.document_processor = DocumentProcessor(settings)
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
            # Implementation for web search integration (could be added in future)
            # Actual web search logic would be added here

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


class LLMService:
    """
    Service for interacting with LLM models.

    Handles both Ollama and vLLM backends, with support for
    chain initialization, prompt building, and query execution.
    """

    def __init__(self, request: ChatRequest):
        """
        Initialize the LLM service.

        Args:
            request (ChatRequest): Chat request instance.
        """
        self.request = request
        self.settings_key = f"{request.meta.rag_sys_info}-{request.meta.session_id}"

        # Load query check dictionary
        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(settings, query_check_dict)

        # Initialize components based on backend
        if settings.llm.llm_backend.lower() == "ollama" and not settings.chat_history.enabled:
            self.chain = self._initialize_chain()
        elif settings.llm.llm_backend.lower() == "vllm" and not settings.chat_history.enabled:
            self.system_prompt_template = self._initialize_system_prompt_vllm()

        # Set default timeout based on settings
        self.timeout = getattr(settings.llm, 'timeout', 60)

        # For performance monitoring
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0
        }

    def _initialize_chain(self):
        """
        Initialize or retrieve a chain instance.

        Returns:
            chain: The initialized chain instance.

        Raises:
            Exception: If chain initialization fails.
        """
        start_time = time.time()
        session_id = self.request.meta.session_id
        try:
            prompt_template = self.response_generator.get_rag_qa_prompt(self.request.meta.rag_sys_info)
            chat_prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = get_or_create_chain(self.settings_key, mai_chat_llm, chat_prompt)

            logger.debug(
                f"[{session_id}] Chain initialization complete: {time.time() - start_time:.4f}s"
            )
            return chain
        except Exception as e:
            logger.error(
                f"[{session_id}] Chain initialization failed: {str(e)}",
                exc_info=True
            )
            raise

    def _initialize_system_prompt_vllm(self):
        """
        Initialize system prompt template for vLLM.

        Returns:
            str: System prompt template.

        Raises:
            Exception: If system prompt initialization fails.
        """
        session_id = self.request.meta.session_id
        try:
            template = self.response_generator.get_rag_qa_prompt(self.request.meta.rag_sys_info)
            logger.debug(f"[{session_id}] vLLM system prompt initialization complete")
            return template
        except Exception as e:
            logger.error(
                f"[{session_id}] vLLM system prompt initialization failed: {str(e)}",
                exc_info=True
            )
            raise

    def build_system_prompt(self, context):
        """
        Build system prompt with dynamic variables and chat history.

        Args:
            context (dict): Context with keys like {input}, {context}, {language}, {today}.

        Returns:
            str: Formatted system prompt.
        """
        session_id = self.request.meta.session_id
        try:
            prompt = self.system_prompt_template.format(**context)
            return prompt
        except KeyError as e:
            logger.error(f"[{session_id}] Key error in system prompt build: {str(e)}")
            # Try to format with a placeholder for the missing key
            missing_key = str(e).strip("'")
            context[missing_key] = f"[{missing_key} missing]"
            return self.system_prompt_template.format(**context)
        except Exception as e:
            logger.error(f"[{session_id}] System prompt build failed: {str(e)}")
            raise

    @async_retry(max_retries=2, backoff_factor=2, circuit_breaker=_vllm_circuit_breaker)
    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        Call vLLM endpoint with retry and circuit breaker.

        Args:
            data (VllmInquery): vLLM request data.

        Returns:
            dict: Response from vLLM endpoint.

        Raises:
            Exception: If vLLM endpoint call fails after retries.
        """
        start_time = time.time()
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] Calling vLLM endpoint (stream={data.stream})")

        vllm_url = settings.vllm.endpoint_url

        try:
            if data.stream:
                full_text = ""
                async for chunk in rc.restapi_stream_async(session_id, vllm_url, data):
                    # 실제 청크 내용 확인
                    logger.debug(f"Received chunk: {chunk}")

                    # 각 청크에서 텍스트 추출 및 누적
                    if 'new_text' in chunk:
                        text_chunk = chunk.get('new_text', '')
                        full_text += text_chunk

                        # 로깅 (옵션)
                        logger.debug(f"[{session_id}] Streaming chunk: {text_chunk}")

                    # 생성 완료 확인
                    if chunk.get('finished', False):
                        break

                # 스트리밍 완료 후 전체 텍스트 반환
                elapsed = time.time() - start_time
                logger.debug(f"[{session_id}] vLLM streaming response received: {elapsed:.4f}s elapsed")

                self.metrics["request_count"] += 1
                self.metrics["total_time"] += elapsed

                return {"generated_text": full_text}
            else:
                response = await rc.restapi_post_async(vllm_url, data)
                elapsed = time.time() - start_time
                logger.debug(f"[{session_id}] vLLM response received: {elapsed:.4f}s elapsed")

                self.metrics["request_count"] += 1
                self.metrics["total_time"] += elapsed

                return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM endpoint error: {elapsed:.4f}s after: {str(e)}")

            self.metrics["error_count"] += 1

            raise

    async def _stream_vllm_generate(self, session_id: str, url: str, data: VllmInquery):
        """
        스트리밍 생성을 위한 내부 제너레이터 메서드

        Args:
            url: 엔드포인트 URL
            data: VllmInquery 객체

        Yields:
            스트리밍 청크
        """
        try:
            session_id = self.request.meta.session_id
            async for chunk in await rc.restapi_stream_async(url, data):
                if chunk is not None:
                    yield chunk

                    # [DONE] 청크로 스트리밍 종료 확인
                    if chunk == '[DONE]':
                        break
        except Exception as e:
            logger.error(f"[{session_id}] vLLM streaming error: {str(e)}")
            raise

    async def ask(self, documents, language, streaming: bool = False):
        """
        Perform a query to the LLM.

        Args:
            documents (list): List of documents for context.
            language (str): Response language.
            streaming (bool): Whether to stream the response.

        Returns:
            str: Generated response.

        Raises:
            TimeoutError: If query exceeds timeout.
            Exception: If query fails.
        """
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] Starting LLM query")

        start_time = time.time()

        # Prepare context
        context = {
            "input": self.request.chat.user,
            "context": documents,
            "language": language,
            "today": self.response_generator.get_today(),
        }

        # Add VOC related context if needed
        if self.request.meta.rag_sys_info == "komico_voc":
            context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        result = None

        try:
            if settings.llm.llm_backend.lower() == "ollama":
                logger.debug(f"[{session_id}] Starting Ollama chain invocation")
                result = await limited_chain_invoke(self.chain, context, timeout=self.timeout)
                if inspect.iscoroutine(result):
                    result = await result
            elif settings.llm.llm_backend.lower() == "vllm":
                logger.debug(f"[{session_id}] Starting vLLM invocation")
                vllm_inquery_context = self.build_system_prompt(context)
                logger.error(context)
                logger.error(documents)

                vllm_request = VllmInquery(
                    request_id=session_id,
                    prompt=vllm_inquery_context,
                    stream=settings.llm.steaming_enabled
                )

                response = await self.call_vllm_endpoint(vllm_request)
                logger.debug(response)
                result = response.get("generated_text", "")

            elapsed = time.time() - start_time
            logger.info(
                f"[{session_id}] LLM query complete: {elapsed:.4f}s elapsed "
                f"[backend={settings.llm.llm_backend}]"
            )

            # Update service metrics
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            return result
        except TimeoutError as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{session_id}] LLM query timeout: {elapsed:.4f}s after {str(e)}"
            )

            self.metrics["error_count"] += 1

            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{session_id}] LLM query failed: {elapsed:.4f}s after: {type(e).__name__}: {str(e)}",
                exc_info=True
            )

            self.metrics["error_count"] += 1

            raise

    def get_metrics(self):
        """
        Get service performance metrics.

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


class ChatService:
    """
    Main service for handling chat interactions.

    This class coordinates the entire chat processing workflow including document retrieval,
    query optimization, LLM interactions, and response generation. It implements comprehensive
    performance monitoring, caching strategies, and asynchronous logging to ensure efficient
    and reliable chat processing.

    Features:
    - Caching for fast responses to repeated queries
    - Asynchronous logging for non-blocking operations
    - Performance tracking for each processing stage
    - Parallel processing of document retrieval and language settings
    - Integration with chat history (when enabled)
    - Comprehensive error handling

    Class Attributes:
        _prompt_cache: TTL cache for storing greeting and FAQ responses
        _response_cache: TTL cache for storing complete chat responses
        _circuit_breaker: Circuit breaker for external service protection
        _log_queue: Asynchronous queue for log messages
        _log_task: Task handling asynchronous log processing
        _log_initialized: Flag indicating if logging task is running
    """

    # Class-level caches for performance improvement
    _prompt_cache = TTLCache(maxsize=100, ttl=3600)  # 1 hour cache
    _response_cache = TTLCache(maxsize=200, ttl=900)  # 15 minute cache
    _circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=300)

    # Async logging queue
    _log_queue = asyncio.Queue()
    _log_task = None
    _log_initialized = False

    def __init__(self, request: ChatRequest):
        """
        Initialize the chat service with the given request.

        Args:
            request (ChatRequest): The chat request containing user query and metadata.
        """
        self.request = request
        self.retriever_service = RetrieverService(request)
        self.llm_service = LLMService(request)

        # Initialize history handler if chat history is enabled in settings
        self.history_handler = LlmHistoryHandler(mai_chat_llm, self.request) if settings.chat_history.enabled else None

        # Initialize processing components
        self.query_check_dict = QueryCheckDict(settings.lm_check.query_dict_config_path)
        self.response_generator = ResponseGenerator(settings, self.query_check_dict)
        self.query_processor = QueryProcessor(settings, self.query_check_dict)
        self.document_processor = DocumentProcessor(settings)
        self.voc_processor = VOCLinkProcessor(settings)
        self.search_engine = SearchEngine(settings)

        # Timer for performance monitoring
        self.start_time = time.time()
        self.processing_stages = {}

        # Create cache key based on session and query
        self.cache_key = f"{self.request.meta.rag_sys_info}:{self.request.meta.session_id}:{self.request.chat.user}"

    async def _ensure_log_task_running(self):
        """
        Ensure the async logging task is running, start it if not.

        This method checks if the logging task has been initialized and starts it
        if needed to handle asynchronous logging operations.
        """
        if not ChatService._log_initialized:
            ChatService._log_task = asyncio.create_task(self._process_logs())
            ChatService._log_initialized = True
            logger.info("Asynchronous logging system initialized")

    @classmethod
    def is_log_initialized(cls):
        """
        Check if the logging system is initialized.

        Returns:
            bool: True if the logging system is initialized, False otherwise.
        """
        return cls._log_initialized

    @classmethod
    def start_log_task(cls, coroutine):
        """
        Start the logging task with the given coroutine.

        Args:
            coroutine: The asynchronous coroutine to run.

        Returns:
            Task: The created asyncio task.
        """
        cls._log_task = asyncio.create_task(coroutine)
        cls._log_initialized = True
        return cls._log_task

    @classmethod
    def get_log_task(cls):
        """
        Get the current logging task.

        Returns:
            Task: The current logging task or None if not initialized.
        """
        return cls._log_task

    @classmethod
    async def _process_logs(cls):
        """
        Process log entries from the log queue asynchronously.

        This method runs in an infinite loop, pulling log entries from the queue
        and sending them to the appropriate logger methods. It handles exceptions
        gracefully to ensure the logging system remains operational.
        """
        while True:
            try:
                # Get log entry from queue
                log_entry = await cls._log_queue.get()
                level, message, extra = log_entry

                # Extract 'exc_info' parameter and remove from extra
                exc_info = extra.pop('exc_info', False) if isinstance(extra, dict) else False

                # Log to appropriate level
                if level == "debug":
                    logger.debug(message, exc_info=exc_info, extra=extra)
                elif level == "info":
                    logger.info(message, exc_info=exc_info, extra=extra)
                elif level == "warning":
                    logger.warning(message, exc_info=exc_info, extra=extra)
                elif level == "error":
                    logger.error(message, exc_info=exc_info, extra=extra)

                # Mark task as done
                cls._log_queue.task_done()
            except Exception as e:
                print(f"Log processing error: {e}")
                await asyncio.sleep(1)  # Wait before retry to avoid tight loop

    async def _log(self, level, message, **kwargs):
        """
        Add a log entry to the asynchronous logging queue.

        Args:
            level (str): Log level ("debug", "info", "warning", "error")
            message (str): Log message
            **kwargs: Additional parameters to pass to the logger
        """
        # Ensure logging task is running
        await self._ensure_log_task_running()
        await ChatService._log_queue.put((level, message, kwargs))

    def _record_stage(self, stage_name):
        """
        Record timing for a processing stage and return elapsed time.

        This method calculates the time elapsed since the last stage,
        records it in the processing_stages dictionary, and resets
        the start_time for the next stage.

        Args:
            stage_name (str): Name of the processing stage

        Returns:
            float: Elapsed time in seconds since the last recorded stage
        """
        current_time = time.time()
        elapsed = current_time - self.start_time
        self.processing_stages[stage_name] = elapsed
        self.start_time = current_time
        return elapsed

    async def process_chat(self) -> ChatResponse:
        """
        Process a chat request and generate an appropriate response.

        This method implements the complete workflow for chat query processing:
        - Detect greetings and filter invalid queries
        - Validate retrieval documents and optimize FAQ queries
        - Retrieve documents and determine language settings
        - Generate a response using LLM (with chat history if enabled)
        - Finalize the answer with references and formatting

        The method includes comprehensive error handling, caching for
        repeated queries, and detailed performance tracking.

        Returns:
            ChatResponse: Response object containing result, metadata, and context
        """
        # Ensure logging task is initialized
        await self._ensure_log_task_running()

        session_id = self.request.meta.session_id
        await self._log("info", f"[{session_id}] Starting chat request processing", session_id=session_id)

        self.start_time = time.time()
        self.processing_stages = {}

        # Check response cache for identical requests
        if self.cache_key in ChatService._response_cache:
            await self._log("info", f"[{session_id}] Returning cached response", session_id=session_id,
                            cache_key=self.cache_key)
            return ChatService._response_cache[self.cache_key]

        try:
            # Handle greetings or filter invalid queries
            response = await self._handle_greeting_or_filter()
            if response:
                elapsed = self._record_stage("greeting_filter")
                await self._log("info", f"[{session_id}] Greeting/filter stage complete: {elapsed:.4f}s elapsed",
                                session_id=session_id)
                # Cache simple responses
                ChatService._response_cache[self.cache_key] = response
                return response

            # Validate retrieval documents if filtering is enabled
            if settings.retriever.filter_flag:
                filter_res = self.document_processor.validate_retrieval_documents(self.request)
                if filter_res:
                    elapsed = self._record_stage("document_validation")
                    await self._log("info", f"[{session_id}] Document validation complete: {elapsed:.4f}s elapsed",
                                    session_id=session_id)
                    return filter_res

            # Optimize FAQ queries if applicable
            faq_query = self._handle_faq_query()
            if faq_query:
                self.request.chat.user = faq_query
                elapsed = self._record_stage("faq_optimization")
                await self._log("info", f"[{session_id}] FAQ query optimization: {elapsed:.4f}s elapsed",
                                session_id=session_id)

            # Start parallel tasks
            # Process document retrieval and language settings in parallel
            retrieval_task = asyncio.create_task(self._retrieve_documents())
            language_task = asyncio.create_task(self._get_language_settings())

            # Wait for tasks to complete
            await asyncio.gather(retrieval_task, language_task)

            # Collect results
            documents = retrieval_task.result()
            lang, trans_lang, reference_word = language_task.result()

            elapsed = self._record_stage("parallel_processing")
            await self._log(
                "info",
                f"[{session_id}] Parallel processing complete(prepare): {elapsed:.4f}s elapsed, "
                f"{len(documents)} documents retrieved",
                session_id=session_id
            )

            # Process LLM query with chat history or direct query
            start_llm_time = time.time()
            query_answer, vllm_retrival_document = await self._process_llm_query(documents, lang, trans_lang)

            llm_elapsed = time.time() - start_llm_time
            await self._log(
                "info",
                f"[{session_id}] LLM processing complete: {llm_elapsed:.4f}s "
                f"elapsed [backend={settings.llm.llm_backend}]",
                session_id=session_id
            )
            self._record_stage("llm_processing")

            # Clean generated answer by removing role markers
            cleaned_answer = re.sub(r'(AI:|Human:)', '', query_answer).strip()

            # Save chat history if enabled (run asynchronously without waiting)
            if settings.chat_history.enabled:
                await asyncio.create_task(self._save_chat_history(cleaned_answer))

            # Finalize answer with references and formatting
            start_finalize = time.time()
            final_query_answer = await self._finalize_answer_async(cleaned_answer, reference_word,
                                                                   vllm_retrival_document)

            finalize_elapsed = time.time() - start_finalize
            await self._log("debug", f"[{session_id}] Answer finalization complete: {finalize_elapsed:.4f}s elapsed",
                            session_id=session_id)
            self._record_stage("answer_finalization")

            # Create and return final response
            final_response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), final_query_answer)

            total_time = sum(self.processing_stages.values())
            await self._log(
                "info",
                f"[{session_id}] Chat processing complete: {total_time:.4f}s elapsed",
                session_id=session_id,
                stages=self.processing_stages
            )

            # Cache response for complex queries that take significant time
            if total_time > 1.0:  # Only cache queries that take more than 1 second
                ChatService._response_cache[self.cache_key] = final_response

            return final_response

        except ValueError as err:
            await self._log("error", f"[{session_id}] Value error during chat processing: {err}", session_id=session_id,
                            exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "Invalid input was provided.")
        except KeyError as err:
            await self._log("error", f"[{session_id}] Key error during chat processing: {err}", session_id=session_id,
                            exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "A required key was missing.")
        except Exception as err:
            await self._log("error", f"[{session_id}] Unexpected error during chat processing: {err}",
                            session_id=session_id, exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "Unable to process the request.")

    async def _retrieve_documents(self):
        """
        Retrieve documents asynchronously.

        This method runs the document retrieval in a separate thread pool
        to avoid blocking the main event loop, as document retrieval can be
        a time-consuming operation.

        Returns:
            list: Retrieved documents
        """
        # Run synchronous document retrieval in thread pool to avoid blocking
        return await asyncio.to_thread(self.retriever_service.retrieve_documents)

    async def _get_language_settings(self):
        """
        Get language settings asynchronously.

        This method retrieves the appropriate language settings based on
        the request language. It's a lightweight operation but is executed
        asynchronously for consistency with the parallel processing pattern.

        Returns:
            tuple: (language, translation_language, reference_word)
        """
        # This could be run in a separate thread, but it's a lightweight operation
        return self.response_generator.get_translation_language_word(self.request.chat.lang)

    async def _process_llm_query(self, documents, lang, trans_lang):
        """
        Process the LLM query with or without chat history.

        This method handles the core interaction with the LLM, using either
        chat history (if enabled) or direct query processing. It supports
        both Ollama and vLLM backends.

        Args:
            documents (list): List of documents for context
            lang (str): Response language
            trans_lang (str): Translation language (if different)

        Returns:
            tuple: (query_answer, vllm_retrieval_document)
        """
        session_id = self.request.meta.session_id
        query_answer = None
        vllm_retrival_document = None

        try:
            if settings.chat_history.enabled:
                # Initialize retriever with documents
                await self.history_handler.init_retriever(documents)

                if settings.llm.llm_backend.lower() == "ollama":
                    # Process chat with history for Ollama
                    await self._log("debug", f"[{session_id}] Processing chat with history for Ollama backend")
                    rag_chat_chain = self.history_handler.init_chat_chain_with_history()
                    chat_history_response = await self.history_handler.handle_chat_with_history(
                        self.request, trans_lang, rag_chat_chain
                    ) or {"context": [], "answer": ""}

                    # Update documents with history context
                    context = chat_history_response.get("context", [])
                    self.request.chat.payload = self.document_processor.convert_document_to_payload(
                        context
                    ) + self.request.chat.payload

                    # Update documents
                    # documents = context + documents

                    query_answer = chat_history_response.get("answer", "")

                elif settings.llm.llm_backend.lower() == "vllm":
                    # Process chat with history for vLLM
                    await self._log("debug", f"[{session_id}] Processing chat with history for vLLM backend")
                    chat_history_result = await self.history_handler.handle_chat_with_history_vllm(
                        self.request, trans_lang
                    )

                    if chat_history_result:
                        query_answer = chat_history_result[0]
                        vllm_retrival_document = chat_history_result[1]
                    else:
                        query_answer = ""
                        vllm_retrival_document = []
            else:
                # Direct LLM query without chat history
                await self._log("debug", f"[{session_id}] Processing direct LLM query without chat history")
                query_answer = await self.llm_service.ask(documents, lang)

            return query_answer, vllm_retrival_document

        except Exception as e:
            await self._log("error", f"[{session_id}] Error processing LLM query: {str(e)}", exc_info=True)
            raise

    async def _save_chat_history(self, answer):
        """
        Save chat history to Redis asynchronously.

        This method stores the current conversation exchange in Redis
        for future reference. It's designed to work asynchronously to
        avoid blocking the main processing flow.

        Args:
            answer (str): The generated answer to save
        """
        session_id = self.request.meta.session_id
        try:
            await self._log("debug", f"[{session_id}] Saving chat history to Redis")
            await RedisUtils.async_save_message_to_redis(
                system_info=self.request.meta.rag_sys_info,
                session_id=session_id,
                message=create_chat_data(session_id, [
                    create_message("HumanMessage", self.request.chat.user),
                    create_message("AIMessage", answer)
                ])
            )
            await self._log("debug", f"[{session_id}] Chat history saved successfully")
        except Exception as e:
            await self._log(
                "error",
                f"[{session_id}] Failed to save chat history: {str(e)}",
                exc_info=True
            )
            # Continue processing even if history saving fails

    async def _handle_greeting_or_filter(self):
        """
        Handle greetings or filter invalid queries using predefined responses.

        This method checks if the query matches predefined greeting patterns
        or should be filtered based on content rules. It utilizes caching to
        improve performance for repeated queries.

        Returns:
            ChatResponse or None: Response object if greeting/filtering applies, None otherwise
        """
        session_id = self.request.meta.session_id

        # Check for cached result
        greeting_cache_key = f"greeting:{session_id}:{self.request.chat.user}"
        if greeting_cache_key in ChatService._prompt_cache:
            await self._log("debug", f"[{session_id}] Using cached greeting response")
            return ChatService._prompt_cache[greeting_cache_key]

        # Check for predefined greeting or response patterns
        check_res = self.query_processor.check_query_sentence(self.request)
        if check_res:
            await self._log("debug", f"[{session_id}] Greeting detected")
            response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), check_res)
            # Cache greeting response
            ChatService._prompt_cache[greeting_cache_key] = response
            return response

        # Apply query filtering if enabled
        if settings.query_filter.enabled and not self.query_processor.filter_query(self.request.chat.user):
            await self._log("debug", f"[{session_id}] Query filtered")
            farewell_msg = random.choice(
                self.query_check_dict.get_dict_data(self.request.chat.lang, "farewells_msg")
            )
            response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), farewell_msg)
            # Cache filtered response
            ChatService._prompt_cache[greeting_cache_key] = response
            return response

        return None

    def _handle_faq_query(self):
        """
        Optimize FAQ queries for better response accuracy.

        This method attempts to reconstruct the query to better match
        FAQ patterns, improving the likelihood of finding relevant
        answers in the knowledge base.

        Returns:
            str or None: Optimized query string if successful, None otherwise
        """
        session_id = self.request.meta.session_id
        try:
            # Check FAQ query cache
            faq_cache_key = f"faq:{session_id}:{self.request.chat.user}"
            if faq_cache_key in ChatService._prompt_cache:
                logger.debug(f"[{session_id}] Using cached FAQ query optimization")
                return ChatService._prompt_cache[faq_cache_key]

            faq_query = self.query_processor.construct_faq_query(self.request)
            if faq_query:
                logger.debug(f"[{session_id}] FAQ query optimized: {self.request.chat.user} -> {faq_query}")
                # Cache successful FAQ query optimization
                ChatService._prompt_cache[faq_cache_key] = faq_query

            return faq_query
        except Exception as e:
            logger.warning(
                f"[{session_id}] Failed to optimize FAQ query: {str(e)}",
                exc_info=True
            )
            return None

    async def _finalize_answer_async(self, query_answer, reference_word, retrieval_document=None):
        """
        Add references and finalize the response asynchronously.

        This method enhances the generated answer by adding source references,
        processing VOC-specific content, and formatting URLs as clickable links.
        Heavy processing tasks are run in a thread pool.

        Args:
            query_answer (str): The raw answer from LLM
            reference_word (str): Reference word to use in citations
            retrieval_document (list, optional): List of retrieval documents for vLLM

        Returns:
            str: Finalized and formatted answer
        """
        session_id = self.request.meta.session_id

        try:
            # Add source references
            if settings.prompt.source_count:
                start_time = time.time()
                # Run heavy tasks in thread pool
                if retrieval_document is not None:
                    query_answer = await asyncio.to_thread(
                        self.response_generator.make_answer_reference,
                        query_answer,
                        self.request.meta.rag_sys_info,
                        reference_word,
                        retrieval_document,
                        self.request
                    )
                else:
                    query_answer = await asyncio.to_thread(
                        self.response_generator.make_answer_reference,
                        query_answer,
                        self.request.meta.rag_sys_info,
                        reference_word,
                        self.retriever_service.documents,
                        self.request
                    )
                await self._log(
                    "debug",
                    f"[{session_id}] References added: {time.time() - start_time:.4f}s elapsed",
                    session_id=session_id
                )

            # Process VOC
            if "komico_voc" in settings.voc.voc_type.split(',') and self.request.meta.rag_sys_info == "komico_voc":
                start_time = time.time()
                result = await asyncio.to_thread(
                    self.voc_processor.make_komico_voc_groupware_docid_url,
                    query_answer
                )
                await self._log(
                    "debug",
                    f"[{session_id}] VOC processing complete: {time.time() - start_time:.4f}s elapsed",
                    session_id=session_id
                )
                return result

            # Process URL links
            start_time = time.time()
            result = await asyncio.to_thread(
                self.search_engine.replace_urls_with_links,
                query_answer
            )
            await self._log(
                "debug",
                f"[{session_id}] URL link processing complete: {time.time() - start_time:.4f}s elapsed",
                session_id=session_id
            )
            return result

        except Exception as e:
            await self._log("error", f"[{session_id}] Error during answer finalization: {str(e)}",
                            session_id=session_id, exc_info=True)
            # Return original answer if finalization fails
            return query_answer

    def _create_response(self, error_code, system_msg):
        """
        Create a ChatResponse object with appropriate metadata.

        This method constructs the response object with error code,
        system message, metadata, and payload information. It also
        adds performance data if supported.

        Args:
            error_code (dict): Error code information
            system_msg (str): System response message

        Returns:
            ChatResponse: Complete response object
        """
        session_id = self.request.meta.session_id

        try:
            # Prepare payload - optimize for large data
            request_payload = self.request.chat.payload or []

            payloads = [
                PayloadRes(doc_name=doc.doc_name, doc_page=doc.doc_page, content=doc.content)
                for doc in request_payload
            ]

            # Create and return response object
            response = ChatResponse(
                result_cd=error_code.get("code"),
                result_desc=error_code.get("desc"),
                meta=MetaRes(
                    company_id=self.request.meta.company_id,
                    dept_class=self.request.meta.dept_class,
                    session_id=session_id,
                    rag_sys_info=self.request.meta.rag_sys_info,
                ),
                chat=ChatRes(
                    user=self.request.chat.user,
                    system=system_msg,
                    category1=self.request.chat.category1,
                    category2=self.request.chat.category2,
                    category3=self.request.chat.category3,
                    info=payloads,
                )
            )

            # Add performance data if supported
            if hasattr(response, 'add_performance_data'):
                total_time = sum(self.processing_stages.values()) if self.processing_stages else 0
                response.add_performance_data({
                    "total_processing_time": total_time,
                    "processing_stages": self.processing_stages,
                    "llm_metrics": self.llm_service.get_metrics() if hasattr(self.llm_service, 'get_metrics') else {}
                })

                logger.debug(
                    f"[{session_id}] Performance data added to response: {total_time:.4f}s total processing time")

            return response

        except Exception as e:
            logger.error(f"[{session_id}] Error creating response object: {str(e)}", exc_info=True)
            # Create fallback response on error
            return ChatResponse(
                result_cd=ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION).get("code"),
                result_desc=ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION).get("desc"),
                meta=MetaRes(
                    company_id=self.request.meta.company_id if hasattr(self.request.meta, 'company_id') else "",
                    dept_class=self.request.meta.dept_class if hasattr(self.request.meta, 'dept_class') else "",
                    session_id=session_id if hasattr(self.request.meta, 'session_id') else "",
                    rag_sys_info=self.request.meta.rag_sys_info if hasattr(self.request.meta, 'rag_sys_info') else "",
                ),
                chat=ChatRes(
                    user=self.request.chat.user if hasattr(self.request.chat, 'user') else "",
                    system="An error occurred while generating the response.",
                    category1="",
                    category2="",
                    category3="",
                    info=[],
                )
            )

    @classmethod
    def clear_caches(cls):
        """
        Clear all caches in the service.

        This class method empties both the prompt and response caches,
        which can be useful during testing or when deploying updates.
        """
        cls._prompt_cache.clear()
        cls._response_cache.clear()
        logger.info("All ChatService caches have been cleared")
