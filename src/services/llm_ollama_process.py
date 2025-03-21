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
import json
import logging
import random
import re
import time
from asyncio import Semaphore, create_task, wait_for, TimeoutError
from contextlib import asynccontextmanager
from functools import lru_cache, wraps
from threading import Lock
from typing import Dict, Tuple

from cachetools import TTLCache
from fastapi import FastAPI, BackgroundTasks
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from starlette.responses import StreamingResponse

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

    @classmethod
    def is_gemma_model(cls) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # OLLAMA 백엔드인 경우
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # VLLM 백엔드인 경우
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # 기본적으로 False 반환
        return False

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
            # 이미지 데이터 처리
            if (settings.llm.llm_backend.lower() == "vllm" and
                    hasattr(self.request.chat, 'image') and
                    self.request.chat.image):

                # 이미지 데이터가 있고 vLLM 백엔드인 경우만 처리
                image_data = self.request.chat.image

                # 이미지 정보를 프롬프트에 추가
                if 'image_description' not in context:
                    context['image_description'] = self._format_image_data(image_data)

                    # 프롬프트 템플릿에 이미지 설명 토큰이 없으면 입력 앞에 추가
                    if '{image_description}' not in self.system_prompt_template:
                        insert_point = self.system_prompt_template.find('{input}')
                        if insert_point > 0:
                            image_instruction = "\n\n# 이미지 정보\n다음은 사용자가 제공한 이미지에 대한 정보입니다:\n{image_description}\n\n# 질문\n"
                            self.system_prompt_template = (
                                    self.system_prompt_template[:insert_point] +
                                    image_instruction +
                                    self.system_prompt_template[insert_point:]
                            )

            prompt = self.system_prompt_template.format(**context)
            return prompt
        except KeyError as e:
            # Set missing key to empty string and log as warning instead of error
            missing_key = str(e).strip("'")
            logger.warning(f"[{session_id}] Key missing in system prompt build: {missing_key}")
            context[missing_key] = ""
            return self.system_prompt_template.format(**context)
        except Exception as e:
            logger.error(f"[{session_id}] System prompt build failed: {str(e)}")
            raise

    def build_system_prompt_gemma(self, context):
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성합니다.

        Args:
            context (dict): 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        session_id = self.request.meta.session_id
        try:
            # 이미지 데이터 처리
            if hasattr(self.request.chat, 'image') and self.request.chat.image:
                # 이미지 정보를 프롬프트에 추가
                if 'image_description' not in context:
                    context['image_description'] = self._format_image_data(self.request.chat.image)

                    # 프롬프트 템플릿에 이미지 설명 토큰이 없으면 입력 앞에 추가
                    if '{image_description}' not in self.system_prompt_template:
                        insert_point = self.system_prompt_template.find('{input}')
                        if insert_point > 0:
                            image_instruction = "\n\n# 이미지 정보\n다음은 사용자가 제공한 이미지에 대한 정보입니다:\n{image_description}\n\n# 질문\n"
                            self.system_prompt_template = (
                                    self.system_prompt_template[:insert_point] +
                                    image_instruction +
                                    self.system_prompt_template[insert_point:]
                            )

            # 먼저 기존 함수로 프롬프트 생성
            raw_prompt = self.build_system_prompt(context)

            # Gemma 형식으로 변환
            # <start_of_turn>user 형식으로 시작
            formatted_prompt = "<start_of_turn>user\n"

            # 시스템 프롬프트 삽입
            formatted_prompt += raw_prompt

            # 사용자 입력부 종료 및 모델 응답 시작
            formatted_prompt += "\n<end_of_turn>\n<start_of_turn>model\n"

            return formatted_prompt

        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"[{session_id}] 시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt_gemma(context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"[{session_id}] Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n<end_of_turn>\n<start_of_turn>model\n"
            return basic_prompt

    @classmethod
    def _format_image_data(cls, image_data: Dict[str, str]) -> str:
        """
        이미지 데이터를 프롬프트에 추가하기 위한 형식으로 변환합니다.

        Args:
            image_data (Dict[str, str]): 이미지 데이터 (base64, URL 등)

        Returns:
            str: 포맷된 이미지 정보
        """
        # 이미지 데이터 형식에 따라 적절한 설명 생성
        if 'base64' in image_data:
            return "[이미지 데이터가 base64 형식으로 전달되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"
        elif 'url' in image_data:
            return f"[이미지 URL: {image_data.get('url')}]"
        elif 'description' in image_data:
            return f"[이미지 설명: {image_data.get('description')}]"
        else:
            return "[이미지 데이터가 제공되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"

    @async_retry(max_retries=2, backoff_factor=2, circuit_breaker=_vllm_circuit_breaker)
    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        Call vLLM endpoint with retry and circuit breaker.

        Args:
            data (VllmInquery): vLLM request data.

        Returns:
            Dict, AsyncGenerator: 스트리밍 모드에 따라 전체 응답 또는 청크 생성기 반환

        Raises:
            Exception: If vLLM endpoint call fails after retries.
        """
        start_time = time.time()
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] Calling vLLM endpoint (stream={data.stream})")

        # circuit_breaker 확인
        if _vllm_circuit_breaker.is_open():
            logger.warning(f"[{session_id}] circuit_breaker 열려 있어 요청을 건너뜁니다.")
            raise RuntimeError("vLLM 서비스 사용할 수 없음: circuit_breaker가 열려 있습니다")

        vllm_url = settings.vllm.endpoint_url

        try:
            response = await rc.restapi_post_async(vllm_url, data)
            elapsed = time.time() - start_time
            logger.debug(f"[{session_id}] vLLM response received: {elapsed:.4f}s elapsed")

            # 메트릭 업데이트
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            # circuit_breaker 업데이트
            _vllm_circuit_breaker.record_success()

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM endpoint error: {elapsed:.4f}s after: {str(e)}")

            self.metrics["error_count"] += 1

            # circuit_breaker 업데이트
            _vllm_circuit_breaker.record_failure()

            raise

    async def _stream_vllm_response(self, session_id: str, url: str, data: VllmInquery):
        """
        vLLM에서 스트리밍 응답을 받아 처리하는 비동기 제너레이터입니다.

        Args:
            session_id: 세션 ID
            url: vLLM 엔드포인트 URL
            data: vLLM 요청 데이터

        Yields:
            Dict: 응답 청크
        """
        from src.common.restclient import rc
        start_time = time.time()

        try:
            logger.debug(f"[{session_id}] vLLM 스트리밍 시작")

            # RestClient의 스트리밍 메서드를 호출하여 청크 처리
            async for chunk in rc.restapi_stream_async(session_id, url, data):
                if chunk is None:
                    continue

                # 청크 처리 및 표준화
                processed_chunk = self._process_vllm_chunk(chunk)

                # 청크 로깅 (텍스트가 있는 경우 길이만 기록)
                log_chunk = processed_chunk.copy()
                if 'new_text' in log_chunk:
                    log_chunk['new_text'] = f"<{len(log_chunk['new_text'])}자 길이의 텍스트>"
                logger.debug(f"[{session_id}] 청크 처리: {log_chunk}")

                yield processed_chunk

                # 마지막 청크 처리
                if processed_chunk.get('finished', False) or processed_chunk.get('error', False):
                    # 회로 차단기 성공 기록
                    _vllm_circuit_breaker.record_success()

                    # 메트릭 업데이트
                    elapsed = time.time() - start_time
                    self.metrics["request_count"] += 1
                    self.metrics["total_time"] += elapsed

                    logger.debug(f"[{session_id}] vLLM 스트리밍 완료: {elapsed:.4f}초 소요")
                    break

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM 스트리밍 오류: {elapsed:.4f}초 후: {str(e)}")

            # 메트릭 업데이트
            self.metrics["error_count"] += 1

            # 회로 차단기 실패 기록
            _vllm_circuit_breaker.record_failure()

            # 오류 청크 반환
            yield {
                "error": True,
                "message": f"스트리밍 오류: {str(e)}",
                "finished": True
            }

    @classmethod
    def _process_vllm_chunk(cls, chunk):
        """
        vLLM 응답 청크를 표준 형식으로 처리합니다.

        Args:
            chunk: 원시 vLLM 응답 청크

        Returns:
            Dict: 처리된 청크
        """
        # 오류 확인
        if 'error' in chunk:
            return {
                'error': True,
                'message': chunk.get('message', '알 수 없는 오류'),
                'finished': True
            }

        # 종료 마커 확인
        if chunk == '[DONE]':
            return {
                'new_text': '',
                'finished': True
            }

        # vLLM의 다양한 응답 형식 처리
        if isinstance(chunk, dict):
            # 텍스트 청크 (일반 스트리밍)
            if 'new_text' in chunk:
                return {
                    'new_text': chunk['new_text'],
                    'finished': chunk.get('finished', False)
                }
            # 완료 신호
            elif 'finished' in chunk and chunk['finished']:
                return {
                    'new_text': '',
                    'finished': True
                }
            # 전체 텍스트 응답 (비스트리밍 형식)
            elif 'generated_text' in chunk:
                return {
                    'new_text': chunk['generated_text'],
                    'finished': True
                }
            # OpenAI 호환 형식
            elif 'delta' in chunk:
                return {
                    'new_text': chunk['delta'].get('content', ''),
                    'finished': chunk.get('finished', False)
                }
            # 알 수 없는 형식
            else:
                return chunk

        # 문자열 응답 (드문 경우)
        elif isinstance(chunk, str):
            return {
                'new_text': chunk,
                'finished': False
            }

        # 기타 타입 처리
        return {
            'new_text': str(chunk),
            'finished': False
        }

    async def stream_response(self, documents, language):
        """
        스트리밍 응답을 제공하는 메서드.

        Args:
            documents (list): 컨텍스트용 문서 목록
            language (str): 응답 언어

        Returns:
            AsyncGenerator: 응답 청크를 생성하는 비동기 제너레이터
        """
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] 스트리밍 응답 시작")

        # 컨텍스트 준비
        context = {
            "input": self.request.chat.user,
            "context": documents,
            "language": language,
            "today": self.response_generator.get_today(),
        }

        # VOC 관련 컨텍스트 추가 (필요한 경우)
        if self.request.meta.rag_sys_info == "komico_voc":
            context.update({
                "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                "check_gw_word_link": settings.voc.check_gw_word_link,
                "check_gw_word": settings.voc.check_gw_word,
                "check_block_line": settings.voc.check_block_line,
            })

        try:
            # vLLM 엔진만 현재 스트리밍 지원
            if settings.llm.llm_backend.lower() == "vllm":
                # 모델이 Gemma인지 확인
                use_gemma_format = self.__class__.is_gemma_model()

                if use_gemma_format:
                    logger.debug(f"[{session_id}] Gemma 모델 감지됨, Gemma 형식 사용")
                    vllm_inquery_context = self.build_system_prompt_gemma(context)
                else:
                    # 시스템 프롬프트 생성
                    vllm_inquery_context = self.build_system_prompt(context)

                # 스트리밍을 위한 vLLM 요청 준비
                vllm_request = VllmInquery(
                    request_id=session_id,
                    prompt=vllm_inquery_context,
                    stream=True  # 스트리밍 모드 활성화
                )

                # vLLM 스트리밍 엔드포인트 호출
                vllm_url = settings.vllm.endpoint_url

                # 스트리밍 응답 제공
                async for chunk in self._stream_vllm_generate(session_id, vllm_url, vllm_request):
                    yield chunk

            else:
                # 다른 백엔드는 스트리밍을 지원하지 않음 - 에러 응답
                logger.error(f"[{session_id}] 스트리밍은 vLLM 백엔드에서만 지원됩니다.")
                yield {
                    "error": True,
                    "message": "스트리밍은 vLLM 백엔드에서만 지원됩니다."
                }

        except Exception as e:
            logger.error(f"[{session_id}] 스트리밍 응답 중 오류 발생: {str(e)}", exc_info=True)
            yield {"error": True, "message": str(e)}

    @classmethod
    async def _stream_vllm_generate(cls, session_id: str, url: str, data: VllmInquery):
        """
        개선된 vLLM 스트리밍 제너레이터

        Args:
            session_id: 세션 ID
            url: vLLM 엔드포인트 URL
            data: vLLM 요청 데이터

        Yields:
            dict: 스트리밍 청크
        """
        from src.common.restclient import rc

        try:
            # StreamingResponse 구현을 위한 제너레이터
            async for chunk in rc.restapi_stream_async(session_id, url, data):
                if chunk is None:
                    continue

                # 올바른 형식으로 청크 반환
                if isinstance(chunk, dict):
                    # 새로운 텍스트가 있는 경우
                    if 'new_text' in chunk:
                        yield {
                            "text": chunk['new_text'],
                            "finished": chunk.get('finished', False)
                        }
                    # 완료 신호만 있는 경우
                    elif chunk.get('finished', False):
                        yield {"text": "", "finished": True}
                elif chunk == '[DONE]':
                    # 종료 마커
                    yield {"text": "", "finished": True}
                else:
                    # 기타 형식의 청크
                    yield {"text": str(chunk), "finished": False}

        except Exception as e:
            logger.error(f"[{session_id}] vLLM 스트리밍 오류: {str(e)}")
            yield {"error": True, "message": str(e)}

    async def ask(self, documents, language):
        """
        Perform a query to the LLM.

        Args:
            documents (list): List of documents for context.
            language (str): Response language.

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

        # 이미지 데이터가 있는 경우 추가
        if hasattr(self.request.chat, 'image') and self.request.chat.image:
            context["image_description"] = self._format_image_data(self.request.chat.image)

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

                # 모델이 Gemma인지 확인 (클래스 메서드로 호출)
                use_gemma_format = self.__class__.is_gemma_model()

                if use_gemma_format:
                    logger.debug(f"[{session_id}] Gemma 모델 감지됨, Gemma 형식 사용")
                    vllm_inquery_context = self.build_system_prompt_gemma(context)
                else:
                    vllm_inquery_context = self.build_system_prompt(context)

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

    # 캐시 정리 주기
    _cache_cleanup_interval = 3600  # 1시간마다 캐시 정리
    _last_cache_cleanup = time.time()

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
        self.history_handler = LlmHistoryHandler(
            mai_chat_llm,
            self.request,
            max_history_turns=getattr(settings.chat_history, 'max_turns', 10)
        )

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

        # 캐시 정리 확인
        self._check_cache_cleanup()

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
    def is_gemma_model(cls) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # OLLAMA 백엔드인 경우
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # VLLM 백엔드인 경우
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # 기본적으로 False 반환
        return False

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

    async def stream_chat(self, background_tasks: BackgroundTasks = None) -> StreamingResponse:
        """
        문자 단위 처리를 통한 실시간 스트리밍 응답 제공, 히스토리 지원 기능 추가
        개선된 2단계 접근법으로 대화 이력을 처리

        Returns:
            StreamingResponse: 스트리밍 응답 객체
        """
        session_id = self.request.meta.session_id
        await self._log("info", f"[{session_id}] 문자 단위 스트리밍 채팅 요청 시작", session_id=session_id)

        # 처리 시간 측정 시작
        self.start_time = time.time()
        self.processing_stages = {}

        try:
            # 인사말 또는 간단한 응답 처리
            greeting_response = await self._handle_greeting_or_filter()
            if greeting_response:
                elapsed = self._record_stage("greeting_filter")
                await self._log("info", f"[{session_id}] 인사말/필터 단계 완료: {elapsed:.4f}초 소요", session_id=session_id)

                # 간단한 응답은 바로 반환
                async def simple_stream():
                    # 일반 응답 청크
                    error_data = {'text': greeting_response.chat.system, 'finished': False}
                    json_str = json.dumps(error_data, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"
                    # 완료 신호 전송
                    error_data = {'text': '', 'finished': True}
                    json_str = json.dumps(error_data, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"
                    # 전체 응답 전송
                    error_data = {'complete_response': greeting_response.chat.system}
                    json_str = json.dumps(error_data, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"

                return StreamingResponse(simple_stream(), media_type="text/event-stream")

            # 문서 검색 및 언어 설정 병렬 처리
            language_task = asyncio.create_task(self._get_language_settings())
            await language_task

            # 결과 수집
            lang, trans_lang, reference_word = language_task.result()
            elapsed = self._record_stage("language_processing")
            await self._log("info", f"[{session_id}] 언어 처리 완료: {elapsed:.4f}초 소요", session_id=session_id)

            # vLLM 백엔드 확인
            if settings.llm.llm_backend.lower() != "vllm":
                await self._log("error", f"[{session_id}] 스트리밍은 vLLM 백엔드에서만 지원됩니다.")

                # 오류 메시지 스트림으로 반환
                async def error_stream():
                    error_msg = "스트리밍은 vLLM 백엔드에서만 지원됩니다."
                    error_data = {'error': True, 'text': error_msg, 'finished': True}
                    json_str = json.dumps(error_data, ensure_ascii=False)
                    yield f"data: {json_str}\n\n"

                return StreamingResponse(error_stream(), media_type="text/event-stream")

            # 문자 단위 처리 프로세서 초기화 (빈 문서 리스트로 시작)
            empty_documents = []  # 명시적으로 빈 문서 리스트 생성
            post_processor = StreamResponsePostProcessor(
                self.response_generator,
                self.voc_processor,
                self.search_engine,
                self.request,
                empty_documents  # 빈 문서 리스트로 시작 (나중에 업데이트)
            )

            # vLLM 요청 준비
            vllm_url = settings.vllm.endpoint_url

            # 히스토리 기능이 활성화된 경우의 처리
            if settings.chat_history.enabled and hasattr(self, 'history_handler'):
                await self._log("info", f"[{session_id}] 히스토리 기능이 활성화된 상태로 스트리밍 처리 진행", session_id=session_id)

                # 개선된 히스토리 처리 사용 여부 확인
                use_improved_history = getattr(settings.llm, 'use_improved_history', False)

                # 히스토리 핸들러에 검색기 초기화
                await self.history_handler.init_retriever(empty_documents)

                # 모델 유형에 따른 스트리밍 핸들러 디스패치
                if self.__class__.is_gemma_model():
                    logger.info(f"[{session_id}] Gemma 모델 감지됨, Gemma 스트리밍 핸들러로 처리")
                    vllm_request, retrieval_document = await self.history_handler.handle_chat_with_history_gemma_streaming(
                        self.request, trans_lang
                    )
                else:
                    # 기존 모델을 위한 처리
                    if settings.llm.steaming_enabled and settings.llm.llm_backend.lower() == "vllm":
                        # 개선된 히스토리 처리 기능 확인
                        if getattr(settings.llm, 'use_improved_history', False):
                            # 개선된 2단계 접근법 사용
                            vllm_request, retrieval_document = await self.history_handler.handle_chat_with_history_vllm_streaming_improved(
                                self.request, trans_lang
                            )
                        else:
                            # 기존 방식 사용
                            vllm_request, retrieval_document = await self.history_handler.handle_chat_with_history_vllm_streaming(
                                self.request, trans_lang
                            )
                    else:
                        # 스트리밍이 지원되지 않거나 vLLM이 아닌 경우
                        logger.error(f"[{session_id}] 스트리밍은 vLLM 백엔드에서만 지원됩니다.")

                        # 오류 스트림 생성
                        async def error_stream():
                            error_msg = "스트리밍은 vLLM 백엔드에서만 지원됩니다."
                            error_data = {'error': True, 'text': error_msg, 'finished': True}
                            json_str = json.dumps(error_data, ensure_ascii=False)
                            yield f"data: {json_str}\n\n"

                        return StreamingResponse(error_stream(), media_type="text/event-stream")

                logger.info(f"[{session_id}] vLLM 요청 준비 완료: {use_improved_history and '개선된 방식' or '기존 방식'}")

                # 문서 업데이트
                if retrieval_document:
                    await self._log("debug", f"[{session_id}] 검색된 문서로 업데이트: {len(retrieval_document)}개",
                                    session_id=session_id)
                    # 포스트 프로세서의 문서도 업데이트
                    post_processor.documents = retrieval_document
            else:
                # 히스토리 없는 일반 스트리밍 처리
                # 컨텍스트 준비
                await self._log("info", f"[{session_id}] 히스토리 없는 일반 스트리밍 처리 시작", session_id=session_id)

                # 문서 검색
                retrieval_task = asyncio.create_task(self._retrieve_documents())
                await retrieval_task
                documents = retrieval_task.result()

                # 포스트 프로세서 업데이트
                post_processor.documents = documents

                context = {
                    "input": self.request.chat.user,
                    "context": documents,
                    "language": lang,
                    "today": self.response_generator.get_today(),
                }

                # 이미지 데이터가 있는 경우 추가
                if hasattr(self.request.chat, 'image') and self.request.chat.image:
                    # 이미지 정보를 포맷팅하여 컨텍스트에 추가
                    if hasattr(self.llm_service, '_format_image_data'):
                        context["image_description"] = self.llm_service._format_image_data(self.request.chat.image)
                    else:
                        # llm_service에 해당 메서드가 없을 경우 간단한 설명 생성
                        if 'base64' in self.request.chat.image:
                            context["image_description"] = "[이미지 데이터가 base64 형식으로 전달되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"
                        elif 'url' in self.request.chat.image:
                            context["image_description"] = f"[이미지 URL: {self.request.chat.image.get('url')}]"
                        else:
                            context["image_description"] = "[이미지 데이터가 제공되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"

                # VOC 관련 설정 추가
                if self.request.meta.rag_sys_info == "komico_voc":
                    context.update({
                        "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                        "check_gw_word_link": settings.voc.check_gw_word_link,
                        "check_gw_word": settings.voc.check_gw_word,
                        "check_block_line": settings.voc.check_block_line,
                    })

                # 시스템 프롬프트 생성
                system_prompt_template = self.llm_service._initialize_system_prompt_vllm()

                # 이미지 정보가 있고 시스템 프롬프트에 관련 토큰이 없으면 추가
                if "image_description" in context and "{image_description}" not in system_prompt_template:
                    insert_point = system_prompt_template.find("{input}")
                    if insert_point > 0:
                        image_instruction = "\n\n# 이미지 정보\n다음은 사용자가 제공한 이미지에 대한 정보입니다:\n{image_description}\n\n# 질문\n"
                        system_prompt_template = (
                                system_prompt_template[:insert_point] +
                                image_instruction +
                                system_prompt_template[insert_point:]
                        )

                # 시스템 프롬프트 생성
                vllm_inquery_context = self.llm_service.build_system_prompt(context)

                # 일반 vLLM 요청 준비
                vllm_request = VllmInquery(
                    request_id=session_id,
                    prompt=vllm_inquery_context,
                    stream=True  # 스트리밍 모드 활성화
                )

            # 문자 단위 배치 설정
            char_buffer = ""  # 문자 버퍼
            max_buffer_time = 100  # 최대 100ms 지연 허용
            min_chars_to_send = 2  # 최소 2자 이상일 때 전송 (한글 자모 조합 고려)
            last_send_time = time.time()  # 마지막 전송 시간

            # 스트리밍 응답 생성
            async def generate_stream():
                nonlocal char_buffer, last_send_time
                start_llm_time = time.time()
                error_occurred = False
                full_response = None  # 전체 응답을 저장할 변수

                try:
                    # 스트리밍 처리
                    async for chunk in rc.restapi_stream_async(session_id, vllm_url, vllm_request):
                        current_time = time.time()

                        # 빈 청크 스킵
                        if chunk is None:
                            continue

                        # 청크 유형에 따른 처리
                        if isinstance(chunk, dict):
                            # 텍스트 청크 처리
                            if 'new_text' in chunk:
                                text_chunk = chunk.get('new_text', '')
                                is_finished = chunk.get('finished', False)

                                # 문자 버퍼에 추가
                                char_buffer += text_chunk

                                # 문자 단위 처리
                                processed_text, char_buffer = post_processor.process_partial(char_buffer)

                                # 처리된 텍스트가 있으면 즉시 전송
                                if processed_text:
                                    json_data = json.dumps(
                                        {'text': processed_text, 'finished': False},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"
                                    last_send_time = current_time

                                # 시간 기반 강제 전송 확인
                                elapsed_since_send = current_time - last_send_time
                                if char_buffer and elapsed_since_send > (max_buffer_time / 1000):
                                    # 최대 지연 시간 초과 시 현재 버퍼 강제 전송
                                    json_data = json.dumps(
                                        {'text': char_buffer, 'finished': False},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"
                                    char_buffer = ""
                                    last_send_time = current_time

                                # 완료 신호 처리
                                if is_finished:
                                    # 남은 버퍼가 있으면 전송
                                    if char_buffer:
                                        json_data = json.dumps(
                                            {'text': char_buffer, 'finished': False},
                                            ensure_ascii=False
                                        )
                                        yield f"data: {json_data}\n\n"
                                        char_buffer = ""

                                    # 전체 텍스트 최종 처리
                                    full_response = await post_processor.finalize("")

                                    # 완료 신호 전송 (빈 텍스트, finished=true)
                                    json_data = json.dumps(
                                        {'text': "", 'finished': True},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"

                                    # 전체 완성된 응답 한 번 더 전송
                                    json_data = json.dumps(
                                        {'complete_response': full_response},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"
                                    break

                            # 완료 신호만 있는 경우
                            elif chunk.get('finished', False):
                                # 남은 버퍼 처리
                                if char_buffer:
                                    json_data = json.dumps(
                                        {'text': char_buffer, 'finished': False},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"
                                    char_buffer = ""

                                # 전체 텍스트 최종 처리
                                full_response = await post_processor.finalize("")

                                # 완료 신호 전송
                                json_data = json.dumps(
                                    {'text': "", 'finished': True},
                                    ensure_ascii=False
                                )
                                yield f"data: {json_data}\n\n"

                                # 전체 완성된 응답 전송
                                json_data = json.dumps(
                                    {'complete_response': full_response},
                                    ensure_ascii=False
                                )
                                yield f"data: {json_data}\n\n"
                                break

                        # 완료 마커 처리
                        elif chunk == '[DONE]':
                            if char_buffer:
                                json_data = json.dumps(
                                    {'text': char_buffer, 'finished': False},
                                    ensure_ascii=False
                                )
                                yield f"data: {json_data}\n\n"
                                char_buffer = ""

                            # 전체 텍스트 최종 처리
                            full_response = await post_processor.finalize("")

                            # 완료 신호 전송
                            json_data = json.dumps(
                                {'text': "", 'finished': True},
                                ensure_ascii=False
                            )
                            yield f"data: {json_data}\n\n"

                            # 전체 완성된 응답 전송
                            json_data = json.dumps(
                                {'complete_response': full_response},
                                ensure_ascii=False
                            )
                            yield f"data: {json_data}\n\n"
                            break

                    # LLM 처리 시간 기록
                    llm_elapsed = time.time() - start_llm_time
                    await self._log(
                        "info",
                        f"[{session_id}] LLM 처리 완료: {llm_elapsed:.4f}초 소요 "
                        f"[backend={settings.llm.llm_backend}]",
                        session_id=session_id
                    )
                    self._record_stage("llm_processing")

                    # 채팅 이력 저장
                    if settings.chat_history.enabled and full_response:
                        if background_tasks:
                            # BackgroundTasks가 제공된 경우 활용
                            background_tasks.add_task(self._save_chat_history, full_response)
                        else:
                            # 없으면 직접 태스크 생성
                            self._fire_and_forget(self._save_chat_history(full_response))

                except Exception as err:
                    error_occurred = True
                    await self._log("error", f"[{session_id}] 스트리밍 처리 중 오류: {str(err)}", exc_info=True)
                    # 오류 정보 전송
                    error_json = json.dumps(
                        {'error': True, 'text': str(err), 'finished': True},
                        ensure_ascii=False
                    )
                    yield f"data: {error_json}\n\n"

                finally:
                    # 총 처리 시간 계산
                    if not error_occurred:
                        total_time = sum(self.processing_stages.values())
                        await self._log(
                            "info",
                            f"[{session_id}] 채팅 처리 완료: {total_time:.4f}초 소요",
                            session_id=session_id,
                            stages=self.processing_stages
                        )

            # 스트리밍 응답 반환
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )

        except Exception as e:
            await self._log("error", f"[{session_id}] 스트리밍 초기화 중 오류: {str(e)}",
                            session_id=session_id, exc_info=True)

            # 오류 스트림 반환
            async def error_stream():
                error_data = {'error': True, 'text': f'처리 중 오류가 발생했습니다: {str(e)}', 'finished': True}
                json_str = json.dumps(error_data, ensure_ascii=False)
                yield f"data: {json_str}\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )

    def _fire_and_forget(self, coro):
        """코루틴을 안전하게 실행하고 에러를 로깅합니다."""

        async def wrapper():
            try:
                await coro
            except Exception as e:
                logger.error(f"백그라운드 태스크 오류: {e}", exc_info=True)

        task = asyncio.create_task(wrapper())
        if not hasattr(self.__class__, '_background_tasks'):
            self.__class__._background_tasks = []
        self.__class__._background_tasks.append(task)
        task.add_done_callback(
            lambda t: self.__class__._background_tasks.remove(t) if t in self.__class__._background_tasks else None)

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
            if settings.chat_history.enabled and hasattr(self, 'history_handler'):
                # Initialize retriever with documents
                await self.history_handler.init_retriever(documents)

                if settings.llm.llm_backend.lower() == "ollama":
                    # Process chat with history for Ollama
                    await self._log("debug", f"[{session_id}] Processing chat with history for Ollama backend")
                    rag_chat_chain = self.history_handler.init_chat_chain_with_history()
                    chat_history_response = await self.history_handler.handle_chat_with_history(
                        self.request, trans_lang, rag_chat_chain
                    ) or {"context": [], "answer": ""}

                    # Update documents with history context if available
                    context = chat_history_response.get("context", [])
                    if context:
                        self.request.chat.payload = self.document_processor.convert_document_to_payload(
                            context
                        ) + self.request.chat.payload

                    query_answer = chat_history_response.get("answer", "")

                elif settings.llm.llm_backend.lower() == "vllm":
                    # Process chat with history for vLLM
                    await self._log("debug", f"[{session_id}] Processing chat with history for vLLM backend")

                    # 기존 히스토리 핸들러 호출 - 이미 내부에서 자동으로 Gemma 모델을 감지합니다
                    result = await self.history_handler.handle_chat_with_history_vllm(
                        self.request, trans_lang
                    )

                    if result:
                        query_answer = result[0]
                        vllm_retrival_document = result[1]
                    else:
                        query_answer = ""
                        vllm_retrival_document = []

                    # 히스토리 처리 중 오류 발생 시 로깅
                    if not query_answer and not vllm_retrival_document:
                        await self._log("warning",
                                        f"[{session_id}] 히스토리 처리 결과가 비어 있습니다. 기본 LLM 쿼리로 대체합니다.",
                                        session_id=session_id)
                        # 히스토리 처리 실패 시 기본 LLM 쿼리로 대체
                        query_answer = await self.llm_service.ask(documents, lang)
            else:
                # Direct LLM query without chat history
                await self._log("debug", f"[{session_id}] Processing direct LLM query without chat history")
                query_answer = await self.llm_service.ask(documents, lang)

            # 응답이 비어있을 경우 대체 메시지 제공
            if not query_answer or query_answer.strip() == "":
                await self._log("warning",
                                f"[{session_id}] LLM 응답이 비어 있습니다. 대체 메시지를 제공합니다.",
                                session_id=session_id)
                query_answer = "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

            return query_answer, vllm_retrival_document

        except Exception as e:
            await self._log("error", f"[{session_id}] Error processing LLM query: {str(e)}", exc_info=True)
            # 오류 발생 시 대체 메시지 제공
            return "죄송합니다. 질문 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.", None

    async def _save_chat_history(self, answer):
        """
        Save chat history to Redis asynchronously.

        개선된 오류 처리와 재시도 메커니즘을 추가하여 안정성 향상
        """
        session_id = self.request.meta.session_id
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                await self._log("debug", f"[{session_id}] Saving chat history to Redis (attempt {retry_count + 1})")

                # 메시지 생성
                chat_data = create_chat_data(session_id, [
                    create_message("HumanMessage", self.request.chat.user),
                    create_message("AIMessage", answer)
                ])

                # Redis에 저장
                await RedisUtils.async_save_message_to_redis(
                    system_info=self.request.meta.rag_sys_info,
                    session_id=session_id,
                    message=chat_data
                )

                await self._log("debug", f"[{session_id}] Chat history saved successfully")
                return True

            except Exception as e:
                retry_count += 1
                await self._log(
                    "warning" if retry_count < max_retries else "error",
                    f"[{session_id}] Failed to save chat history (attempt {retry_count}): {str(e)}",
                    exc_info=True
                )

                # 마지막 시도가 아니면 잠시 대기 후 재시도
                if retry_count < max_retries:
                    await asyncio.sleep(0.5 * retry_count)  # 점진적 지연

        # 모든 재시도 실패 후에도 처리는 계속 진행
        return False

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

    def _check_cache_cleanup(self):
        """
        주기적으로 캐시를 정리하여 메모리 사용량 최적화
        """
        current_time = time.time()
        if current_time - ChatService._last_cache_cleanup > ChatService._cache_cleanup_interval:
            try:
                # 캐시 크기 체크 및 로깅
                prompt_cache_size = len(ChatService._prompt_cache)
                response_cache_size = len(ChatService._response_cache)

                logger.info(
                    f"Performing periodic cache cleanup. Before: prompt_cache={prompt_cache_size}, "
                    f"response_cache={response_cache_size}")

                # 오래된 캐시 항목 정리
                expired_prompt_keys = [k for k, v in ChatService._prompt_cache.items()
                                       if k.split(':')[1] != self.request.meta.session_id]
                expired_response_keys = [k for k, v in ChatService._response_cache.items()
                                         if k.split(':')[1] != self.request.meta.session_id]

                # 다른 세션의 캐시만 일부 정리 (현재 세션 유지)
                for key in expired_prompt_keys[:max(prompt_cache_size // 2, 10)]:
                    ChatService._prompt_cache.pop(key, None)

                for key in expired_response_keys[:max(response_cache_size // 2, 20)]:
                    ChatService._response_cache.pop(key, None)

                # 히스토리 핸들러의 메모리 관리 함수 호출
                if hasattr(self, 'history_handler'):
                    self.history_handler.cleanup_processed_sets()

                # 정리 후 크기 확인
                logger.info(
                    f"Cache cleanup completed. After: prompt_cache={len(ChatService._prompt_cache)}, "
                    f"response_cache={len(ChatService._response_cache)}")

                # 타임스탬프 업데이트
                ChatService._last_cache_cleanup = current_time

            except Exception as e:
                logger.error(f"Error during cache cleanup: {str(e)}")

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


class StreamResponsePostProcessor:
    """
    스트리밍 응답을 문자 단위로 처리하여 보다 빠른 사용자 경험 제공

    최적화 포인트:
    1. 문장 완성을 기다리지 않고 문자 단위로 처리
    2. 최소 표시 단위 설정으로 자연스러운 흐름 유지
    3. 특수 문자 처리로 텍스트 일관성 보장
    4. URL과 같은 특수 패턴은 여전히 완성 후 처리
    5. 마지막 청크 전송 후 전체 응답을 완성된 형태로 한 번 더 전송
    """

    def __init__(self, response_generator, voc_processor, search_engine, request, documents):
        """초기화"""
        self.response_generator = response_generator
        self.voc_processor = voc_processor
        self.search_engine = search_engine
        self.request = request
        self.documents = documents
        self.logger = logging.getLogger(__name__)

        # 전체 응답 저장
        self.full_text = ""
        self.processed_chunks = []

        # 처리 설정
        self.min_chars = 2  # 한글은 자모 조합 고려해 최소 2자 이상일 때 전송
        self.force_interval = 100  # 최대 100ms 이상 지연되지 않도록 함
        self.last_send_time = time.time()

        # URL 및 특수 패턴 감지용
        self.url_pattern = re.compile(r'https?://\S+')
        self.url_buffer = ""  # URL 완성까지 임시 저장
        self.in_url = False  # URL 처리 중 상태

    def process_partial(self, text: str) -> Tuple[str, str]:
        """
        문자 단위로 텍스트 처리 - 문장 완성을 기다리지 않음

        Args:
            text: 처리할 텍스트

        Returns:
            tuple: (처리된_텍스트, 남은_버퍼)
        """
        current_time = time.time()
        force_send = (current_time - self.last_send_time) > (self.force_interval / 1000)

        # 텍스트가 없으면 처리하지 않음
        if not text:
            return "", ""

        # URL 패턴 검사 - URL은 완성될 때까지 버퍼링
        if self.in_url:
            # URL 종료 조건 확인 (공백, 줄바꿈 등)
            end_idx = -1
            for i, char in enumerate(text):
                if char.isspace():
                    end_idx = i
                    break

            if end_idx >= 0:
                # URL 완성됨
                self.url_buffer += text[:end_idx]
                processed_url = self._quick_process_urls(self.url_buffer)

                # 처리 결과와 남은 텍스트 반환
                self.in_url = False
                self.full_text += self.url_buffer + text[end_idx:end_idx + 1]
                remaining = text[end_idx + 1:]
                self.url_buffer = ""

                self.last_send_time = current_time
                return processed_url + text[end_idx:end_idx + 1], remaining
            else:
                # URL 계속 축적
                self.url_buffer += text
                self.full_text += text
                return "", ""  # URL 완성될 때까지 출력 보류

        # URL 시작 감지
        url_match = self.url_pattern.search(text)
        if url_match:
            start_idx = url_match.start()
            if start_idx > 0:
                # URL 이전 텍스트 처리
                prefix = text[:start_idx]
                self.full_text += prefix

                # URL 부분 버퍼링 시작
                self.in_url = True
                self.url_buffer = text[start_idx:]

                self.last_send_time = current_time
                return prefix, ""
            else:
                # 텍스트가 URL로 시작함
                self.in_url = True
                self.url_buffer = text
                self.full_text += text
                return "", ""

        # 일반 텍스트 처리 (URL 아님)
        # 충분한 텍스트가 있거나 강제 전송 조건 충족 시 전송
        if len(text) >= self.min_chars or force_send:
            self.full_text += text
            self.last_send_time = current_time
            return text, ""

        # 최소 길이 미달 시 버퍼 유지
        self.full_text += text
        return "", ""

    def _quick_process_urls(self, text: str) -> str:
        """URL을 빠르게 링크로 변환"""
        return self.url_pattern.sub(lambda m: f'<a href="{m.group(0)}" target="_blank">{m.group(0)}</a>', text)

    async def finalize(self, remaining_text: str) -> str:
        """최종 처리 - 참조 및 VOC 처리 등 무거운 작업 수행"""
        session_id = self.request.meta.session_id
        self.logger.debug(f"[{session_id}] 응답 최종 처리 시작")

        # 남은 텍스트 및 URL 버퍼 처리
        final_text = remaining_text
        if self.url_buffer:
            final_text = self.url_buffer + final_text
            self.url_buffer = ""
            self.in_url = False

        if final_text:
            self.full_text += final_text

        # 처리할 내용 없으면 빈 문자열 반환
        if not final_text and not self.full_text:
            return ""

        try:
            # 언어 설정 가져오기
            _, _, reference_word = self.response_generator.get_translation_language_word(
                self.request.chat.lang
            )

            # 전체 텍스트에 대한 최종 처리 수행
            processed_text = self.full_text

            # 1. 참조 추가
            if settings.prompt.source_count:
                processed_text = await asyncio.to_thread(
                    self.response_generator.make_answer_reference,
                    processed_text,
                    self.request.meta.rag_sys_info,
                    reference_word,
                    self.documents,
                    self.request
                )

            # 2. VOC 처리
            if "komico_voc" in settings.voc.voc_type.split(',') and self.request.meta.rag_sys_info == "komico_voc":
                processed_text = await asyncio.to_thread(
                    self.voc_processor.make_komico_voc_groupware_docid_url,
                    processed_text
                )

            # 3. URL 처리
            final_text = await asyncio.to_thread(
                self.search_engine.replace_urls_with_links,
                processed_text
            )

            self.logger.debug(f"[{session_id}] 응답 최종 처리 완료")
            return final_text

        except Exception as e:
            self.logger.error(f"[{session_id}] 응답 최종 처리 중 오류: {str(e)}", exc_info=True)
            # 오류 시 원본 반환
            return self.full_text

    def get_full_text(self) -> str:
        """전체 응답 텍스트 반환"""
        # URL 버퍼에 남은 내용도 포함
        if self.url_buffer:
            return self.full_text + self.url_buffer
        return self.full_text
