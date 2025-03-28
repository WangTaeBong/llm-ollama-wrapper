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
import hashlib
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
from typing import Any, Dict, Tuple

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
from src.services.chat.circuit import CircuitBreaker
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse, MetaRes, PayloadRes, ChatRes
from src.schema.vllm_inquery import VllmInquery
from src.services.messaging.formatters import MessageFormatter
from src.services.document_processor import DocumentProcessor
from src.services.history.llm_history_handler import LlmHistoryHandler
from src.services.query_processor import QueryProcessor
from src.services.response_generator.core import ResponseGenerator
from src.services.search_engine import SearchEngine
from src.services.voc import VOCLinkProcessor
from src.utils.redis_utils import RedisUtils
from src.services.chat.retriever import RetrieverService
from src.services.chat.processor.stream_processor import StreamResponsePostProcessor
from src.services.base_service import BaseService
from src.services.utils.cache_service import CacheService
from src.services.utils.model_utils import ModelUtils
from src.services.common.error_handler import ErrorHandler
from src.services.llm.service import LLMService

# httpx 및 httpcore 로그 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load application settings from configuration
settings = ConfigLoader().get_settings()

# Configure module-level logger
logger = logging.getLogger(__name__)

# Thread-safe lock for chain registry access
_chain_lock = Lock()

# Semaphore to limit concurrent tasks for resource management
_semaphore = Semaphore(settings.cache.max_concurrent_tasks)

# Global LLM instance to be initialized during startup
mai_chat_llm = None


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
    session_id = context.get("input", "")[:20] if isinstance(context, dict) and "input" in context else "unknown"

    async with _semaphore:
        try:
            # 비동기 호출 시도
            if hasattr(chain, "ainvoke") and callable(getattr(chain, "ainvoke")):
                logger.debug(f"[{session_id}] 체인 비동기 호출 (ainvoke)")
                # Apply timeout to async call
                result = await wait_for(chain.ainvoke(context), timeout=timeout)

                # 결과가 코루틴인지 추가 확인
                if asyncio.iscoroutine(result):
                    logger.debug(f"[{session_id}] 코루틴 응답 감지, 추가 await 처리")
                    result = await wait_for(result, timeout=timeout)

                return result

            # 동기 함수를 별도 스레드에서 실행
            elif hasattr(chain, "invoke") and callable(getattr(chain, "invoke")):
                logger.debug(f"[{session_id}] 체인 동기 호출 (invoke via thread)")
                # For sync calls, use to_thread
                return await wait_for(asyncio.to_thread(chain.invoke, context), timeout=timeout)

            # 호출 가능한 객체인 경우
            elif callable(chain):
                logger.debug(f"[{session_id}] 호출 가능한 체인 객체 호출")
                # Try direct call if chain is callable
                result = chain(context)

                # 결과가 코루틴인지 확인
                if asyncio.iscoroutine(result):
                    logger.debug(f"[{session_id}] 코루틴 결과 감지, await 처리")
                    return await wait_for(result, timeout=timeout)
                return result

            else:
                # 적절한 호출 메서드를 찾을 수 없음
                raise ValueError(f"[{session_id}] 체인 호출 메서드를 찾을 수 없습니다")

        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] Chain invocation timed out: {elapsed:.2f}s (limit: {timeout}s)")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] Chain invocation failed: {elapsed:.2f}s after: {type(e).__name__}: {str(e)}")
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

        # 캐시 서비스 초기화
        CacheService.initialize({
            "chain": {"maxsize": settings.cache.max_size, "ttl": settings.cache.chain_ttl},
            "prompt": {"maxsize": 100, "ttl": 3600},  # 1시간
            "response": {"maxsize": 200, "ttl": 900},  # 15분
        })
        logger.info("캐시 서비스가 초기화 되었습니다.")

        # Validate configuration settings
        ModelUtils.validate_settings(settings)

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


class ChatService(BaseService):
    """
    채팅 상호작용을 처리하는 메인 서비스

    이 클래스는 문서 검색, 쿼리 최적화, LLM 상호작용, 응답 생성을 포함한
    전체 채팅 처리 워크플로우를 조정합니다. 효율적이고 신뢰성 있는 채팅 처리를
    위한 종합적인 성능 모니터링, 캐싱 전략, 비동기 로깅을 구현합니다.

    클래스 속성:
        _prompt_cache: 인사말 및 FAQ 응답 저장을 위한 TTL 캐시
        _response_cache: 전체 채팅 응답 저장을 위한 TTL 캐시
        _circuit_breaker: 외부 서비스 보호를 위한 회로 차단기
    """
    _circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=300)

    # 캐시 정리 주기
    _cache_cleanup_interval = 3600  # 1시간마다 캐시 정리
    _last_cache_cleanup = time.time()

    def __init__(self, request: ChatRequest):
        """
        Initialize the chat service with the given request.

        Args:
            request (ChatRequest): The chat request containing user query and metadata.
        """
        super().__init__(request.meta.session_id, settings)
        self.error_handler = ErrorHandler()

        self.request = request
        self.retriever_service = RetrieverService(request, settings)
        self.llm_service = LLMService(request)

        # Initialize history handlers if chat history is enabled in settings
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

        # Create cache key based on session and query
        self.cache_key = f"{self.request.meta.rag_sys_info}:{self.request.meta.session_id}:{self.request.chat.user}"

        # 캐시 정리 확인
        self._check_cache_cleanup()

        # 인스턴스 생성
        self.formatter = MessageFormatter()

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
        await self.log("info", f"[{session_id}] Starting chat request processing", session_id=session_id)

        self.start_time = time.time()
        self.processing_stages = {}

        # 동일한 요청에 대한 응답 캐시 확인
        cached_response = CacheService.get("response", self.cache_key)
        if cached_response:
            await self.log("info", f"[{session_id}] 캐시된 응답 변환", session_id=session_id,
                           cache_key=self.cache_key)
            return cached_response

        try:
            # Handle greetings or filter invalid queries
            response = await self._handle_greeting_or_filter()
            if response:
                elapsed = self.record_stage("greeting_filter")
                await self.log("info", f"[{session_id}] Greeting/filter stage complete: {elapsed:.4f}s elapsed",
                               session_id=session_id)
                return response

            # Validate retrieval documents if filtering is enabled
            if settings.retriever.filter_flag:
                filter_res = self.document_processor.validate_retrieval_documents(self.request)
                if filter_res:
                    elapsed = self.record_stage("document_validation")
                    await self.log("info", f"[{session_id}] Document validation complete: {elapsed:.4f}s elapsed",
                                   session_id=session_id)
                    return filter_res

            # Optimize FAQ queries if applicable
            faq_query = self._handle_faq_query()
            if faq_query:
                self.request.chat.user = faq_query
                elapsed = self.record_stage("faq_optimization")
                await self.log("info", f"[{session_id}] FAQ query optimization: {elapsed:.4f}s elapsed",
                               session_id=session_id)

            # Start parallel tasks
            # Process document retrieval and language settings in parallel
            # retrieval_task = asyncio.create_task(self._retrieve_documents())
            retrieval_task = asyncio.create_task(self._process_documents())
            language_task = asyncio.create_task(self._get_language_settings())

            # Wait for tasks to complete
            await asyncio.gather(retrieval_task, language_task)

            # Collect results
            documents = retrieval_task.result()
            lang, trans_lang, reference_word = language_task.result()

            elapsed = self.record_stage("parallel_processing")
            await self.log(
                "info",
                f"[{session_id}] Parallel processing complete(prepare): {elapsed:.4f}s elapsed, "
                f"{len(documents)} documents retrieved",
                session_id=session_id
            )

            # 히스토리 설정 확인 로깅
            await self.log("debug",
                           f"[{session_id}] 히스토리 설정: chat_history.enabled={settings.chat_history.enabled}, "
                           f"use_improved_history={getattr(settings.llm, 'use_improved_history', False)}")

            # Process LLM query with chat history or direct query
            start_llm_time = time.time()
            query_answer, vllm_retrival_document = await self._process_llm_query(documents, lang, trans_lang)

            llm_elapsed = time.time() - start_llm_time
            await self.log(
                "info",
                f"[{session_id}] LLM processing complete: {llm_elapsed:.4f}s "
                f"elapsed [backend={settings.llm.llm_backend}]",
                session_id=session_id
            )
            self.record_stage("llm_processing")

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
            await self.log("debug", f"[{session_id}] Answer finalization complete: {finalize_elapsed:.4f}s elapsed",
                           session_id=session_id)
            self.record_stage("answer_finalization")

            # Create and return final response
            final_response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), final_query_answer)

            total_time = sum(self.processing_stages.values())
            await self.log(
                "info",
                f"[{session_id}] Chat processing complete: {total_time:.4f}s elapsed",
                session_id=session_id,
                stages=self.processing_stages
            )

            # 상당한 시간이 소요되는 복잡한 쿼리에 대한 응답 캐싱
            if total_time > 5.0:  # 5초 이상 소요되는 쿼리만 캐싱
                CacheService.set("response", self.cache_key, final_response)

            return final_response

        except Exception as err:
            return self.error_handler.handle_error(
                error=err,
                session_id=self.request.meta.session_id,
                request=self.request
            )

    async def stream_chat(self, background_tasks: BackgroundTasks = None) -> StreamingResponse:
        """
        문자 단위 처리를 통한 실시간 스트리밍 응답 제공, 히스토리 지원 기능 추가
        개선된 2단계 접근법으로 대화 이력을 처리

        Returns:
            StreamingResponse: 스트리밍 응답 객체
        """
        session_id = self.request.meta.session_id
        await self.log("info", f"[{session_id}] 문자 단위 스트리밍 채팅 요청 시작", session_id=session_id)

        # 처리 시간 측정 시작
        self.start_time = time.time()
        self.processing_stages = {}

        try:
            # 인사말 또는 간단한 응답 처리
            greeting_response = await self._handle_greeting_or_filter()
            if greeting_response:
                elapsed = self.record_stage("greeting_filter")
                await self.log("info", f"[{session_id}] 인사말/필터 단계 완료: {elapsed:.4f}초 소요", session_id=session_id)

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
            elapsed = self.record_stage("language_processing")
            await self.log("info", f"[{session_id}] 언어 처리 완료: {elapsed:.4f}초 소요", session_id=session_id)

            # vLLM 백엔드 확인
            if settings.llm.llm_backend.lower() != "vllm":
                await self.log("error", f"[{session_id}] 스트리밍은 vLLM 백엔드에서만 지원됩니다.")

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
                await self.log("info", f"[{session_id}] 히스토리 기능이 활성화된 상태로 스트리밍 처리 진행", session_id=session_id)

                # 개선된 히스토리 처리 사용 여부 확인
                use_improved_history = getattr(settings.llm, 'use_improved_history', False)

                # 히스토리 핸들러에 검색기 초기화
                await self.history_handler.init_retriever(empty_documents)

                # 모델 유형에 따른 스트리밍 핸들러 디스패치
                if BaseService.is_gemma_model(settings):
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
                    await self.log("debug", f"[{session_id}] 검색된 문서로 업데이트: {len(retrieval_document)}개",
                                   session_id=session_id)
                    # 포스트 프로세서의 문서도 업데이트
                    post_processor.documents = retrieval_document
            else:
                # 히스토리 없는 일반 스트리밍 처리
                # 컨텍스트 준비
                await self.log("info", f"[{session_id}] 히스토리 없는 일반 스트리밍 처리 시작", session_id=session_id)

                # 문서 검색
                retrieval_task = asyncio.create_task(self._retrieve_documents())
                await retrieval_task
                documents = retrieval_task.result()

                elapsed = self.record_stage("document_retrieval")
                await self.log("info", f"[{session_id}] 문서 검색 완료: {elapsed:.4f}초 소요, {len(documents)}개 문서 검색됨",
                               session_id=session_id)

                # 포스트 프로세서 업데이트
                post_processor.documents = documents

                context = {
                    "input": self.request.chat.user,
                    "context": documents,
                    "history": "",  # 빈 값으로 전달
                    "language": lang,
                    "today": self.response_generator.get_today(),
                }

                # VOC 관련 설정 추가
                if self.request.meta.rag_sys_info == "komico_voc":
                    context.update({
                        "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                        "check_gw_word_link": settings.voc.check_gw_word_link,
                        "check_gw_word": settings.voc.check_gw_word,
                        "check_block_line": settings.voc.check_block_line,
                    })

                # 시스템 프롬프트 생성
                # vllm_inquery_context = self.llm_service.build_system_prompt(context)
                vllm_inquery_context = self.llm_service.build_system_prompt_gemma(context)

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

                                    # 문서 존재 여부 확인
                                    documents_exist = False
                                    if hasattr(post_processor, "documents") and post_processor.documents:
                                        # 'wiki_ko'가 아닌 문서만 필터링하여 카운트
                                        valid_documents = [
                                            doc for doc in post_processor.documents
                                            if not (hasattr(doc, 'metadata') and
                                                    doc.metadata.get('source', doc.metadata.get('doc_name', '')).find('wiki_ko') >= 0)
                                        ]

                                        await self.log("debug", f"[{session_id}] 문서 필터링: 전체 {len(post_processor.documents)}개 중 유효한 문서 {len(valid_documents)}개")
                                        documents_exist = len(valid_documents) > 0

                                    # 결과 코드 설정
                                    result_code = ErrorCd.get_code(ErrorCd.SUCCESS)
                                    if not documents_exist:
                                        result_code = ErrorCd.get_code(ErrorCd.SUCCESS_NO_EXIST_KNOWLEDGE_DATA)

                                    # 완료 신호 전송 (빈 텍스트, finished=true)
                                    json_data = json.dumps(
                                        {'text': "", 'finished': True},
                                        ensure_ascii=False
                                    )
                                    yield f"data: {json_data}\n\n"

                                    # 전체 완성된 응답 한 번 더 전송
                                    complete_data = {
                                        'complete_response': full_response,
                                        'result_cd': result_code,
                                        'result_desc': ErrorCd.get_description(
                                            ErrorCd.SUCCESS_NO_EXIST_KNOWLEDGE_DATA if not documents_exist else ErrorCd.SUCCESS
                                        )
                                    }

                                    json_data = json.dumps(complete_data, ensure_ascii=False)
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
                    await self.log(
                        "info",
                        f"[{session_id}] LLM 처리 완료: {llm_elapsed:.4f}초 소요 "
                        f"[backend={settings.llm.llm_backend}]",
                        session_id=session_id
                    )
                    self.record_stage("llm_processing")

                    # 채팅 이력 저장
                    if settings.chat_history.enabled and full_response:
                        if background_tasks:
                            # BackgroundTasks가 제공된 경우 활용
                            background_tasks.add_task(self._save_chat_history, full_response)
                        else:
                            # 없으면 직접 태스크 생성
                            self.fire_and_forget(self._save_chat_history(full_response))

                except Exception as err:
                    error_occurred = True
                    await self.log("error", f"[{session_id}] 스트리밍 처리 중 오류: {str(err)}", exc_info=True)
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
                        await self.log(
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
            self.error_handler.handle_error(
                error=e,
                session_id=self.request.meta.session_id,
                request=self.request
            )

            # 오류 스트림 반환
            async def error_stream():
                error_data = {'error': True, 'text': f'처리 중 오류가 발생했습니다: {str(e)}', 'finished': True}
                json_str = json.dumps(error_data, ensure_ascii=False)
                yield f"data: {json_str}\n\n"

            return StreamingResponse(
                error_stream(),
                media_type="text/event-stream"
            )

    async def _retrieve_documents(self):
        """
        Retrieve documents asynchronously.

        This method runs the document retrieval in a separate thread pool
        to avoid blocking the main event loop, as document retrieval can be
        a time-consuming operation.

        Returns:
            list: Retrieved documents
        """
        return await self.retriever_service.retrieve_documents_async()

    async def _process_documents(self):
        """
        문서를 검색하고 처리하는 향상된 메서드.
        기본 문서 검색 및 웹 검색 결과를 통합합니다.
        """
        session_id = self.request.meta.session_id
        await self.log("debug", f"[{session_id}] 문서 처리 시작 (웹 검색 포함)")

        start_time = time.time()

        try:
            # 기본 문서 검색 - 비동기 메서드 사용
            documents = await self.retriever_service.retrieve_documents_async()

            # 설정에 따라 웹 검색 결과 통합
            if getattr(settings, 'web_search', {}).get('enabled', False):
                await self.log("debug", f"[{session_id}] 웹 검색 시작")

                # 개선된 웹 검색 메서드 호출
                documents = await self.retriever_service.add_web_search_results()

                await self.log("debug",
                               f"[{session_id}] 웹 검색 완료: 총 {len(documents)}개 문서")

            # 문서 품질 향상을 위한 필터링 (최소 10자 이상)
            filtered_docs = self.retriever_service.filter_documents(min_content_length=10)

            # 쿼리 관련성 기준으로 문서 정렬
            sorted_docs = self.retriever_service.sort_documents_by_relevance()

            elapsed = time.time() - start_time
            self.record_stage("document_processing")

            await self.log("info",
                           f"[{session_id}] 문서 처리 완료: {elapsed:.4f}초 소요, "
                           f"{len(sorted_docs)}개 문서 준비됨",
                           session_id=session_id)

            return sorted_docs

        except Exception as e:
            elapsed = time.time() - start_time
            await self.log("error",
                           f"[{session_id}] 문서 처리 중 오류: {str(e)} - {elapsed:.4f}초 소요",
                           session_id=session_id,
                           exc_info=True)

            # 오류 발생 시 빈 문서 목록 반환
            return []

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
                    await self.log("info",
                                   f"[{session_id}] Ollama 히스토리 처리 시작. "
                                   f"use_improved_history={getattr(settings.llm, 'use_improved_history', False)}")

                    rag_chat_chain = self.history_handler.init_chat_chain_with_history()

                    # 더 명확한 진단 정보
                    if rag_chat_chain is None:
                        await self.log("error", f"[{session_id}] 채팅 체인 초기화 실패")
                        return "채팅 체인 초기화에 실패했습니다. 다시 시도해 주세요.", None

                    # 개선된 히스토리 사용 여부 로깅
                    use_improved_history = getattr(settings.llm, 'use_improved_history', False)
                    await self.log("info", f"[{session_id}] 개선된 히스토리 기능 상태: {use_improved_history}")

                    chat_history_response = await self.history_handler.handle_chat_with_history(
                        self.request, trans_lang, rag_chat_chain
                    ) or {"context": [], "answer": ""}

                    # 히스토리 처리 결과 로깅
                    await self.log("debug",
                                   f"[{session_id}] Ollama 히스토리 처리 결과: "
                                   f"{len(chat_history_response.get('context', []))}개 문서, "
                                   f"응답 길이: {len(chat_history_response.get('answer', ''))}")

                    # 문서 컨텍스트 업데이트
                    context_docs = chat_history_response.get("context", [])
                    query_answer = chat_history_response.get("answer", "")
                    vllm_retrival_document = context_docs  # Ollama에서도 문서 정보 활용

                    # 문서가 있으면 payload 업데이트
                    if context_docs:
                        self.request.chat.payload = self.document_processor.convert_document_to_payload(
                            context_docs
                        ) + self.request.chat.payload

                    # Update documents with history context if available
                    """
                    context = chat_history_response.get("context", [])
                    if context:
                        self.request.chat.payload = self.document_processor.convert_document_to_payload(
                            context
                        ) + self.request.chat.payload
                    query_answer = chat_history_response.get("answer", "")
                    """

                elif settings.llm.llm_backend.lower() == "vllm":
                    # Process chat with history for vLLM
                    await self.log("debug", f"[{session_id}] Processing chat with history for vLLM backend")

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
                        await self.log("warning",
                                       f"[{session_id}] 히스토리 처리 결과가 비어 있습니다. 기본 LLM 쿼리로 대체합니다.",
                                       session_id=session_id)
                        # 히스토리 처리 실패 시 기본 LLM 쿼리로 대체
                        query_answer = await self.llm_service.ask(documents, lang)
            else:
                # Direct LLM query without chat history
                await self.log("debug", f"[{session_id}] Processing direct LLM query without chat history")
                query_answer = await self.llm_service.ask(documents, lang)

            # 응답이 비어있을 경우 대체 메시지 제공
            if not query_answer or query_answer.strip() == "":
                await self.log("warning",
                               f"[{session_id}] LLM 응답이 비어 있습니다. 대체 메시지를 제공합니다.",
                               session_id=session_id)
                query_answer = "죄송합니다. 현재 응답을 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

            return query_answer, vllm_retrival_document

        except Exception as e:
            await self.log("error", f"[{session_id}] Error processing LLM query: {str(e)}", exc_info=True)
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
                await self.log("debug", f"[{session_id}] Saving chat history to Redis (attempt {retry_count + 1})")

                # 메시지 생성
                chat_data = self.formatter.create_chat_data(session_id, [
                    self.formatter.create_message("HumanMessage", self.request.chat.user),
                    self.formatter.create_message("AIMessage", answer)
                ])

                # Redis에 저장
                await RedisUtils.async_save_message_to_redis(
                    system_info=self.request.meta.rag_sys_info,
                    session_id=session_id,
                    message=chat_data
                )

                await self.log("debug", f"[{session_id}] Chat history saved successfully")
                return True

            except Exception as e:
                retry_count += 1
                await self.log(
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
        cached_response = CacheService.get("prompt", greeting_cache_key)
        if cached_response:
            await self.log("debug", f"[{session_id}] 캐시된 인사말 응답 사용")
            return cached_response

        # Check for predefined greeting or response patterns
        check_res = self.query_processor.check_query_sentence(self.request)
        if check_res:
            await self.log("debug", f"[{session_id}] Greeting detected")
            response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), check_res)
            # 인사말 응답 캐싱
            CacheService.set("prompt", greeting_cache_key, response)
            return response

        # Apply query filtering if enabled
        if settings.query_filter.enabled and not self.query_processor.filter_query(self.request.chat.user):
            await self.log("debug", f"[{session_id}] Query filtered")
            farewell_msg = random.choice(
                self.query_check_dict.get_dict_data(self.request.chat.lang, "farewells_msg")
            )
            response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), farewell_msg)
            # 인사말 응답 캐싱
            CacheService.set("prompt", greeting_cache_key, response)
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
            cached_query = CacheService.get("prompt", faq_cache_key)
            if cached_query:
                logger.debug(f"[{session_id}] 캐시된 LLM 쿼리 최적화 사용")
                return cached_query

            faq_query = self.query_processor.construct_faq_query(self.request)
            if faq_query:
                logger.debug(f"[{session_id}] FAQ query optimized: {self.request.chat.user} -> {faq_query}")
                # Cache successful LLM query optimization
                CacheService.set("prompt", faq_cache_key, faq_query)

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

        # 문서 참조 패턴 제거
        doc_reference_pattern = r"\(\[Document\(metadata=\{.*?\}\s*,\s*…\)\]\)"
        query_answer = re.sub(doc_reference_pattern, "", query_answer)

        try:
            # Add source references
            if settings.prompt.source_count:
                start_time = time.time()
                # 문서가 제공되었는지 확인하고 사용
                if retrieval_document is not None and len(retrieval_document) > 0:
                    logger.debug(f"[{session_id}] 제공된 문서 사용: {len(retrieval_document)}개")
                    query_answer = await asyncio.to_thread(
                        self.response_generator.make_answer_reference,
                        query_answer,
                        self.request.meta.rag_sys_info,
                        reference_word,
                        retrieval_document,
                        self.request
                    )
                # 제공된 문서가 없으면 기본 문서 사용
                elif hasattr(self, 'retriever_service') and hasattr(self.retriever_service, 'documents'):
                    logger.debug(f"[{session_id}] retriever_service 문서 사용: {len(self.retriever_service.documents)}개")
                    query_answer = await asyncio.to_thread(
                        self.response_generator.make_answer_reference,
                        query_answer,
                        self.request.meta.rag_sys_info,
                        reference_word,
                        self.retriever_service.documents,
                        self.request
                    )
                else:
                    logger.debug(f"[{session_id}] 출처 추가에 사용할 문서가 없습니다.")

                await self.log(
                    "debug",
                    f"[{session_id}] References added: {time.time() - start_time:.4f}s elapsed",
                    session_id=session_id
                )

            # Process VOC
            if "komico_voc" in settings.voc.voc_type.split(',') and self.request.meta.rag_sys_info == "komico_voc":
                start_time = time.time()
                result = await asyncio.to_thread(
                    self.voc_processor.process_voc_document_links,
                    query_answer
                )
                await self.log(
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
            await self.log(
                "debug",
                f"[{session_id}] URL link processing complete: {time.time() - start_time:.4f}s elapsed",
                session_id=session_id
            )
            return result

        except Exception as e:
            await self.log("error", f"[{session_id}] Error during answer finalization: {str(e)}",
                           session_id=session_id, exc_info=True)
            # Return original answer if finalization fails
            return query_answer

    def _create_response(self, error_code, system_msg, retrieval_documents=None):
        """
        채팅 응답 객체를 생성합니다.

        vllm_retrival_document 기준으로 지식 데이터 존재 여부를 판단하여
        적절한 에러 코드를 설정합니다.

        Args:
            error_code (dict): 에러 코드 정보
            system_msg (str): 시스템 응답 메시지
            retrieval_documents (list, optional): 검색된 문서 목록

        Returns:
            ChatResponse: 완성된 응답 객체
        """
        session_id = self.request.meta.session_id

        try:
            # 검색된 문서 존재 여부 확인
            has_documents = False

            # retrieval_documents 매개변수가 제공된 경우 우선 사용
            if retrieval_documents is not None:
                has_documents = len(retrieval_documents) > 0

            # 문서가 없는 경우 에러 코드 수정
            if not has_documents and error_code.get("code") == ErrorCd.get_code(ErrorCd.SUCCESS):
                # 문서가 없지만 성공 코드가 전달된 경우, 적절한 코드로 변경
                error_code = ErrorCd.get_error(ErrorCd.SUCCESS_NO_EXIST_KNOWLEDGE_DATA)
                logger.debug(f"[{session_id}] 문서가 없어 에러 코드를 SUCCESS_NO_EXIST_KNOWLEDGE_DATA로 변경")

            # 페이로드 준비
            payloads = []

            # retrieval_documents 가 있는 경우 이를 페이로드로 변환
            if has_documents:
                payload_objects = self.document_processor.convert_document_to_payload(retrieval_documents)
                payloads = [
                    PayloadRes(doc_name=doc.doc_name, doc_page=doc.doc_page, content=doc.content)
                    for doc in payload_objects
                ]

            # 응답 객체 생성 및 반환
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
                    info=[],  # 일단은 페이로드를 빈 값으로 전달
                )
            )

            # 성능 데이터 추가 (지원되는 경우)
            if hasattr(response, 'add_performance_data'):
                total_time = sum(self.processing_stages.values()) if self.processing_stages else 0
                retriever_metrics = self.retriever_service.get_performance_metrics()

                response.add_performance_data({
                    "total_processing_time": total_time,
                    "processing_stages": self.processing_stages,
                    "retriever_metrics": retriever_metrics,
                    "llm_metrics": self.llm_service.get_metrics() if hasattr(self.llm_service, 'get_metrics') else {}
                })

                logger.debug(
                    f"[{session_id}] 응답에 성능 데이터 추가: {total_time:.4f}초 총 처리 시간")

            return response

        except Exception as e:
            logger.error(f"[{session_id}] 응답 객체 생성 중 오류: {str(e)}", exc_info=True)
            # 오류 발생 시 대체 응답 생성
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
                    system="응답 생성 중 오류가 발생했습니다.",
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
        # CacheService는 자체적으로 자동 정리 기능을 제공하므로 기존 구현 제거
        pass

    @classmethod
    def clear_caches(cls):
        """
        서비스의 모든 캐시를 지웁니다.

        이 클래스 메서드는 프롬프트와 응답 캐시를 모두 비웁니다.
        테스트 중이나 업데이트 배포 시 유용할 수 있습니다.
        """
        CacheService.clear("prompt")
        CacheService.clear("response")
        logger.info("모든 ChatService 캐시가 지워졌습니다")
