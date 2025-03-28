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
