"""
RAG(Retrieval-Augmented Generation) 시스템이 통합된 엔터프라이즈 채팅 서비스
===================================================================

이 모듈은 문서 검색 및 처리 기능과 통합된 다양한 LLM 백엔드를 구현합니다.
동시 요청, 캐싱, 오류 관리 및 성능 모니터링을 위한 강력한 처리 기능을 제공합니다.

주요 구성 요소:
- LLM 상호 작용 (Ollama 및 vLLM 백엔드)
- 문서 검색 및 처리
- 쿼리 전처리 및 최적화
- 채팅 이력 관리
- 컨텍스트 강화된 응답 생성
- 동시성 제어가 포함된 비동기 처리
- 외부 서비스 보호를 위한 회로 차단기 패턴
- 종합적인 로깅 및 성능 추적

의존성:
- FastAPI: 웹 프레임워크
- LangChain: 체인 운영
- Redis: 채팅 이력 캐싱
- 특정 처리 작업을 위한 다양한 유틸리티 모듈
"""

import asyncio
import logging
import time
from asyncio import Semaphore, wait_for, TimeoutError
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, BackgroundTasks
from langchain_ollama.llms import OllamaLLM
from starlette.responses import StreamingResponse

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse
from src.services.utils.cache_service import CacheService
from src.services.utils.model_utils import ModelUtils

# httpx 및 httpcore 로그 레벨 조정
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# 설정 파일에서 애플리케이션 설정 로드
settings = ConfigLoader().get_settings()

# 모듈 수준 로거 설정
logger = logging.getLogger(__name__)

# 체인 레지스트리 접근을 위한 스레드 안전 잠금
_chain_lock = Lock()

# 리소스 관리를 위한 동시 작업 제한 세마포어
_semaphore = Semaphore(settings.cache.max_concurrent_tasks)

# 시작 중에 초기화될 전역 LLM 인스턴스
mai_chat_llm = None


async def limited_chain_invoke(chain, context, timeout=60):
    """
    동시성 및 타임아웃 제어가 포함된 체인 호출.

    세마포어를 사용하여 리소스 사용량을 관리하고
    작업이 중단되지 않도록 타임아웃을 적용합니다.

    Args:
        chain: 호출할 체인
        context (dict): 체인의 컨텍스트
        timeout (int): 타임아웃(초)

    Returns:
        Any: 체인의 결과

    Raises:
        TimeoutError: 작업이 타임아웃을 초과하는 경우
    """
    start_time = time.time()
    session_id = context.get("input", "")[:20] if isinstance(context, dict) and "input" in context else "unknown"

    async with _semaphore:
        try:
            # 비동기 호출 시도
            if hasattr(chain, "ainvoke") and callable(getattr(chain, "ainvoke")):
                logger.debug(f"[{session_id}] 체인 비동기 호출 (ainvoke)")
                # 비동기 호출에 타임아웃 적용
                result = await wait_for(chain.ainvoke(context), timeout=timeout)

                # 결과가 코루틴인지 추가 확인
                if asyncio.iscoroutine(result):
                    logger.debug(f"[{session_id}] 코루틴 응답 감지, 추가 await 처리")
                    result = await wait_for(result, timeout=timeout)

                return result

            # 동기 함수를 별도 스레드에서 실행
            elif hasattr(chain, "invoke") and callable(getattr(chain, "invoke")):
                logger.debug(f"[{session_id}] 체인 동기 호출 (invoke via thread)")
                # 동기 호출의 경우 to_thread 사용
                return await wait_for(asyncio.to_thread(chain.invoke, context), timeout=timeout)

            # 호출 가능한 객체인 경우
            elif callable(chain):
                logger.debug(f"[{session_id}] 호출 가능한 체인 객체 호출")
                # 체인이 호출 가능한 경우 직접 호출 시도
                result = chain(context)

                # 결과가 코루틴인지 확인
                if asyncio.iscoroutine(result):
                    logger.debug(f"[{session_id}] 코루틴 결과 감지, await 처리")
                    return await wait_for(result, timeout=timeout)
                return result

            else:
                # 적절한 호출 메서드를 찾을 수 없음
                msg = f"[{session_id}] 체인 호출 메서드를 찾을 수 없습니다"
                logger.error(msg)
                raise ValueError(msg)

        except TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] 체인 호출 타임아웃: {elapsed:.2f}초 (제한: {timeout}초)")
            raise
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] 체인 호출 실패: {elapsed:.2f}초 후: {type(e).__name__}: {str(e)}")
            raise


@asynccontextmanager
async def model_llm_init(app: FastAPI):
    """
    LLM 모델 초기화를 위한 비동기 컨텍스트 관리자.
    설정에 따라 Ollama 또는 vLLM을 초기화합니다.

    Args:
        app (FastAPI): FastAPI 앱 인스턴스.

    Yields:
        dict: 초기화된 모델 사전.
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

        # 설정 유효성 검사
        ModelUtils.validate_settings(settings)

        backend = settings.llm.llm_backend.lower()
        initialized_models = {}

        logger.info(f"LLM 초기화 시작: {backend} 백엔드")

        if backend == 'ollama':
            logger.info(f"Ollama LLM 초기화: {settings.ollama.access_type} 접근 방식")
            if settings.ollama.access_type.lower() == 'url':
                # URL 접근 방식으로 Ollama 초기화
                mai_chat_llm = OllamaLLM(
                    base_url=settings.ollama.ollama_url,
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )
                initialized_models[backend] = mai_chat_llm
            elif settings.ollama.access_type.lower() == 'local':
                # 로컬 접근 방식으로 Ollama 초기화
                mai_chat_llm = OllamaLLM(
                    model=settings.ollama.model_name,
                    mirostat=settings.ollama.mirostat,
                    temperature=settings.ollama.temperature if hasattr(settings.ollama, 'temperature') else 0.7,
                )
                initialized_models[backend] = None
            else:
                raise ValueError(f"지원되지 않는 ollama 접근 방식: {settings.ollama.access_type}")
        elif backend == 'vllm':
            # vLLM은 여기서 초기화가 필요하지 않음, 초기화됨으로 표시
            initialized_models[backend] = None
            logger.info(f"vLLM 백엔드 연결 준비: {settings.vllm.endpoint_url}")
        else:
            raise ValueError(f"지원되지 않는 LLM 백엔드: {backend}")

        logger.info(f"LLM 초기화 완료: {backend} 백엔드")

        # 상태 확인 등록 (지원되는 경우)
        if hasattr(app, 'add_healthcheck') and callable(app.add_healthcheck):
            app.add_healthcheck("llm", lambda: mai_chat_llm is not None if backend == 'ollama' else True)

        yield initialized_models

    except Exception as err:
        logger.error(f"LLM 초기화 실패: {err}", exc_info=True)
        raise RuntimeError(f"LLM 초기화 실패: {err}")
    finally:
        logger.info("LLM 컨텍스트 종료")


def get_llm_model():
    """
    초기화된 LLM 모델을 반환합니다.

    Returns:
        Any: 초기화된 LLM 모델 인스턴스 또는 None
    """
    return mai_chat_llm


async def process_chat_request(request: ChatRequest) -> ChatResponse:
    """
    채팅 요청을 처리하여 ChatService를 통해 응답을 생성합니다.

    Args:
        request (ChatRequest): 처리할 채팅 요청

    Returns:
        ChatResponse: 생성된 응답
    """
    # ChatService 인스턴스 생성 (초기화된 모델 전달)
    chat_service = ChatService(request)

    # 요청 처리 및 응답 반환
    return await chat_service.process_chat()


async def process_stream_request(request: ChatRequest, background_tasks: BackgroundTasks = None) -> StreamingResponse:
    """
    스트리밍 응답을 위한 채팅 요청을 처리합니다.

    Args:
        request (ChatRequest): 처리할 채팅 요청
        background_tasks (BackgroundTasks, optional): 백그라운드 작업

    Returns:
        StreamingResponse: 스트리밍 응답
    """
    # ChatService 인스턴스 생성 (초기화된 모델 전달)
    chat_service = ChatService(request)

    # 스트리밍 요청 처리 및 응답 반환
    return await chat_service.stream_chat(background_tasks)
