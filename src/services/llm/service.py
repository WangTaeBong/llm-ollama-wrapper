"""
LLM 서비스 모듈

다양한 LLM 백엔드(Ollama, vLLM)와의 상호작용을 관리하는 서비스를 제공합니다.
프롬프트 생성, 모델 호출, 스트리밍 응답 처리 등 LLM 관련 핵심 기능을 담당합니다.
"""

import asyncio
import logging
import re
import time
from typing import Dict, Any, List, Optional, AsyncGenerator

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.common.config_loader import ConfigLoader
from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.base_service import BaseService
from src.services.common.error_handler import ErrorHandler
from src.services.utils.model_utils import ModelUtils
from src.services.response_generator import ResponseGenerator
from src.common.query_check_dict import QueryCheckDict
from src.services.chat.circuit import CircuitBreaker

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)

_llm_circuit_breaker = CircuitBreaker.create_from_config(name="llm_service")


# 비동기 재시도 데코레이터
def async_retry(max_retries=3, backoff_factor=1.5, circuit_breaker=None):
    """
    비동기 함수에 재시도 로직과 회로 차단기 보호를 추가하는 데코레이터

    Args:
        max_retries (int): 최대 재시도 횟수
        backoff_factor (float): 재시도 간 대기 시간 증가 계수
        circuit_breaker (CircuitBreaker): 회로 차단기 인스턴스(선택적)

    Returns:
        callable: 재시도 로직이 포함된 데코레이터된 함수
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retry_count = 0
            last_exception = None

            while retry_count < max_retries:
                # 회로 차단기 확인
                if circuit_breaker and circuit_breaker.is_open():
                    logger.warning(f"회로 열림, {func.__name__} 호출 건너뜀")
                    raise RuntimeError(f"서비스 사용 불가: {func.__name__}에 대한 회로가 열려 있음")

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    # 로그 실행 시간 모니터링
                    logger.debug(f"함수 {func.__name__} 완료: {execution_time:.4f}초")

                    # 회로 차단기에 성공 기록
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result
                except (asyncio.TimeoutError, ConnectionError) as e:
                    # 특정 오류에 대해 회로 차단기에 실패 기록
                    if circuit_breaker:
                        circuit_breaker.record_failure(e)  # 예외 객체 전달

                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    last_exception = e

                    logger.warning(
                        f"{func.__name__}에 대한 재시도 {retry_count}/{max_retries} "
                        f"{wait_time:.2f}초 후 - 원인: {type(e).__name__}: {str(e)}"
                    )

                    # 재시도 전 대기
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logger.warning(f"async_retry 예외: {e}")
                    # 다른 예외에 대해서는 실패 기록하되 재시도 안 함
                    if circuit_breaker:
                        circuit_breaker.record_failure(e)  # 예외 객체 전달
                    raise

            # 모든 재시도 실패
            logger.error(f"{func.__name__}에 대한 모든 {max_retries}회 재시도 실패")
            raise last_exception or RuntimeError(f"{func.__name__}에 대한 모든 재시도 실패")

        return wrapper

    return decorator


class LLMService(BaseService):
    """
    LLM 모델 상호작용 서비스

    Ollama 및 vLLM 백엔드를 지원하며 프롬프트 생성, 체인 초기화,
    쿼리 실행, 스트리밍 응답 처리 등의 기능을 제공합니다.
    """

    def __init__(self, request: ChatRequest):
        """
        LLM 서비스 초기화

        Args:
            request (ChatRequest): 채팅 요청 인스턴스
        """
        super().__init__(request.meta.session_id, settings)
        self.error_handler = ErrorHandler()

        self.request = request
        self.settings_key = f"{request.meta.rag_sys_info}-{request.meta.session_id}"

        # 프롬프트 사전 로드
        query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(settings, query_check_dict)

        # 백엔드에 따른 컴포넌트 초기화
        if settings.llm.llm_backend.lower() == "ollama" and not settings.chat_history.enabled:
            self.chain = self._initialize_chain()
        elif settings.llm.llm_backend.lower() == "vllm" and not settings.chat_history.enabled:
            self.system_prompt_template = self._initialize_system_prompt_vllm()

        # 기본 타임아웃 설정
        self.timeout = getattr(settings.llm, 'timeout', 60)

    def _initialize_chain(self):
        """
        체인 인스턴스를 초기화하거나 가져옵니다.

        Returns:
            chain: 초기화된 체인 인스턴스

        Raises:
            Exception: 체인 초기화 실패 시
        """
        start_time = time.time()
        session_id = self.request.meta.session_id
        try:
            # 전역 LLM 모델 참조
            from src.services.llm_ollama_process import mai_chat_llm

            if mai_chat_llm is None:
                raise ValueError("LLM 모델이 초기화되지 않았습니다. 설정을 확인하세요.")

            prompt_template = self.response_generator.get_rag_qa_prompt(self.request.meta.rag_sys_info)
            chat_prompt = ChatPromptTemplate.from_template(prompt_template)

            chain = ModelUtils.get_or_create_chain(self.settings_key, mai_chat_llm, chat_prompt)

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
        vLLM용 시스템 프롬프트 템플릿을 초기화합니다.

        Returns:
            str: 시스템 프롬프트 템플릿

        Raises:
            Exception: 시스템 프롬프트 초기화 실패 시
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
        동적 변수 및 채팅 이력이 포함된 시스템 프롬프트를 구성합니다.

        Args:
            context (dict): {input}, {context}, {language}, {today}와 같은 키를 포함하는 컨텍스트

        Returns:
            str: 포맷된 시스템 프롬프트
        """
        session_id = self.request.meta.session_id

        try:
            prompt = self.system_prompt_template.format(**context)
            return prompt
        except KeyError as e:
            # 누락된 키에 빈 문자열 설정 및 경고로 로깅
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

    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        재시도 및 회로 차단기가 적용된 vLLM 엔드포인트 호출 메서드

        Args:
            data (VllmInquery): vLLM 요청 데이터

        Returns:
            Dict: vLLM 응답
        """
        return await self.handle_request_with_retry(
            self._call_vllm_endpoint_impl,
            max_retries=2,
            backoff_factor=2,
            circuit_breaker=_llm_circuit_breaker,
            data=data
        )

    @async_retry(max_retries=2, backoff_factor=2, circuit_breaker=_llm_circuit_breaker)
    async def _call_vllm_endpoint_impl(self, data: VllmInquery):
        """
        재시도 및 회로 차단기가 적용된 vLLM 엔드포인트 내부 구현

        Args:
            data (VllmInquery): vLLM 요청 데이터

        Returns:
            Dict, AsyncGenerator: 스트리밍 모드에 따라 전체 응답 또는 청크 생성기 반환

        Raises:
            Exception: 여러 번의 재시도 후에도 vLLM 엔드포인트 호출 실패 시
        """
        from src.common.restclient import rc

        start_time = time.time()
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] Calling vLLM endpoint (stream={data.stream})")

        # circuit_breaker 확인
        if _llm_circuit_breaker.is_open():
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
            _llm_circuit_breaker.record_success()

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM endpoint error: {elapsed:.4f}s after: {str(e)}")

            self.metrics["error_count"] += 1

            # circuit_breaker 업데이트
            _llm_circuit_breaker.record_failure()

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
                processed_chunk = ModelUtils.process_vllm_chunk(chunk)

                # 청크 로깅 (텍스트가 있는 경우 길이만 기록)
                log_chunk = processed_chunk.copy()
                if 'new_text' in log_chunk:
                    log_chunk['new_text'] = f"<{len(log_chunk['new_text'])}자 길이의 텍스트>"
                logger.debug(f"[{session_id}] 청크 처리: {log_chunk}")

                yield processed_chunk

                # 마지막 청크 처리
                if processed_chunk.get('finished', False) or processed_chunk.get('error', False):
                    # 회로 차단기 성공 기록
                    _llm_circuit_breaker.record_success()

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
            _llm_circuit_breaker.record_failure()

            # 오류 청크 반환
            yield {
                "error": True,
                "message": f"스트리밍 오류: {str(e)}",
                "finished": True
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
                use_gemma_format = BaseService.is_gemma_model(settings)

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
        LLM에 쿼리를 수행합니다.

        Args:
            documents (list): 컨텍스트용 문서 목록
            language (str): 응답 언어

        Returns:
            str: 생성된 응답

        Raises:
            TimeoutError: 쿼리가 타임아웃을 초과할 경우
            Exception: 쿼리 실패 시
        """
        session_id = self.request.meta.session_id
        logger.debug(f"[{session_id}] Starting LLM query")

        start_time = time.time()

        # 컨텍스트 준비
        context = {
            "input": self.request.chat.user,
            "context": documents,
            "history": "",  # ollama에서는 빈 값으로 전달
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

        result = None

        try:
            if settings.llm.llm_backend.lower() == "ollama":
                logger.debug(f"[{session_id}] Starting Ollama chain invocation")

                # 체인 호출 방식 결정 (동기/비동기)
                if hasattr(self.chain, "ainvoke") and callable(getattr(self.chain, "ainvoke")):
                    # 코루틴에 대한 직접 await
                    logger.debug(f"[{session_id}] Ollama 체인 비동기 호출")
                    try:
                        result = await self.chain.ainvoke(context)
                    except Exception as e:
                        logger.error(f"[{session_id}] 비동기 체인 호출 오류: {str(e)}")
                        raise
                elif hasattr(self.chain, "invoke") and callable(getattr(self.chain, "invoke")):
                    # 동기 함수를 별도 스레드에서 실행
                    logger.debug(f"[{session_id}] Ollama 체인 동기 호출 (스레드 풀 사용)")
                    try:
                        result = await asyncio.to_thread(self.chain.invoke, context)
                    except Exception as e:
                        logger.error(f"[{session_id}] 동기 체인 호출 오류: {str(e)}")
                        raise
                else:
                    raise ValueError(f"[{session_id}] 올바른 체인 호출 메서드를 찾을 수 없습니다")

                # 결과가 코루틴인지 확인하고 추가 await 처리
                if asyncio.iscoroutine(result):
                    logger.debug(f"[{session_id}] 코루틴 응답 감지, 추가 await 처리")
                    result = await result
            elif settings.llm.llm_backend.lower() == "vllm":
                logger.debug(f"[{session_id}] Starting vLLM invocation")

                # 모델이 Gemma인지 확인 (클래스 메서드로 호출)
                use_gemma_format = BaseService.is_gemma_model(settings)

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
                result = response.get("generated_text", "")

            # 결과가 없거나 빈 문자열인 경우 처리
            if result is None or (isinstance(result, str) and result.strip() == ""):
                logger.warning(f"[{session_id}] 빈 응답 결과, 기본 메시지 반환")
                result = "죄송합니다. 응답을 생성할 수 없습니다. 다시 시도해 주세요."

            elapsed = time.time() - start_time
            logger.info(
                f"[{session_id}] LLM query complete: {elapsed:.4f}s elapsed "
                f"[backend={settings.llm.llm_backend}]"
            )

            # 서비스 메트릭 업데이트
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            return result
        except Exception as e:
            self.metrics["error_count"] += 1
            return self.error_handler.handle_error(
                error=e,
                session_id=self.request.meta.session_id,
                request=self.request
            )
