"""
vLLM 백엔드 구현 모듈

vLLM API를 사용하여 LLM 서비스와 통신하는 백엔드를 구현합니다.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, AsyncGenerator, Optional, Tuple

from langchain_core.documents import Document

from src.common.config_loader import ConfigLoader
from src.common.restclient import rc
from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.chat.backends.base import LLMBackend
from src.services.chat.utils.circuit_breaker import CircuitBreaker
from src.services.history.llm_history_handler import LlmHistoryHandler
from src.services.response_generator import ResponseGenerator
from src.services.chat.utils.async_utils import async_retry

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class VLLMBackend(LLMBackend):
    """
    vLLM 백엔드 구현

    vLLM API 서버를 통해 LLM 추론을 실행합니다.
    """

    def __init__(self, request: ChatRequest = None):
        """
        vLLM 백엔드 초기화

        Args:
            request: 초기 요청 (옵션)
        """
        self.request = request
        self.settings = settings

        # vLLM 엔드포인트 설정
        self.endpoint_url = settings.vllm.endpoint_url

        # 서킷 브레이커 초기화
        self.circuit_breaker = CircuitBreaker(
            name="vllm",
            failure_threshold=settings.circuit_breaker.failure_threshold,
            recovery_timeout=settings.circuit_breaker.recovery_timeout,
            reset_timeout=settings.circuit_breaker.reset_timeout
        )

        # 성능 메트릭
        self.metrics = {
            "request_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "total_time": 0,
            "avg_response_time": 0
        }

        # 히스토리 핸들러
        self.history_handler = None
        if request and settings.chat_history.enabled:
            self._init_history_handler()

        # 응답 생성기 및 쿼리 체크 딕셔너리 초기화
        from src.common.query_check_dict import QueryCheckDict
        self.query_check_dict = QueryCheckDict(settings.prompt.llm_prompt_path)
        self.response_generator = ResponseGenerator(settings, self.query_check_dict)

    @property
    def backend_name(self) -> str:
        """백엔드 이름 반환"""
        return "vLLM"

    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        return getattr(settings.llm, 'model_type', 'unknown')

    def _init_history_handler(self):
        """히스토리 핸들러 초기화"""
        if self.request is None:
            return

        # 모델 유형에 따라서 llm_model 파라미터는 다름
        llm_model = None  # vLLM은 실제 모델 객체가 필요 없음

        self.history_handler = LlmHistoryHandler(
            llm_model,
            self.request,
            max_history_turns=getattr(settings.chat_history, 'max_turns', 10)
        )

    async def initialize(self) -> bool:
        """백엔드 초기화"""
        # vLLM은 API 서버이므로 별도 초기화 없음
        return True

    async def generate(
            self,
            request: ChatRequest,
            language: str,
            documents: List[Document] = None
    ) -> Dict[str, Any]:
        """
        vLLM으로 텍스트 생성 요청 처리

        Args:
            request: 채팅 요청
            language: 응답 언어
            documents: 컨텍스트 문서 (기본값: None)

        Returns:
            Dict[str, Any]: 응답 데이터
        """
        self.request = request
        session_id = request.meta.session_id

        try:
            # 컨텍스트 준비
            context = {
                "input": request.chat.user,
                "context": documents or [],
                "language": language,
                "today": self.response_generator.get_today(),
            }

            # VOC 관련 설정 추가 (필요한 경우)
            if request.meta.rag_sys_info == "komico_voc":
                context.update({
                    "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                    "check_gw_word_link": settings.voc.check_gw_word_link,
                    "check_gw_word": settings.voc.check_gw_word,
                    "check_block_line": settings.voc.check_block_line,
                })

            # RAG 프롬프트 템플릿 가져오기
            rag_prompt_template = self.response_generator.get_rag_qa_prompt(request.meta.rag_sys_info)

            # 시스템 프롬프트 구성
            vllm_inquery_context = await asyncio.to_thread(
                self._build_system_prompt,
                rag_prompt_template,
                context
            )

            # vLLM 요청 생성
            vllm_request = VllmInquery(
                request_id=session_id,
                prompt=vllm_inquery_context,
                stream=False
            )

            # vLLM 엔드포인트 호출
            start_time = time.time()
            response = await self._call_vllm_endpoint(vllm_request)
            elapsed = time.time() - start_time

            # 메트릭 업데이트
            self.metrics["request_count"] += 1
            self.metrics["success_count"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["avg_response_time"] = self.metrics["total_time"] / self.metrics["request_count"]

            # 응답 추출 및 반환
            answer = response.get("generated_text", "") or response.get("answer", "")

            return {
                "answer": answer,
                "documents": documents or [],
                "metadata": {"elapsed_time": elapsed}
            }

        except Exception as e:
            # 메트릭 업데이트
            self.metrics["failure_count"] += 1

            logger.error(f"[{session_id}] vLLM 생성 중 오류: {str(e)}", exc_info=True)
            raise

    async def process_with_history(
            self,
            request: ChatRequest,
            language: str,
            documents: List[Document] = None
    ) -> Dict[str, Any]:
        """
        대화 이력을 고려한 텍스트 생성

        Args:
            request: 채팅 요청
            language: 응답 언어
            documents: 컨텍스트 문서 (기본값: None)

        Returns:
            Dict[str, Any]: 응답 데이터
        """
        # 요청 업데이트
        self.request = request
        session_id = request.meta.session_id

        # 히스토리 핸들러가 초기화되지 않았으면 초기화
        if self.history_handler is None:
            self._init_history_handler()

        try:
            # 히스토리 핸들러에 검색기 초기화
            await self.history_handler.init_retriever(documents or [])

            # Gemma 모델 여부 확인
            is_gemma_model = self._is_gemma_model()

            # 개선된 히스토리 처리 사용 여부 확인
            use_improved_history = getattr(settings.llm, 'use_improved_history', False)

            # 히스토리 처리 방식 결정 및 실행
            if is_gemma_model:
                logger.info(f"[{session_id}] Gemma 모델 감지, Gemma 히스토리 핸들러 사용")
                result = await self.history_handler.handle_chat_with_history_vllm(
                    request, language
                )
            else:
                logger.info(f"[{session_id}] 일반 모델, {'개선된' if use_improved_history else '기본'} 히스토리 처리 사용")
                result = await self.history_handler.handle_chat_with_history_vllm(
                    request, language
                )

            # 응답 추출
            if result:
                answer, retrieval_documents = result
                return {
                    "answer": answer,
                    "documents": retrieval_documents or documents or [],
                    "metadata": {}
                }
            else:
                raise ValueError("히스토리 처리 결과가 없습니다")

        except Exception as e:
            # 메트릭 업데이트
            self.metrics["failure_count"] += 1

            logger.error(f"[{session_id}] 히스토리 처리 중 오류: {str(e)}", exc_info=True)

            # 히스토리 처리 실패 시 일반 생성으로 폴백
            return await self.generate(request, language, documents)

    async def prepare_streaming_request(
            self,
            request: ChatRequest,
            language: str,
            documents: List[Document] = None
    ) -> Any:
        """
        스트리밍 요청 준비

        Args:
            request: 채팅 요청
            language: 응답 언어
            documents: 컨텍스트 문서 (기본값: None)

        Returns:
            Any: 스트리밍 요청 객체
        """
        self.request = request
        session_id = request.meta.session_id

        # 히스토리 활성화 확인
        if settings.chat_history.enabled:
            # 히스토리 핸들러가 초기화되지 않았으면 초기화
            if self.history_handler is None:
                self._init_history_handler()

            # 개선된 히스토리 처리 사용 여부 확인
            use_improved_history = getattr(settings.llm, 'use_improved_history', False)

            # 히스토리 핸들러에 검색기 초기화
            await self.history_handler.init_retriever(documents or [])

            # 모델 유형에 따른 스트리밍 핸들러 디스패치
            if self._is_gemma_model():
                logger.info(f"[{session_id}] Gemma 모델 감지됨, Gemma 스트리밍 핸들러로 처리")
                vllm_request, _ = await self.history_handler.handle_chat_with_history_gemma_streaming(
                    request, language
                )
            else:
                # 개선된 히스토리 처리 기능 확인
                if use_improved_history:
                    # 개선된 2단계 접근법 사용
                    vllm_request, _ = await self.history_handler.handle_chat_with_history_vllm_streaming_improved(
                        request, language
                    )
                else:
                    # 기존 방식 사용
                    vllm_request, _ = await self.history_handler.handle_chat_with_history_vllm_streaming(
                        request, language
                    )

            return vllm_request
        else:
            # 히스토리 없는 일반 스트리밍 준비
            # 컨텍스트 준비
            context = {
                "input": request.chat.user,
                "context": documents or [],
                "language": language,
                "today": self.response_generator.get_today(),
            }

            # VOC 관련 설정 추가 (필요한 경우)
            if request.meta.rag_sys_info == "komico_voc":
                context.update({
                    "gw_doc_id_prefix_url": settings.voc.gw_doc_id_prefix_url,
                    "check_gw_word_link": settings.voc.check_gw_word_link,
                    "check_gw_word": settings.voc.check_gw_word,
                    "check_block_line": settings.voc.check_block_line,
                })

            # RAG 프롬프트 템플릿 가져오기
            rag_prompt_template = self.response_generator.get_rag_qa_prompt(request.meta.rag_sys_info)

            # 시스템 프롬프트 구성
            vllm_inquery_context = await asyncio.to_thread(
                self._build_system_prompt,
                rag_prompt_template,
                context
            )

            # vLLM 요청 생성
            vllm_request = VllmInquery(
                request_id=session_id,
                prompt=vllm_inquery_context,
                stream=True  # 스트리밍 모드 활성화
            )

            return vllm_request

    async def stream_generate(
            self,
            request: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 방식으로 텍스트 생성

        Args:
            request: 스트리밍 요청 (VllmInquery)

        Yields:
            Dict[str, Any]: 청크 데이터
        """
        if not isinstance(request, VllmInquery):
            raise ValueError("스트리밍 요청은 VllmInquery 유형이어야 합니다")

        session_id = request.request_id

        try:
            # 서킷 브레이커 확인
            if self.circuit_breaker.is_open():
                logger.warning(f"[{session_id}] 서킷 브레이커가 열려있어 요청을 건너뜁니다")
                yield {
                    "error": True,
                    "message": "서비스를 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
                    "finished": True
                }
                return

            # 스트리밍 응답 시작
            async for chunk in rc.restapi_stream_async(session_id, self.endpoint_url, request):
                if chunk is None:
                    continue

                # 청크 처리 및 표준화
                processed_chunk = self._process_vllm_chunk(chunk)

                # 성공 청크면 서킷 브레이커 성공 기록
                if not processed_chunk.get("error", False):
                    self.circuit_breaker.record_success()

                yield processed_chunk

                # 마지막 청크 또는 오류 청크면 종료
                if processed_chunk.get("finished", False) or processed_chunk.get("error", False):
                    break

        except Exception as e:
            # 서킷 브레이커 실패 기록
            self.circuit_breaker.record_failure()

            # 메트릭 업데이트
            self.metrics["failure_count"] += 1

            logger.error(f"[{session_id}] vLLM 스트리밍 중 오류: {str(e)}", exc_info=True)

            # 오류 청크 반환
            yield {
                "error": True,
                "message": f"스트리밍 오류: {str(e)}",
                "finished": True
            }

    def _process_vllm_chunk(self, chunk: Any) -> Dict[str, Any]:
        """
        vLLM 응답 청크를 표준 형식으로 처리합니다.

        Args:
            chunk: 원시 vLLM 응답 청크

        Returns:
            Dict[str, Any]: 처리된 청크
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
                'text': '',
                'finished': True
            }

        # vLLM의 다양한 응답 형식 처리
        if isinstance(chunk, dict):
            # 텍스트 청크 (일반 스트리밍)
            if 'new_text' in chunk:
                return {
                    'text': chunk['new_text'],
                    'finished': chunk.get('finished', False)
                }
            # 완료 신호
            elif 'finished' in chunk and chunk['finished']:
                return {
                    'text': '',
                    'finished': True
                }
            # 전체 텍스트 응답 (비스트리밍 형식)
            elif 'generated_text' in chunk:
                return {
                    'text': chunk['generated_text'],
                    'finished': True
                }
            # OpenAI 호환 형식
            elif 'delta' in chunk:
                return {
                    'text': chunk['delta'].get('content', ''),
                    'finished': chunk.get('finished', False)
                }
            # 알 수 없는 형식
            else:
                # 가능한 경우 텍스트 추출 시도
                for key in ['text', 'content', 'message']:
                    if key in chunk:
                        return {
                            'text': chunk[key],
                            'finished': chunk.get('finished', False)
                        }
                return {
                    'text': str(chunk),
                    'finished': False
                }

        # 문자열 응답 (드문 경우)
        elif isinstance(chunk, str):
            return {
                'text': chunk,
                'finished': False
            }

        # 기타 타입 처리
        return {
            'text': str(chunk),
            'finished': False
        }

    def _build_system_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트 템플릿을 컨텍스트로 채워 완성합니다.

        Args:
            template: 프롬프트 템플릿
            context: 컨텍스트 변수

        Returns:
            str: 완성된 시스템 프롬프트
        """
        session_id = self.request.meta.session_id

        try:
            # Gemma 모델 여부 확인
            if self._is_gemma_model():
                return self._build_gemma_prompt(template, context)

            # 일반 프롬프트 생성
            prompt = template.format(**context)
            return prompt
        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"[{session_id}] 프롬프트 템플릿에 키가 누락됨: {missing_key}")
            context[missing_key] = ""
            return self._build_system_prompt(template, context)
        except Exception as e:
            logger.error(f"[{session_id}] 시스템 프롬프트 생성 중 오류: {str(e)}")
            # 기본 프롬프트 생성
            return f"다음 질문에 답변해 주세요: {context.get('input', '질문 없음')}"

    def _build_gemma_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """
        Gemma 모델에 맞는 형식으로 시스템 프롬프트를 구성합니다.

        Args:
            template: 프롬프트 템플릿
            context: 컨텍스트 변수

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        try:
            # 먼저 기존 함수로 프롬프트 생성
            raw_prompt = template.format(**context)

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
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}")
            context[missing_key] = ""
            return self._build_gemma_prompt(template, context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = (f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n"
                            f"<end_of_turn>\n<start_of_turn>model\n")
            return basic_prompt

    def _is_gemma_model(self) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # 지정된 모델 타입 확인
        if hasattr(settings.llm, 'model_type'):
            model_type = settings.llm.model_type.lower() \
                if hasattr(settings.llm.model_type, 'lower') \
                else str(settings.llm.model_type).lower()
            return model_type == 'gemma'

        return False

    @async_retry(max_retries=2, backoff_factor=2)
    async def _call_vllm_endpoint(self, data: VllmInquery) -> Dict[str, Any]:
        """
        vLLM 엔드포인트를 호출합니다.

        Args:
            data: vLLM 요청 데이터

        Returns:
            Dict[str, Any]: vLLM 응답

        Raises:
            Exception: vLLM 엔드포인트 호출 실패 시
        """
        start_time = time.time()
        session_id = data.request_id
        logger.debug(f"[{session_id}] vLLM 엔드포인트 호출 (stream={data.stream})")

        # 서킷 브레이커 확인
        if self.circuit_breaker.is_open():
            logger.warning(f"[{session_id}] 서킷 브레이커가 열려있어 요청을 건너뜁니다")
            raise RuntimeError("vLLM 서비스를 사용할 수 없음: 서킷 브레이커가 열려 있습니다")

        try:
            response = await rc.restapi_post_async(self.endpoint_url, data)
            elapsed = time.time() - start_time
            logger.debug(f"[{session_id}] vLLM 응답 수신: {elapsed:.4f}초 소요")

            # 서킷 브레이커 성공 기록
            self.circuit_breaker.record_success()

            return response
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] vLLM 엔드포인트 오류: {elapsed:.4f}초 후: {str(e)}")

            # 서킷 브레이커 실패 기록
            self.circuit_breaker.record_failure()

            raise

    def is_available(self) -> bool:
        """
        백엔드 가용성 확인

        Returns:
            bool: 백엔드가 사용 가능하면 True
        """
        # 서킷 브레이커 상태 확인
        return not self.circuit_breaker.is_open()

    def get_metrics(self) -> Dict[str, Any]:
        """
        성능 메트릭 조회

        Returns:
            Dict[str, Any]: 성능 메트릭
        """
        return self.metrics
