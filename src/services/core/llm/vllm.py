"""
VLLM LLM 서비스 구현 모듈
=======================

이 모듈은 VLLM API를 사용하여 LLM 서비스를 구현합니다.
LLMServiceBase 인터페이스를 구현하여 VLLM 백엔드와의 통신을 담당합니다.

기능:
- VLLM API 연결 및 초기화
- 동기 및 비동기 추론 처리
- 스트리밍 응답 지원
- 프롬프트 생성 및 응답 처리
- 메트릭 수집 및 모니터링
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.common.restclient import rc
from src.schema.vllm_inquery import VllmInquery
from src.services.core.llm.base import LLMServiceBase
from src.services.core.llm.factory import LLMServiceFactory

# 로거 설정
logger = logging.getLogger(__name__)


class VLLMLLMService(LLMServiceBase):
    """
    VLLM 백엔드를 사용하는 LLM 서비스 구현

    VLLM API를 통해 LLM 모델과 통신하고, 응답을 생성합니다.
    LLMServiceBase를 상속받아 모든 필수 메서드를 구현합니다.
    """

    def __init__(self, settings):
        """
        VLLM LLM 서비스 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.vllm_settings = getattr(settings, 'vllm', None)
        self.llm_settings = getattr(settings, 'llm', None)

        # 초기화 상태 플래그
        self.is_initialized = False

        # 세마포어 (동시 요청 제한)
        max_concurrent = getattr(settings.cache, 'max_concurrent_tasks', 5)
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # 회로 차단기 설정
        self._circuit_breaker_enabled = getattr(settings.circuit_breaker, 'enabled', True)
        self._failure_threshold = getattr(settings.circuit_breaker, 'failure_threshold', 3)
        self._failure_count = 0
        self._circuit_open = False
        self._reset_time = None

    async def initialize(self) -> bool:
        """
        VLLM 서비스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        if self.is_initialized:
            return True

        try:
            if not self.vllm_settings:
                logger.error("VLLM 설정이 없습니다.")
                return False

            # 엔드포인트 URL 확인
            endpoint_url = getattr(self.vllm_settings, 'endpoint_url', None)
            if not endpoint_url:
                logger.error("VLLM 엔드포인트 URL 설정이 없습니다.")
                return False

            # 기본 설정 확인
            self.endpoint_url = endpoint_url
            self.streaming_enabled = getattr(self.llm_settings, 'steaming_enabled', False)
            self.timeout = getattr(self.llm_settings, 'timeout', 60)
            self.model_type = getattr(self.llm_settings, 'model_type', '').lower()

            # 간단한 상태 점검 요청
            test_request = VllmInquery(
                request_id="init_test",
                prompt="test",
                stream=False
            )

            # 상태 확인 (5초 타임아웃)
            try:
                response = await asyncio.wait_for(
                    rc.restapi_post_async(self.endpoint_url, test_request),
                    timeout=5
                )
                if not response or response.get("status", 0) == 500:
                    logger.error(f"VLLM 서비스 상태 확인 실패: {response}")
                    return False

                logger.info(f"VLLM 서비스 초기화 완료 (엔드포인트: {self.endpoint_url})")
                self.is_initialized = True
                return True

            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"VLLM 서비스 상태 확인 중 오류: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"VLLM 서비스 초기화 실패: {str(e)}")
            return False

    def _check_circuit_breaker(self) -> bool:
        """
        회로 차단기 상태 확인

        Returns:
            bool: 요청 허용 여부 (True: 허용, False: 차단)
        """
        if not self._circuit_breaker_enabled:
            return True

        if not self._circuit_open:
            return True

        # 회로가 열려있고 리셋 시간이 지났는지 확인
        current_time = time.time()
        if self._reset_time and current_time > self._reset_time:
            # 반열림 상태로 전환
            logger.info("회로 차단기 반열림 상태로 전환")
            self._circuit_open = False
            self._failure_count = 0
            return True

        return False

    def _record_success(self):
        """
        성공 기록 및 회로 차단기 상태 업데이트
        """
        if self._circuit_breaker_enabled:
            self._failure_count = 0
            self._circuit_open = False
            self._reset_time = None

    def _record_failure(self):
        """
        실패 기록 및 회로 차단기 상태 업데이트
        """
        if not self._circuit_breaker_enabled:
            return

        self._failure_count += 1

        if self._failure_count >= self._failure_threshold:
            if not self._circuit_open:
                logger.warning(f"회로 차단기 열림: 연속 {self._failure_count}회 실패")
                self._circuit_open = True
                # 30초 후 재시도
                self._reset_time = time.time() + 30

    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        VLLM 엔드포인트 호출

        Args:
            data: VLLM 요청 데이터

        Returns:
            Dict: VLLM 응답

        Raises:
            RuntimeError: 회로 차단기가 열린 경우
            Exception: API 호출 실패
        """
        # 회로 차단기 확인
        if not self._check_circuit_breaker():
            logger.warning("회로 차단기가 열려 있어 요청을 차단합니다.")
            raise RuntimeError("VLLM 서비스 사용할 수 없음: 회로 차단기가 열려 있습니다")

        start_time = time.time()
        session_id = data.request_id
        logger.debug(f"[{session_id}] VLLM 엔드포인트 호출 (stream={data.stream})")

        try:
            response = await rc.restapi_post_async(self.endpoint_url, data)
            elapsed = time.time() - start_time
            logger.debug(f"[{session_id}] VLLM 응답 수신: {elapsed:.4f}초 소요")

            # 메트릭 업데이트
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            # 토큰 수 추적 (가능한 경우)
            if "usage" in response:
                usage = response.get("usage", {})
                self.metrics["token_count"] += usage.get("total_tokens", 0)

            # 회로 차단기 성공 기록
            self._record_success()

            return response

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] VLLM 엔드포인트 오류: {elapsed:.4f}초 후: {str(e)}")

            self.metrics["error_count"] += 1

            # 회로 차단기 실패 기록
            self._record_failure()

            raise

    async def _stream_vllm_response(self, session_id: str, data: VllmInquery):
        """
        VLLM 스트리밍 응답 처리

        Args:
            session_id: 세션 ID
            data: VLLM 요청 데이터

        Yields:
            Dict: 응답 청크
        """
        from src.common.restclient import rc
        start_time = time.time()

        try:
            logger.debug(f"[{session_id}] VLLM 스트리밍 시작")

            # 회로 차단기 확인
            if not self._check_circuit_breaker():
                logger.warning(f"[{session_id}] 회로 차단기가 열려 있어 스트리밍 요청을 차단합니다.")
                yield {
                    "error": True,
                    "message": "서비스를 일시적으로 사용할 수 없습니다. 잠시 후 다시 시도해 주세요.",
                    "finished": True
                }
                return

            # 스트리밍 응답 처리
            async for chunk in rc.restapi_stream_async(session_id, self.endpoint_url, data):
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
                    self._record_success()

                    # 메트릭 업데이트
                    elapsed = time.time() - start_time
                    self.metrics["request_count"] += 1
                    self.metrics["total_time"] += elapsed

                    logger.debug(f"[{session_id}] VLLM 스트리밍 완료: {elapsed:.4f}초 소요")
                    break

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] VLLM 스트리밍 오류: {elapsed:.4f}초 후: {str(e)}")

            # 메트릭 업데이트
            self.metrics["error_count"] += 1

            # 회로 차단기 실패 기록
            self._record_failure()

            # 오류 청크 반환
            yield {
                "error": True,
                "message": f"스트리밍 오류: {str(e)}",
                "finished": True
            }

    @classmethod
    def _process_vllm_chunk(cls, chunk):
        """
        VLLM 응답 청크를 표준 형식으로 처리

        Args:
            chunk: 원시 VLLM 응답 청크

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

        # VLLM의 다양한 응답 형식 처리
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

    def build_system_prompt(self, template: Union[str, PromptTemplate], context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트 생성

        Args:
            template: 프롬프트 템플릿 (문자열 또는 PromptTemplate)
            context: 프롬프트 컨텍스트

        Returns:
            str: 완성된 시스템 프롬프트
        """
        try:
            # 템플릿이 PromptTemplate 인스턴스인 경우
            if isinstance(template, PromptTemplate):
                return template.format(**context)

            # 이미지 데이터 처리
            if "image" in context and context["image"]:
                if "image_description" not in context:
                    context["image_description"] = self._format_image_data(context["image"])

                # 프롬프트 템플릿에 이미지 설명 토큰이 없으면 추가
                if '{image_description}' not in template:
                    insert_point = template.find('{input}')
                    if insert_point > 0:
                        image_instruction = "\n\n# 이미지 정보\n다음은 사용자가 제공한 이미지에 대한 정보입니다:\n{image_description}\n\n# 질문\n"
                        template = (
                                template[:insert_point] +
                                image_instruction +
                                template[insert_point:]
                        )

            # Gemma 모델인 경우 특별 처리
            if self.is_gemma_model():
                return self.build_system_prompt_gemma(template, context)

            # 문자열 템플릿인 경우
            return template.format(**context)

        except KeyError as e:
            # 누락된 키가 있는 경우 빈 문자열로 대체
            missing_key = str(e).strip("'")
            logger.warning(f"프롬프트 템플릿에 누락된 키: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt(template, context)

        except Exception as e:
            logger.error(f"시스템 프롬프트 생성 중 오류: {str(e)}")
            # 기본 프롬프트로 대체
            return f"다음 질문에 답변하세요: {context.get('input', '질문 없음')}"

    def is_gemma_model(self) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        return self.model_type == 'gemma'

    def build_system_prompt_gemma(self, template: str, context: Dict[str, Any]) -> str:
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성

        Args:
            template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        try:
            # 먼저 기존 format 메서드로 변수를 대체
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
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt_gemma(template, context)

        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = (f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n<end_of_turn>\n"
                            f"<start_of_turn>model\n")
            return basic_prompt

    async def ask(self, query: str, documents: List[Document], language: str,
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        VLLM LLM에 질의하고 응답 반환

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            str: LLM 응답 텍스트
        """
        if not self.is_initialized:
            logger.error("VLLM 모델이 초기화되지 않았습니다.")
            return "죄송합니다. LLM 모델이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."

        start_time = time.time()
        session_id = context.get("session_id") if context else "unknown"

        try:
            # 컨텍스트 준비
            prompt_context = {
                "input": query,
                "context": documents,
                "language": language,
                "today": self._get_today(),
            }

            # 추가 컨텍스트 병합
            if context:
                prompt_context.update(context)

            # 이미지 처리
            if "image" in prompt_context and prompt_context["image"]:
                prompt_context["image_description"] = self._format_image_data(prompt_context["image"])

            # VOC 관련 컨텍스트 추가
            rag_sys_info = prompt_context.get("rag_sys_info")
            if rag_sys_info == "komico_voc":
                voc_settings = getattr(self.settings, 'voc', None)
                if voc_settings:
                    prompt_context.update({
                        "gw_doc_id_prefix_url": getattr(voc_settings, 'gw_doc_id_prefix_url', ''),
                        "check_gw_word_link": getattr(voc_settings, 'check_gw_word_link', ''),
                        "check_gw_word": getattr(voc_settings, 'check_gw_word', ''),
                        "check_block_line": getattr(voc_settings, 'check_block_line', ''),
                    })

            # 프롬프트 템플릿 가져오기
            from src.services.response_generator import ResponseGenerator
            from src.common.query_check_dict import QueryCheckDict

            query_check_dict = QueryCheckDict(self.settings.prompt.llm_prompt_path)
            response_generator = ResponseGenerator(self.settings, query_check_dict)
            prompt_template = response_generator.get_rag_qa_prompt(rag_sys_info or "")

            # 시스템 프롬프트 생성
            vllm_inquery_context = self.build_system_prompt(prompt_template, prompt_context)

            # VLLM 요청 생성
            vllm_request = VllmInquery(
                request_id=session_id,
                prompt=vllm_inquery_context,
                stream=False
            )

            # VLLM 엔드포인트 호출
            response = await self.call_vllm_endpoint(vllm_request)
            result = response.get("generated_text", "")

            # 메트릭 업데이트
            elapsed = time.time() - start_time
            logger.info(
                f"[{session_id}] VLLM 쿼리 완료: {elapsed:.4f}초 소요"
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{session_id}] VLLM 쿼리 실패: {elapsed:.4f}초 후: {type(e).__name__}: {str(e)}"
            )

            self.metrics["error_count"] += 1

            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def stream_response(self, query: str, documents: List[Document], language: str,
                              context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 모드로 VLLM LLM에 질의하고 응답 생성

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            AsyncGenerator: 응답 청크를 생성하는 비동기 제너레이터
        """
        if not self.is_initialized:
            logger.error("VLLM 모델이 초기화되지 않았습니다.")
            yield {
                "error": True,
                "text": "LLM 모델이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요.",
                "finished": True
            }
            return

        session_id = context.get("session_id") if context else "unknown"
        logger.debug(f"[{session_id}] 스트리밍 응답 시작")

        try:
            # 컨텍스트 준비
            prompt_context = {
                "input": query,
                "context": documents,
                "language": language,
                "today": self._get_today(),
            }

            # 추가 컨텍스트 병합
            if context:
                prompt_context.update(context)

            # 이미지 처리
            if "image" in prompt_context and prompt_context["image"]:
                prompt_context["image_description"] = self._format_image_data(prompt_context["image"])

            # VOC 관련 컨텍스트 추가
            rag_sys_info = prompt_context.get("rag_sys_info")
            if rag_sys_info == "komico_voc":
                voc_settings = getattr(self.settings, 'voc', None)
                if voc_settings:
                    prompt_context.update({
                        "gw_doc_id_prefix_url": getattr(voc_settings, 'gw_doc_id_prefix_url', ''),
                        "check_gw_word_link": getattr(voc_settings, 'check_gw_word_link', ''),
                        "check_gw_word": getattr(voc_settings, 'check_gw_word', ''),
                        "check_block_line": getattr(voc_settings, 'check_block_line', ''),
                    })

            # 프롬프트 템플릿 가져오기
            from src.services.response_generator import ResponseGenerator
            from src.common.query_check_dict import QueryCheckDict

            query_check_dict = QueryCheckDict(self.settings.prompt.llm_prompt_path)
            response_generator = ResponseGenerator(self.settings, query_check_dict)
            prompt_template = response_generator.get_rag_qa_prompt(rag_sys_info or "")

            # 시스템 프롬프트 생성
            vllm_inquery_context = self.build_system_prompt(prompt_template, prompt_context)

            # 스트리밍을 위한 VLLM 요청 생성
            vllm_request = VllmInquery(
                request_id=session_id,
                prompt=vllm_inquery_context,
                stream=True  # 스트리밍 모드 활성화
            )

            # 스트리밍 응답 처리
            async for chunk in self._stream_vllm_response(session_id, vllm_request):
                if 'new_text' in chunk:
                    yield {
                        "text": chunk['new_text'],
                        "finished": chunk.get('finished', False)
                    }
                elif chunk.get('finished', False):
                    yield {"text": "", "finished": True}
                elif chunk.get('error', False):
                    yield {
                        "error": True,
                        "text": chunk.get('message', '오류가 발생했습니다.'),
                        "finished": True
                    }

        except Exception as e:
            logger.error(f"[{session_id}] 스트리밍 응답 생성 중 오류: {str(e)}")
            yield {
                "error": True,
                "text": f"스트리밍 응답 생성 중 오류가 발생했습니다: {str(e)}",
                "finished": True
            }

    @staticmethod
    def _format_image_data(image_data: Dict[str, str]) -> str:
        """
        이미지 데이터를 프롬프트에 추가하기 위한 형식으로 변환

        Args:
            image_data: 이미지 데이터 (base64, URL 등)

        Returns:
            str: 포맷된 이미지 정보
        """
        if 'base64' in image_data:
            return "[이미지 데이터가 base64 형식으로 전달되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"
        elif 'url' in image_data:
            return f"[이미지 URL: {image_data.get('url')}]"
        elif 'description' in image_data:
            return f"[이미지 설명: {image_data.get('description')}]"
        else:
            return "[이미지 데이터가 제공되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"

    @staticmethod
    def _get_today() -> str:
        """
        현재 날짜와 요일을 한국어 형식으로 반환

        Returns:
            str: 형식화된 날짜 문자열
        """
        today = datetime.now()
        weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        weekday = weekday_names[today.weekday()]

        return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."


# 팩토리에 서비스 등록
LLMServiceFactory.register_service("vllm", VLLMLLMService)
