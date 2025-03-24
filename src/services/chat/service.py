"""
채팅 서비스 핵심 모듈

이 모듈은 다양한 LLM 백엔드를 지원하는 채팅 시스템의 중앙 조정 역할을 합니다.
여러 처리 단계를 관리하고 캐싱, 오류 처리, 응답 생성을 조정합니다.
"""

import asyncio
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, AsyncGenerator

from fastapi import BackgroundTasks
from starlette.responses import StreamingResponse

from src.common.config_loader import ConfigLoader
from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse, MetaRes, PayloadRes, ChatRes

from src.services.chat.backends.factory import BackendFactory
from src.services.chat.cache.manager import CacheManager
from src.services.chat.processors.pipeline import ChatPipeline
from src.services.chat.processors.post_processor import StreamResponsePostProcessor
from src.services.chat.utils.logging import AsyncLogger
from src.services.chat.utils.performance import PerformanceTracker
from src.services.document_processor import DocumentProcessor
from src.services.response_generator import ResponseGenerator
from src.services.query_processor import QueryProcessor

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class ChatService:
    """
    채팅 서비스 핵심 클래스

    다양한 LLM 백엔드를 지원하는 채팅 시스템을 조정하고 관리합니다.
    문서 검색, 쿼리 처리, 응답 생성 및 스트리밍을 처리합니다.
    """

    def __init__(self, request: ChatRequest):
        """
        채팅 서비스 초기화

        Args:
            request: 처리할 채팅 요청
        """
        self.request = request
        self.session_id = request.meta.session_id

        # 성능 추적 초기화
        self.performance = PerformanceTracker(self.session_id)

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager.get_instance()

        # 백엔드 팩토리를 통한 LLM 백엔드 초기화
        self.backend = BackendFactory.create_backend(request)

        # 처리 파이프라인 초기화
        self.pipeline = ChatPipeline(request)

        # 처리 구성요소 초기화
        self.query_processor = QueryProcessor(settings,
                                              self.pipeline.get_component("query_processor").query_check_dict)
        self.document_processor = DocumentProcessor(settings)
        self.response_generator = ResponseGenerator(
            settings, self.pipeline.get_component("response_generator").query_check_dict)

        # 비동기 로거 초기화
        self.logger = AsyncLogger.get_instance()

        # 캐시 키 생성
        self.cache_key = f"{self.request.meta.rag_sys_info}:{self.session_id}:{self.request.chat.user}"

    async def process_chat(self) -> ChatResponse:
        """
        채팅 요청을 처리하고 응답을 생성합니다.

        Returns:
            ChatResponse: 처리된 채팅 응답
        """
        await self.logger.log("info", f"[{self.session_id}] 채팅 요청 처리 시작")

        # 성능 추적 시작
        self.performance.start_tracking()

        # 캐시 확인
        cached_response = await self.cache_manager.get("response", self.cache_key)
        if cached_response:
            await self.logger.log("info", f"[{self.session_id}] 캐싱된 응답 반환")
            return cached_response

        try:
            # 처리 파이프라인 실행
            result = await self.pipeline.execute()

            # 조기 반환 확인
            if result.get("early_response"):
                response = result["early_response"]
                # 결과 캐싱 (필요한 경우)
                if result.get("cacheable", False):
                    await self.cache_manager.set("response", self.cache_key, response)
                return response

            # 일반 응답 처리
            # LLM 백엔드에 요청 전송
            llm_result = await self._process_llm_request(result)

            # 응답 후처리
            final_result = await self._post_process_response(llm_result, result)

            # 최종 응답 생성
            response = self._create_response(
                ErrorCd.get_error(ErrorCd.SUCCESS),
                final_result["answer"],
                result.get("documents", [])
            )

            # 결과 캐싱
            total_time = self.performance.get_total_elapsed()
            if total_time > 1.0:  # 빠른 응답은 캐싱하지 않음
                await self.cache_manager.set("response", self.cache_key, response)

            # 성능 데이터 추가
            if hasattr(response, 'add_performance_data'):
                response.add_performance_data(self.performance.get_metrics())

            await self.logger.log(
                "info",
                f"[{self.session_id}] 채팅 처리 완료: {total_time:.4f}초 소요",
                stages=self.performance.get_stages()
            )

            return response

        except ValueError as err:
            await self.logger.log(
                "error",
                f"[{self.session_id}] 채팅 처리 중 값 오류: {err}",
                exc_info=True
            )
            return self._create_response(
                ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION),
                "입력된 값이 올바르지 않습니다."
            )
        except Exception as err:
            await self.logger.log(
                "error",
                f"[{self.session_id}] 채팅 처리 중 예상치 못한 오류: {err}",
                exc_info=True
            )
            return self._create_response(
                ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION),
                "요청을 처리할 수 없습니다."
            )

    async def stream_chat(self, background_tasks: BackgroundTasks = None) -> StreamingResponse:
        """
        스트리밍 응답을 제공하는 채팅 처리 메서드

        Args:
            background_tasks: 백그라운드 작업 실행기

        Returns:
            StreamingResponse: 스트리밍 응답 객체
        """
        await self.logger.log("info", f"[{self.session_id}] 스트리밍 채팅 요청 시작")

        # 성능 추적 시작
        self.performance.start_tracking()

        try:
            # 인사말 또는 간단한 응답 처리
            greeting_result = await self.pipeline.execute_stages(["greeting_filter"])
            if greeting_result.get("early_response"):
                # 간단한 응답은 스트리밍으로 변환하여 반환
                return await self._simple_response_to_stream(greeting_result["early_response"])

            # 스트리밍 활성화 확인
            if not settings.llm.steaming_enabled:
                await self.logger.log(
                    "warning",
                    f"[{self.session_id}] 스트리밍이 설정에서 비활성화되어 있습니다."
                )
                return self._create_error_stream("스트리밍이 서버 설정에서 비활성화되어 있습니다.")

            # vLLM 백엔드 확인
            if settings.llm.llm_backend.lower() != "vllm":
                await self.logger.log(
                    "warning",
                    f"[{self.session_id}] 스트리밍은 vLLM 백엔드에서만 지원됩니다."
                )
                return self._create_error_stream("스트리밍은 vLLM 백엔드에서만 지원됩니다.")

            # 필요한 단계만 실행
            result = await self.pipeline.execute_stages(["language_processing", "document_retrieval"])

            # 스트리밍 설정
            stream_request = await self.backend.prepare_streaming_request(
                self.request, result["language"], result.get("documents", [])
            )

            # 스트리밍 응답 생성
            return await self._handle_streaming_response(
                stream_request,
                result.get("documents", []),
                background_tasks
            )

        except Exception as e:
            await self.logger.log(
                "error",
                f"[{self.session_id}] 스트리밍 초기화 중 오류: {str(e)}",
                exc_info=True
            )
            return self._create_error_stream(f"처리 중 오류가 발생했습니다: {str(e)}")

    async def _process_llm_request(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 백엔드에 요청을 전송하고 결과를 처리합니다.

        Args:
            context: 현재까지의 처리 컨텍스트

        Returns:
            Dict[str, Any]: LLM 응답 결과
        """
        self.performance.start_stage("llm_processing")

        # 히스토리 활성화 확인
        if settings.chat_history.enabled:
            # 히스토리 처리기를 통한 요청 처리
            result = await self.backend.process_with_history(
                self.request,
                context.get("language", "ko"),
                context.get("documents", [])
            )
        else:
            # 직접 LLM 요청 처리
            result = await self.backend.generate(
                self.request,
                context.get("language", "ko"),
                context.get("documents", [])
            )

        self.performance.end_stage("llm_processing")
        return result

    async def _post_process_response(
            self,
            llm_result: Dict[str, Any],
            context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM 응답에 대한 후처리를 수행합니다.

        Args:
            llm_result: LLM 응답 결과
            context: 처리 컨텍스트

        Returns:
            Dict[str, Any]: 처리된 결과
        """
        self.performance.start_stage("post_processing")

        # 응답 추출
        answer = llm_result.get("answer", "")
        retrieval_documents = llm_result.get("documents", context.get("documents", []))

        # 역할 마커 제거
        cleaned_answer = self._clean_role_markers(answer)

        # 히스토리 저장 (활성화된 경우)
        if settings.chat_history.enabled:
            self._save_chat_history_in_background(cleaned_answer)

        # 참조 정보 추가 및 형식 지정
        reference_word = context.get("reference_word", "[참고문헌]")
        final_answer = await self._finalize_answer(cleaned_answer, reference_word, retrieval_documents)

        self.performance.end_stage("post_processing")
        return {"answer": final_answer, "documents": retrieval_documents}

    def _clean_role_markers(self, text: str) -> str:
        """
        텍스트에서 역할 마커(AI:, Human: 등)를 제거합니다.

        Args:
            text: 정리할 텍스트

        Returns:
            str: 역할 마커가 제거된 텍스트
        """
        import re
        return re.sub(r'(AI:|Human:|Assistant:|User:)', '', text).strip()

    async def _finalize_answer(
            self,
            query_answer: str,
            reference_word: str,
            retrieval_document: List = None
    ) -> str:
        """
        응답에 참조와 링크를 추가하고 최종 형식을 지정합니다.

        Args:
            query_answer: 원본 LLM 응답
            reference_word: 참조 섹션 표시어
            retrieval_document: 검색된 문서 목록

        Returns:
            str: 최종 형식이 지정된 응답
        """
        # 참조 정보 추가
        if settings.prompt.source_count:
            try:
                # 참조 추가 작업을 스레드 풀에서 실행
                query_answer = await asyncio.to_thread(
                    self.response_generator.make_answer_reference,
                    query_answer,
                    self.request.meta.rag_sys_info,
                    reference_word,
                    retrieval_document or [],
                    self.request
                )
            except Exception as e:
                await self.logger.log(
                    "error",
                    f"[{self.session_id}] 참조 추가 중 오류: {str(e)}",
                    exc_info=True
                )

        # VOC 처리
        if "komico_voc" in settings.voc.voc_type.split(',') and self.request.meta.rag_sys_info == "komico_voc":
            try:
                from src.services.voc import VOCLinkProcessor
                voc_processor = VOCLinkProcessor(settings)

                # VOC 처리 작업을 스레드 풀에서 실행
                query_answer = await asyncio.to_thread(
                    voc_processor.process_voc_document_links,
                    query_answer
                )
            except Exception as e:
                await self.logger.log(
                    "error",
                    f"[{self.session_id}] VOC 처리 중 오류: {str(e)}",
                    exc_info=True
                )

        # URL 링크 처리
        try:
            from src.services.search_engine import SearchEngine
            search_engine = SearchEngine(settings)

            # URL 처리 작업을 스레드 풀에서 실행
            query_answer = await asyncio.to_thread(
                search_engine.replace_urls_with_links,
                query_answer
            )
        except Exception as e:
            await self.logger.log(
                "error",
                f"[{self.session_id}] URL 링크 처리 중 오류: {str(e)}",
                exc_info=True
            )

        return query_answer

    def _save_chat_history_in_background(self, answer: str) -> None:
        """
        채팅 이력을 비동기적으로 저장합니다.

        Args:
            answer: 저장할 응답 텍스트
        """
        if not settings.chat_history.enabled:
            return

        async def _save_history():
            try:
                from src.services.history.llm_history_handler import LlmHistoryHandler
                from src.utils.redis_utils import RedisUtils
                from src.services.messaging.formatters import MessageFormatter

                formatter = MessageFormatter()
                session_id = self.request.meta.session_id

                # 메시지 생성
                chat_data = formatter.create_chat_data(session_id, [
                    formatter.create_message("HumanMessage", self.request.chat.user),
                    formatter.create_message("AIMessage", answer)
                ])

                # Redis에 저장
                await RedisUtils.async_save_message_to_redis(
                    system_info=self.request.meta.rag_sys_info,
                    session_id=session_id,
                    message=chat_data
                )

                await self.logger.log("debug", f"[{session_id}] 채팅 이력이 성공적으로 저장됨")
            except Exception as e:
                await self.logger.log(
                    "error",
                    f"[{self.session_id}] 채팅 이력 저장 중 오류: {str(e)}",
                    exc_info=True
                )

        # 백그라운드 태스크로 저장
        self._fire_and_forget(_save_history())

    def _fire_and_forget(self, coro):
        """
        코루틴을 백그라운드에서 실행하고 예외를 안전하게 처리합니다.

        Args:
            coro: 실행할 코루틴
        """

        async def wrapper():
            try:
                await coro
            except Exception as e:
                logger.error(f"백그라운드 태스크 오류: {e}", exc_info=True)

        task = asyncio.create_task(wrapper())

        # 태스크 추적을 위한 클래스 변수가 없다면 생성
        if not hasattr(self.__class__, '_background_tasks'):
            self.__class__._background_tasks = []

        # 태스크 추적에 추가하고 완료 시 제거하는 콜백 설정
        self.__class__._background_tasks.append(task)
        task.add_done_callback(
            lambda t: self.__class__._background_tasks.remove(t)
            if t in self.__class__._background_tasks else None
        )

    async def _handle_streaming_response(
            self,
            stream_request: Any,
            documents: List,
            background_tasks: Optional[BackgroundTasks]
    ) -> StreamingResponse:
        """
        스트리밍 응답을 처리합니다.

        Args:
            stream_request: 스트리밍 요청 객체
            documents: 검색된 문서 목록
            background_tasks: 백그라운드 태스크 실행기

        Returns:
            StreamingResponse: 스트리밍 응답 객체
        """
        # 스트리밍 후처리기 초기화
        post_processor = StreamResponsePostProcessor(
            self.response_generator,
            self.request,
            documents
        )

        # 문자 단위 배치 설정
        char_buffer = ""  # 문자 버퍼
        max_buffer_time = 100  # 최대 100ms 지연 허용
        min_chars_to_send = 2  # 최소 2자 이상일 때 전송 (한글 자모 조합 고려)
        last_send_time = time.time()  # 마지막 전송 시간

        # 스트리밍 응답 생성기
        async def generate_stream():
            nonlocal char_buffer, last_send_time
            import json

            start_llm_time = time.time()
            error_occurred = False
            full_response = None  # 전체 응답을 저장할 변수

            try:
                # 백엔드로부터 스트리밍 시작
                async for chunk in self.backend.stream_generate(stream_request):
                    current_time = time.time()

                    # 청크 처리
                    if "error" in chunk:
                        # 오류 전송
                        error_json = json.dumps(
                            {'error': True, 'text': chunk.get("message", "알 수 없는 오류"), 'finished': True},
                            ensure_ascii=False
                        )
                        yield f"data: {error_json}\n\n"
                        error_occurred = True
                        break

                    # 텍스트 청크 처리
                    if "text" in chunk:
                        text_chunk = chunk.get("text", "")
                        is_finished = chunk.get("finished", False)

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

                            # 전체 완성된 응답 전송
                            json_data = json.dumps(
                                {'complete_response': full_response},
                                ensure_ascii=False
                            )
                            yield f"data: {json_data}\n\n"
                            break

                # LLM 처리 시간 기록
                llm_elapsed = time.time() - start_llm_time
                await self.logger.log(
                    "info",
                    f"[{self.session_id}] LLM 스트리밍 처리 완료: {llm_elapsed:.4f}초 소요",
                    session_id=self.session_id
                )
                self.performance.record_stage("llm_streaming", llm_elapsed)

                # 채팅 이력 저장
                if settings.chat_history.enabled and full_response:
                    if background_tasks:
                        # BackgroundTasks가 제공된 경우 활용
                        background_tasks.add_task(self._save_chat_history_in_background, full_response)
                    else:
                        # 직접 백그라운드로 실행
                        self._save_chat_history_in_background(full_response)

            except Exception as err:
                error_occurred = True
                await self.logger.log("error", f"[{self.session_id}] 스트리밍 처리 중 오류: {str(err)}", exc_info=True)

                # 오류 정보 전송
                error_json = json.dumps(
                    {'error': True, 'text': str(err), 'finished': True},
                    ensure_ascii=False
                )
                yield f"data: {error_json}\n\n"

            finally:
                # 총 처리 시간 계산
                if not error_occurred:
                    total_time = self.performance.get_total_elapsed()
                    await self.logger.log(
                        "info",
                        f"[{self.session_id}] 스트리밍 채팅 처리 완료: {total_time:.4f}초 소요",
                        session_id=self.session_id,
                        stages=self.performance.get_stages()
                    )

        # 스트리밍 응답 반환
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream; charset=utf-8"
        )

    async def _simple_response_to_stream(self, response: ChatResponse) -> StreamingResponse:
        """
        일반 응답을 스트리밍 형식으로 변환합니다.

        Args:
            response: 변환할 ChatResponse 객체

        Returns:
            StreamingResponse: 스트리밍 응답 객체
        """
        import json

        # 스트리밍 생성기
        async def simple_stream():
            # 응답 텍스트 청크
            json_str = json.dumps(
                {'text': response.chat.system, 'finished': False},
                ensure_ascii=False
            )
            yield f"data: {json_str}\n\n"

            # 완료 신호
            json_str = json.dumps(
                {'text': '', 'finished': True},
                ensure_ascii=False
            )
            yield f"data: {json_str}\n\n"

            # 전체 응답
            json_str = json.dumps(
                {'complete_response': response.chat.system},
                ensure_ascii=False
            )
            yield f"data: {json_str}\n\n"

        return StreamingResponse(
            simple_stream(),
            media_type="text/event-stream; charset=utf-8"
        )

    def _create_error_stream(self, error_message: str) -> StreamingResponse:
        """
        오류 메시지를 포함한 스트리밍 응답을 생성합니다.

        Args:
            error_message: 오류 메시지

        Returns:
            StreamingResponse: 스트리밍 오류 응답
        """
        import json

        async def error_stream():
            error_data = {
                'error': True,
                'text': error_message,
                'finished': True
            }
            json_str = json.dumps(error_data, ensure_ascii=False)
            yield f"data: {json_str}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream; charset=utf-8"
        )

    def _create_response(
            self,
            error_code: Dict[str, str],
            system_msg: str,
            document_payload: List = None
    ) -> ChatResponse:
        """
        ChatResponse 객체를 생성합니다.

        Args:
            error_code: 오류 코드 정보
            system_msg: 시스템 응답 메시지
            document_payload: 문서 페이로드 (기본값: None)

        Returns:
            ChatResponse: 생성된 응답 객체
        """
        try:
            # 페이로드 준비
            request_payload = document_payload or self.request.chat.payload or []

            if hasattr(self.request.chat, 'payload') and self.request.chat.payload:
                payloads = [
                    PayloadRes(doc_name=doc.doc_name, doc_page=doc.doc_page, content=doc.content)
                    for doc in self.request.chat.payload
                ]
            else:
                payloads = []

            # 응답 생성
            response = ChatResponse(
                result_cd=error_code.get("code"),
                result_desc=error_code.get("desc"),
                meta=MetaRes(
                    company_id=self.request.meta.company_id,
                    dept_class=self.request.meta.dept_class,
                    session_id=self.session_id,
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

            # 성능 데이터 추가
            if hasattr(response, 'add_performance_data'):
                response.add_performance_data(self.performance.get_metrics())

            return response

        except Exception as e:
            logger.error(f"[{self.session_id}] 응답 객체 생성 중 오류: {str(e)}", exc_info=True)

            # 오류 시 기본 응답 생성
            return ChatResponse(
                result_cd=ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION).get("code"),
                result_desc=ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION).get("desc"),
                meta=MetaRes(
                    company_id=self.request.meta.company_id if hasattr(self.request.meta, 'company_id') else "",
                    dept_class=self.request.meta.dept_class if hasattr(self.request.meta, 'dept_class') else "",
                    session_id=self.session_id if hasattr(self.request, 'session_id') else "",
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
