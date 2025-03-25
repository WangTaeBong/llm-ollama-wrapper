"""
채팅 서비스 모듈

LLM 기반 채팅 처리를 위한 핵심 서비스를 구현합니다. 이 모듈은 RAG(Retrieval-Augmented Generation) 시스템,
대화 관리, 응답 생성 및 스트리밍을 통합하여 고성능 채팅 인터페이스를 제공합니다.

주요 기능:
- 채팅 요청 처리 및 LLM 응답 생성
- 문서 검색 및 컨텍스트 증강
- 대화 이력 관리 및 컨텍스트 유지
- 응답 스트리밍 및 후처리
- 성능 모니터링 및 캐싱
"""

import asyncio
import json
import logging
import random
import re
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from cachetools import TTLCache
from datetime import datetime
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import BackgroundTasks
from starlette.responses import StreamingResponse

from src.common.config_loader import ConfigLoader
from src.common.error_cd import ErrorCd
from src.schema.chat_req import ChatRequest
from src.schema.chat_res import ChatResponse, MetaRes, PayloadRes, ChatRes

from src.services.llm.utils import (
    fire_and_forget,
    get_performance_timer,
    ContextLogger,
    get_today_formatted,
    truncate_text,
    sanitize_input
)

# 설정 로드
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class ChatService:
    """
    LLM 기반 채팅 서비스 클래스

    채팅 요청 처리, 문서 검색, LLM 응답 생성 및 후처리를 통합적으로 관리합니다.
    대화 이력 관리, 캐싱, 성능 모니터링, 스트리밍 응답을 지원합니다.

    Attributes:
        request (ChatRequest): 처리할 채팅 요청
        retriever_service: 문서 검색 서비스
        llm_service: LLM 상호작용 서비스
        history_handler: 대화 이력 관리 핸들러
        timer: 성능 측정 타이머
        cache_key: 응답 캐싱을 위한 키
    """

    # 클래스 수준 캐시
    _prompt_cache = TTLCache(maxsize=100, ttl=3600)  # 인사말/자주 묻는 질문용 캐시 (1시간)
    _response_cache = TTLCache(maxsize=200, ttl=900)  # 응답 캐시 (15분)

    # 캐시 정리 주기
    _cache_cleanup_interval = 3600  # 1시간마다 캐시 정리
    _last_cache_cleanup = time.time()

    # 비동기 로깅 큐
    _log_queue = asyncio.Queue()
    _log_task = None
    _log_initialized = False

    # 백그라운드 태스크 추적
    _background_tasks = []

    def __init__(self, request: ChatRequest):
        """
        채팅 서비스 초기화

        Args:
            request: 처리할 채팅 요청
        """
        self.request = request
        self.session_id = request.meta.session_id

        # 컨텍스트 로거 초기화
        self.context_logger = ContextLogger(self.session_id)

        # 서비스 컴포넌트 초기화
        from src.services.llm.factory import LLMServiceFactory
        factory = LLMServiceFactory.get_instance()

        # 의존성 주입
        self.factory = factory
        self.retriever_service = factory.create_retriever_service(request)
        self.llm_service = factory.create_llm_service(request)

        # 대화 이력 핸들러 초기화 (사용 가능한 LLM 모델로)
        from src.services.llm_ollama_process import mai_chat_llm
        self.history_handler = factory.create_handler(
            mai_chat_llm,
            self.request,
            max_history_turns=getattr(settings.chat_history, 'max_turns', 10)
        )

        # 처리 컴포넌트 초기화
        from src.common.query_check_dict import QueryCheckDict
        from src.services.response_generator import ResponseGenerator
        from src.services.query_processor import QueryProcessor
        from src.services.document_processor import DocumentProcessor
        from src.services.voc import VOCLinkProcessor
        from src.services.search_engine import SearchEngine
        from src.services.messaging.formatters import MessageFormatter

        self.query_check_dict = QueryCheckDict(settings.lm_check.query_dict_config_path)
        self.response_generator = ResponseGenerator(settings, self.query_check_dict)
        self.query_processor = QueryProcessor(settings, self.query_check_dict)
        self.document_processor = DocumentProcessor(settings)
        self.voc_processor = VOCLinkProcessor(settings)
        self.search_engine = SearchEngine(settings)
        self.formatter = MessageFormatter()

        # 성능 측정 타이머 설정
        self.timer = get_performance_timer()

        # 캐시 키 생성 (세션 및 쿼리 기반)
        self.cache_key = f"{self.request.meta.rag_sys_info}:{self.session_id}:{self.request.chat.user}"

        # 캐시 정리 확인
        self._check_cache_cleanup()

    @classmethod
    async def _ensure_log_task_running(cls):
        """
        비동기 로깅 태스크가 실행 중인지 확인하고, 필요시 시작합니다.
        """
        if not cls._log_initialized:
            cls._log_task = asyncio.create_task(cls._process_logs())
            cls._log_initialized = True
            logger.info("비동기 로깅 시스템 초기화됨")

    @classmethod
    async def _process_logs(cls):
        """
        로그 큐에서 로그 항목을 비동기적으로 처리합니다.

        무한 루프로 실행되며, 로그 항목을 큐에서 가져와 적절한 로거 메서드로 전달합니다.
        로깅 시스템이 정상적으로 계속 작동하도록 예외를 안전하게 처리합니다.
        """
        while True:
            try:
                # 큐에서 로그 항목 가져오기
                log_entry = await cls._log_queue.get()
                level, message, extra = log_entry

                # 'exc_info' 매개변수 추출 및 extra에서 제거
                exc_info = extra.pop('exc_info', False) if isinstance(extra, dict) else False

                # 적절한 수준으로 로깅
                if level == "debug":
                    logger.debug(message, exc_info=exc_info, extra=extra)
                elif level == "info":
                    logger.info(message, exc_info=exc_info, extra=extra)
                elif level == "warning":
                    logger.warning(message, exc_info=exc_info, extra=extra)
                elif level == "error":
                    logger.error(message, exc_info=exc_info, extra=extra)

                # 태스크 완료 표시
                cls._log_queue.task_done()
            except Exception as e:
                print(f"로그 처리 오류: {e}")
                await asyncio.sleep(1)  # 타이트 루프 방지를 위한 대기

    async def _log(self, level, message, **kwargs):
        """
        비동기 로깅 큐에 로그 항목을 추가합니다.

        Args:
            level (str): 로그 수준 ("debug", "info", "warning", "error")
            message (str): 로그 메시지
            **kwargs: 로거에 전달할 추가 매개변수
        """
        # 로깅 태스크 실행 확인
        await self._ensure_log_task_running()
        await ChatService._log_queue.put((level, message, kwargs))

    def _record_stage(self, stage_name):
        """
        처리 단계의 타이밍을 기록하고 경과 시간을 반환합니다.

        이 메서드는 마지막 단계 이후 경과한 시간을 계산하고,
        다음 단계를 위해 start_time을 재설정합니다.

        Args:
            stage_name (str): 처리 단계 이름

        Returns:
            float: 마지막 기록된 단계 이후 경과한 시간(초)
        """
        return self.timer["record_stage"](stage_name)

    async def process_chat(self) -> ChatResponse:
        """
        채팅 요청을 처리하고 적절한 응답을 생성합니다.

        이 메서드는 완전한 채팅 쿼리 처리 워크플로우를 구현합니다:
        - 인사말 감지 및 잘못된 쿼리 필터링
        - 검색 문서 검증 및 FAQ 쿼리 최적화
        - 문서 검색 및 언어 설정 결정
        - LLM을 사용한 응답 생성 (대화 이력 포함 여부에 따라)
        - 참조 및 형식 지정으로 응답 마무리

        Returns:
            ChatResponse: 결과, 메타데이터 및 컨텍스트가 포함된 응답 객체
        """
        # 로깅 태스크 초기화 확인
        await self._ensure_log_task_running()

        await self._log("info", f"채팅 요청 처리 시작", session_id=self.session_id)

        self.timer["reset"]()

        # 동일 요청에 대한 응답 캐시 확인
        if self.cache_key in ChatService._response_cache:
            await self._log("info", f"캐시된 응답 반환", session_id=self.session_id, cache_key=self.cache_key)
            return ChatService._response_cache[self.cache_key]

        try:
            # 인사말 처리 또는 잘못된 쿼리 필터링
            response = await self._handle_greeting_or_filter()
            if response:
                elapsed = self._record_stage("greeting_filter")
                await self._log("info", f"인사말/필터 단계 완료: {elapsed:.4f}초 소요", session_id=self.session_id)
                # 간단한 응답 캐싱
                ChatService._response_cache[self.cache_key] = response
                return response

            # 검색 문서 유효성 검사 (필터링 활성화된 경우)
            if settings.retriever.filter_flag:
                filter_res = self.document_processor.validate_retrieval_documents(self.request)
                if filter_res:
                    elapsed = self._record_stage("document_validation")
                    await self._log("info", f"문서 유효성 검사 완료: {elapsed:.4f}초 소요", session_id=self.session_id)
                    return filter_res

            # FAQ 쿼리 최적화 (해당하는 경우)
            faq_query = self._handle_faq_query()
            if faq_query:
                self.request.chat.user = faq_query
                elapsed = self._record_stage("faq_optimization")
                await self._log("info", f"FAQ 쿼리 최적화: {elapsed:.4f}초 소요", session_id=self.session_id)

            # 병렬 작업 시작
            # 문서 검색 및 언어 설정을 병렬로 처리
            retrieval_task = asyncio.create_task(self._retrieve_documents())
            language_task = asyncio.create_task(self._get_language_settings())

            # 작업 완료 대기
            await asyncio.gather(retrieval_task, language_task)

            # 결과 수집
            documents = retrieval_task.result()
            lang, trans_lang, reference_word = language_task.result()

            elapsed = self._record_stage("parallel_processing")
            await self._log(
                "info",
                f"병렬 처리 완료(준비): {elapsed:.4f}초 소요, {len(documents)}개 문서 검색됨",
                session_id=self.session_id
            )

            # 히스토리 설정 로깅
            await self._log("debug",
                            f"히스토리 설정: chat_history.enabled={settings.chat_history.enabled}, "
                            f"use_improved_history={getattr(settings.llm, 'use_improved_history', False)}")

            # 대화 이력 또는 직접 쿼리로 LLM 쿼리 처리
            start_llm_time = time.time()
            query_answer, vllm_retrival_document = await self._process_llm_query(documents, lang, trans_lang)

            llm_elapsed = time.time() - start_llm_time
            await self._log(
                "info",
                f"LLM 처리 완료: {llm_elapsed:.4f}초 소요 [backend={settings.llm.llm_backend}]",
                session_id=self.session_id
            )
            self._record_stage("llm_processing")

            # 역할 마커를 제거하여 생성된 응답 정리
            cleaned_answer = re.sub(r'(AI:|Human:)', '', query_answer).strip()

            # 대화 이력 저장 (활성화된 경우, 대기 없이 비동기적으로 실행)
            if settings.chat_history.enabled:
                await asyncio.create_task(self._save_chat_history(cleaned_answer))

                # 참조 및 형식 지정으로 응답 마무리
            start_finalize = time.time()
            final_query_answer = await self._finalize_answer_async(cleaned_answer, reference_word,
                                                                   vllm_retrival_document)

            finalize_elapsed = time.time() - start_finalize
            await self._log("debug", f"응답 마무리 완료: {finalize_elapsed:.4f}초 소요", session_id=self.session_id)
            self._record_stage("answer_finalization")

            # 최종 응답 생성 및 반환
            final_response = self._create_response(ErrorCd.get_error(ErrorCd.SUCCESS), final_query_answer)

            total_time = self.timer["get_total_time"]()
            await self._log(
                "info",
                f"채팅 처리 완료: {total_time:.4f}초 소요",
                session_id=self.session_id,
                stages=self.timer["get_stages"]()
            )

            # 상당한 시간이 소요되는 복잡한 쿼리에 대한 응답 캐싱
            if total_time > 5.0:  # 5초 이상 소요되는 쿼리만 캐싱
                ChatService._response_cache[self.cache_key] = final_response

            return final_response

        except ValueError as err:
            await self._log("error", f"채팅 처리 중 값 오류: {err}", session_id=self.session_id, exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "잘못된 입력이 제공되었습니다.")
        except KeyError as err:
            await self._log("error", f"채팅 처리 중 키 오류: {err}", session_id=self.session_id, exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "필수 키가 누락되었습니다.")
        except Exception as err:
            await self._log("error", f"채팅 처리 중 예상치 못한 오류: {err}", session_id=self.session_id, exc_info=True)
            return self._create_response(ErrorCd.get_error(ErrorCd.CHAT_EXCEPTION), "요청을 처리할 수 없습니다.")

