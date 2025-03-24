"""
LLM 히스토리 핸들러 래퍼 모듈

기존 코드와의 호환성을 위한 래퍼 클래스를 제공합니다.
내부적으로 리팩토링된 히스토리 핸들러 모듈을 사용합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document

from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery
from src.services.history.factory import HistoryHandlerFactory
from src.services.history.base import HistoryHandlerBase

# 로거 설정
logger = logging.getLogger(__name__)


class LlmHistoryHandler:
    """
    LLM 히스토리 핸들러 클래스 (래퍼)

    기존 코드와의 호환성을 위한 래퍼 클래스입니다.
    내부적으로 새로운, 모듈화된 히스토리 핸들러 구현을 사용합니다.
    """

    # 클래스 레벨 변수 (기존 코드 호환성)
    session_cache = {}
    session_chain_cache = {}

    def __init__(self, llm_model: Any, request: ChatRequest, max_history_turns: int = 10):
        """
        LlmHistoryHandler 초기화

        Args:
            llm_model: LLM 모델 인스턴스
            request: 채팅 요청 객체
            max_history_turns: 최대 히스토리 턴 수
        """
        logger.debug(f"[{request.meta.session_id}] LlmHistoryHandler 래퍼 초기화")

        # 핸들러 팩토리를 통해 적절한 핸들러 생성
        self._handler = HistoryHandlerFactory.create_handler(
            llm_model=llm_model,
            request=request,
            max_history_turns=max_history_turns
        )

        # 래퍼 속성 (기존 코드 호환성)
        self.llm_model = llm_model
        self.current_session_id = request.meta.session_id
        self.current_rag_sys_info = request.meta.rag_sys_info
        self.request = request
        self.max_history_turns = max_history_turns
        self.processed_inputs = set()
        self.processed_message_ids = set()
        self.retriever = None
        self.response_stats = None

    @classmethod
    def is_gemma_model(cls) -> bool:
        """
        현재 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        return HistoryHandlerFactory._is_gemma_model()

    async def init_retriever(self, retrieval_documents: List[Document]) -> Any:
        """
        검색기를 초기화합니다.

        Args:
            retrieval_documents: 초기 문서 목록

        Returns:
            Any: 초기화된 검색기
        """
        retriever = await self._handler.init_retriever(retrieval_documents)
        self.retriever = retriever  # 래퍼의 retriever 속성 설정 (호환성 유지)
        return retriever

    def init_chat_chain_with_history(self):
        """
        히스토리가 포함된 채팅 체인을 초기화하거나 캐시에서 가져옵니다.

        이 메서드는 내부 핸들러의 init_chat_chain_with_history 메서드를 호출합니다.
        내부 핸들러가 OllamaHistoryHandler인 경우에만 작동합니다.

        Returns:
            Any: 초기화된 채팅 체인 또는 None (핸들러가 이 기능을 지원하지 않는 경우)
        """
        # 내부 핸들러 존재 확인
        if not hasattr(self, '_handler'):
            logger.error(f"[{self.current_session_id}] 내부 핸들러가 초기화되지 않았습니다")
            return None

        # 명시적으로 OllamaHistoryHandler 타입 확인
        from src.services.history.handlers.ollama_handler import OllamaHistoryHandler
        if isinstance(self._handler, OllamaHistoryHandler):
            logger.debug(f"[{self.current_session_id}] OllamaHistoryHandler에서 체인 초기화 실행")
            return self._handler.init_chat_chain_with_history()
        else:
            logger.warning(f"[{self.current_session_id}] 핸들러 타입 ({type(self._handler).__name__})은 초기화 기능을 지원하지 않습니다")
            return None

    def get_session_history(self) -> ChatMessageHistory:
        """
        세션 히스토리를 가져옵니다.

        Returns:
            ChatMessageHistory: 세션 히스토리
        """
        return self._handler.get_session_history()

    async def handle_chat_with_history(self,
                                       request: ChatRequest,
                                       language: str,
                                       rag_chat_chain: Any) -> Optional[Dict[str, Any]]:
        """
        히스토리를 활용하여 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어
            rag_chat_chain: RAG 채팅 체인

        Returns:
            Optional[Dict[str, Any]]: 채팅 응답 또는 None
        """
        logger.error("handle_chat_with_history-llm_history_handler")
        return await self._handler.handle_chat_with_history(request, language, rag_chat_chain)

    async def handle_chat_with_history_vllm(self, request: ChatRequest, language: str) -> Tuple[str, List[Document]]:
        """
        vLLM을 사용하여 히스토리 기반 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        # 모델 유형에 따라 적절한 핸들러 분기
        if self.is_gemma_model():
            logger.info(f"[{self.current_session_id}] Gemma 모델 감지됨, Gemma 전용 핸들러로 처리")
            return await self._handler.handle_chat_with_history_vllm(request, language)
        else:
            # 개선된 히스토리 처리 확인
            if getattr(settings.llm, 'use_improved_history', False):
                return await self._handler.handle_chat_with_history_vllm(request, language)
            else:
                return await self._handler.handle_chat_with_history_vllm(request, language)

    async def handle_chat_with_history_vllm_streaming(self,
                                                      request: ChatRequest,
                                                      language: str) -> Tuple[VllmInquery, List[Document]]:
        """
        vLLM을 사용하여 스트리밍 모드로 히스토리 기반 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        return await self._handler.handle_chat_with_history_vllm_streaming(request, language)

    async def handle_chat_with_history_vllm_streaming_improved(self,
                                                               request: ChatRequest,
                                                               language: str) -> Tuple[VllmInquery, List[Document]]:
        """
        개선된 2단계 접근법으로 히스토리 기반 스트리밍 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        return await self._handler.handle_chat_with_history_vllm_streaming(request, language)

    async def handle_chat_with_history_gemma(self, request: ChatRequest, language: str) -> Tuple[str, List[Document]]:
        """
        Gemma 모델을 위한 히스토리 기반 채팅 요청 처리

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        return await self._handler.handle_chat_with_history_vllm(request, language)

    async def handle_chat_with_history_gemma_streaming(self,
                                                       request: ChatRequest,
                                                       language: str) -> Tuple[VllmInquery, List[Document]]:
        """
        Gemma 모델을 위한 히스토리 기반 스트리밍 요청 처리

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[VllmInquery, List[Document]]: (vLLM 요청 객체, 검색된 문서 목록)
        """
        return await self._handler.handle_chat_with_history_vllm_streaming(request, language)

    async def save_chat_history(self, answer: str) -> bool:
        """
        채팅 히스토리를 저장합니다.

        Args:
            answer: 응답 텍스트

        Returns:
            bool: 저장 성공 여부
        """
        return await self._handler.save_chat_history(answer)

    def cleanup_processed_sets(self):
        """
        처리된 세트를 정리하여 메모리 누수를 방지합니다.
        """
        self._handler.cleanup_processed_sets()

    async def call_vllm_endpoint(self, data: VllmInquery):
        """
        vLLM 엔드포인트를 호출합니다.

        Args:
            data: vLLM 요청 데이터

        Returns:
            dict: vLLM 응답
        """
        return await self._handler.call_vllm_endpoint(data)

    @classmethod
    def format_history_for_prompt(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        대화 이력을 프롬프트 형식으로 변환합니다.

        Args:
            session_history: 대화 메시지 이력
            max_turns: 포함할 최대 대화 턴 수

        Returns:
            str: 형식화된 대화 이력 문자열
        """
        from src.services.history.formatters.prompt_formatter import StandardPromptFormatter
        formatter = StandardPromptFormatter()
        return formatter.format_history_for_prompt(session_history, max_turns)

    @classmethod
    def format_history_for_gemma(cls, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        Gemma 모델에 적합한 형식으로 대화 이력을 구성합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수

        Returns:
            str: Gemma 형식의 대화 이력 문자열
        """
        from src.services.history.formatters.gemma_formatter import GemmaPromptFormatter
        formatter = GemmaPromptFormatter()
        return formatter.format_history_for_prompt(session_history, max_turns)

    @classmethod
    def build_system_prompt(cls, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트를 생성합니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 컨텍스트 변수

        Returns:
            str: 생성된 시스템 프롬프트
        """
        from src.services.history.formatters.prompt_formatter import StandardPromptFormatter
        formatter = StandardPromptFormatter()
        return formatter.build_system_prompt(system_prompt_template, context)

    @classmethod
    def build_system_prompt_gemma(cls, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성합니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        from src.services.history.formatters.gemma_formatter import GemmaPromptFormatter
        formatter = GemmaPromptFormatter()
        return formatter.build_system_prompt(system_prompt_template, context)

    @classmethod
    def build_system_prompt_improved(cls, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        개선된 시스템 프롬프트 빌드 메소드.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: 형식화된 시스템 프롬프트
        """
        from src.services.history.formatters.prompt_formatter import StandardPromptFormatter
        formatter = StandardPromptFormatter()
        return formatter.build_system_prompt_improved(system_prompt_template, context)
