"""
대화 히스토리 관리 기본 인터페이스 모듈

모든 히스토리 핸들러가 구현해야 할 기본 인터페이스와 추상 클래스를 제공합니다.
"""

import abc
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory

from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery


class HistoryStorageBase(abc.ABC):
    """
    히스토리 저장소 인터페이스

    대화 히스토리 저장 및 검색을 위한 기본 인터페이스입니다.
    """

    @abc.abstractmethod
    async def save_message(self, system_info: str, session_id: str, message: Dict[str, Any]) -> bool:
        """
        메시지를 저장소에 저장합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID
            message: 저장할 메시지 데이터

        Returns:
            bool: 저장 성공 여부
        """
        pass

    @abc.abstractmethod
    def get_messages(self, system_info: str, session_id: str) -> List[Dict[str, Any]]:
        """
        저장소에서 메시지를 검색합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            List[Dict[str, Any]]: 저장된 메시지 목록
        """
        pass

    @abc.abstractmethod
    async def clear_messages(self, system_info: str, session_id: str) -> bool:
        """
        지정된 세션의 메시지를 모두 삭제합니다.

        Args:
            system_info: 시스템 정보
            session_id: 세션 ID

        Returns:
            bool: 삭제 성공 여부
        """
        pass


class PromptFormatterBase(abc.ABC):
    """
    프롬프트 포맷터 인터페이스

    다양한 모델 유형에 맞게 프롬프트를 형식화하는 인터페이스입니다.
    """

    @abc.abstractmethod
    def format_history_for_prompt(self, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        대화 이력을 프롬프트 형식으로 변환합니다.

        Args:
            session_history: 대화 메시지 이력
            max_turns: 포함할 최대 대화 턴 수

        Returns:
            str: 형식화된 대화 이력 문자열
        """
        pass

    @abc.abstractmethod
    def build_system_prompt(self, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트를 생성합니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 컨텍스트 변수

        Returns:
            str: 생성된 시스템 프롬프트
        """
        pass


class HistoryHandlerBase(abc.ABC):
    """
    히스토리 핸들러 기본 인터페이스

    대화 이력 관리 및 처리를 위한 기본 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    async def init_retriever(self, retrieval_documents: List[Document]) -> Any:
        """
        검색기를 초기화합니다.

        Args:
            retrieval_documents: 초기 문서 목록

        Returns:
            Any: 초기화된 검색기
        """
        pass

    @abc.abstractmethod
    def get_session_history(self) -> ChatMessageHistory:
        """
        세션 히스토리를 가져옵니다.

        Returns:
            ChatMessageHistory: 세션 히스토리
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    async def handle_chat_with_history_vllm(self, request: ChatRequest, language: str) -> Tuple[str, List[Document]]:
        """
        vLLM을 사용하여 히스토리 기반 채팅 요청을 처리합니다.

        Args:
            request: 채팅 요청
            language: 응답 언어

        Returns:
            Tuple[str, List[Document]]: (응답 텍스트, 검색된 문서 목록)
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractmethod
    async def save_chat_history(self, answer: str) -> bool:
        """
        채팅 히스토리를 저장합니다.

        Args:
            answer: 응답 텍스트

        Returns:
            bool: 저장 성공 여부
        """
        pass

    @abc.abstractmethod
    def cleanup_processed_sets(self):
        """
        처리된 세트를 정리하여 메모리 누수를 방지합니다.
        """
        pass
