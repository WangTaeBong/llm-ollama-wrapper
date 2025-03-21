"""
LLM 서비스 기본 인터페이스 모듈
==============================

이 모듈은 다양한 LLM(Large Language Model) 백엔드에 대한 표준 인터페이스를 정의합니다.
모든 LLM 서비스 구현체는 이 기본 클래스를 상속받아 구현해야 합니다.

기능:
- 동기 및 비동기 LLM 호출 인터페이스
- 스트리밍 응답 처리
- 문서 컨텍스트 처리
- 모델 파라미터 관리
"""

import abc
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# 로거 설정
logger = logging.getLogger(__name__)


class LLMServiceBase(abc.ABC):
    """
    LLM 서비스 기본 인터페이스

    모든 LLM 백엔드 구현은 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, settings: Any):
        """
        LLM 서비스 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings
        self.model = None
        self.session_id = None

        # 메트릭 초기화
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0,
            "token_count": 0
        }

    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        LLM 모델 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        pass

    @abc.abstractmethod
    async def ask(self,
                  query: str,
                  documents: List[Document],
                  language: str,
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        LLM에 질의하고 응답 반환

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            str: LLM 응답 텍스트
        """
        pass

    @abc.abstractmethod
    async def stream_response(self,
                              query: str,
                              documents: List[Document],
                              language: str,
                              context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 모드로 LLM에 질의하고 응답 생성

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            AsyncGenerator: 응답 청크를 생성하는 비동기 제너레이터
        """
        pass

    @abc.abstractmethod
    def build_system_prompt(self,
                            template: Union[str, PromptTemplate],
                            context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트 생성

        Args:
            template: 프롬프트 템플릿
            context: 프롬프트 컨텍스트

        Returns:
            str: 완성된 시스템 프롬프트
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        서비스 메트릭 반환

        Returns:
            Dict[str, Any]: 메트릭 정보
        """
        avg_time = 0
        if self.metrics["request_count"] > 0:
            avg_time = self.metrics["total_time"] / self.metrics["request_count"]

        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "avg_response_time": avg_time,
            "total_time": self.metrics["total_time"],
            "token_count": self.metrics.get("token_count", 0)
        }

    @staticmethod
    def format_documents(documents: List[Document]) -> str:
        """
        문서 리스트를 LLM 프롬프트에 적합한 형식으로 변환

        Args:
            documents: Document 객체 리스트

        Returns:
            str: 형식화된 문서 문자열
        """
        if not documents:
            return ""

        formatted_docs = []
        for i, doc in enumerate(documents):
            if not hasattr(doc, 'page_content') or not doc.page_content:
                continue

            source = doc.metadata.get("source", "알 수 없는 출처")
            page = doc.metadata.get("doc_page", "")

            doc_heading = f"[문서 {i + 1}] 출처: {source}"
            if page:
                doc_heading += f", 페이지: {page}"

            formatted_docs.append(f"{doc_heading}\n{doc.page_content}\n")

        return "\n".join(formatted_docs)

    @classmethod
    def is_valid_backend(cls, backend_name: str) -> bool:
        """
        지원되는 백엔드인지 확인

        Args:
            backend_name: 백엔드 이름

        Returns:
            bool: 지원 여부
        """
        return backend_name.lower() in ["ollama", "vllm"]
