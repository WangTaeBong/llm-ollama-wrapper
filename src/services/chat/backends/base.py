"""
LLM 백엔드 기본 인터페이스 모듈

다양한 LLM 백엔드를 위한 공통 인터페이스를 정의합니다.
이 인터페이스를 구현하는 구체적인 백엔드는 Ollama, vLLM 등입니다.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, AsyncGenerator, Optional, Tuple

from langchain_core.documents import Document
from src.schema.chat_req import ChatRequest
from src.schema.vllm_inquery import VllmInquery


class LLMBackend(ABC):
    """
    LLM 백엔드 추상 인터페이스

    모든 LLM 백엔드 구현체가 준수해야 하는 인터페이스를 정의합니다.
    공통 메서드와 동작을 지정하고 구현체 간 교체 가능성을 제공합니다.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """
        백엔드 이름 반환

        Returns:
            str: 백엔드 식별자 이름
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        모델 이름 반환

        Returns:
            str: 사용 중인 모델 이름
        """
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """
        백엔드 초기화

        LLM 모델과 필요한 리소스를 초기화합니다.

        Returns:
            bool: 초기화 성공 여부
        """
        pass

    @abstractmethod
    async def generate(
            self,
            request: ChatRequest,
            language: str,
            documents: List[Document] = None
    ) -> Dict[str, Any]:
        """
        텍스트 생성 요청 처리

        Args:
            request: 채팅 요청
            language: 응답 언어
            documents: 컨텍스트 문서 (기본값: None)

        Returns:
            Dict[str, Any]: 응답 데이터
            {
                "answer": str,  # 생성된 텍스트
                "documents": List[Document],  # 검색에 사용된 문서
                "metadata": Dict[str, Any]  # 추가 메타데이터
            }
        """
        pass

    @abstractmethod
    async def stream_generate(
            self,
            request: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 방식으로 텍스트 생성

        Args:
            request: 스트리밍 요청 (VllmInquery 등)

        Yields:
            Dict[str, Any]: 청크 데이터
            {
                "text": str,  # 텍스트 청크
                "finished": bool,  # 완료 여부
                "error": bool,  # 오류 발생 여부 (선택적)
                "message": str  # 오류 메시지 (오류 시)
            }
        """
        pass

    @abstractmethod
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
            {
                "answer": str,  # 생성된 텍스트
                "documents": List[Document],  # 검색에 사용된 문서
                "metadata": Dict[str, Any]  # 추가 메타데이터
            }
        """
        pass

    @abstractmethod
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
            Any: 스트리밍 요청 객체 (백엔드에 따라 다름)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        백엔드 가용성 확인

        Returns:
            bool: 백엔드가 사용 가능하면 True
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        성능 메트릭 조회

        Returns:
            Dict[str, Any]: 성능 메트릭
        """
        pass
