"""
문서 처리를 위한 기본 인터페이스 모듈

문서 처리 구현체가 준수해야 하는 인터페이스와 추상 클래스를 제공합니다.
"""

import abc
from typing import List, Optional, TypeVar, Generic

from langchain_core.documents import Document
from src.schema.chat_req import ChatRequest, PayloadReq
from src.schema.chat_res import ChatResponse

T = TypeVar('T')


class DocumentConverterBase(Generic[T], abc.ABC):
    """
    문서 변환 인터페이스

    문서와 다른 형식 간의 변환을 처리하는 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    def convert(self, source: T) -> List[Document]:
        """
        소스 데이터를 Document 객체 목록으로 변환합니다.

        Args:
            source: 변환할 소스 데이터

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        pass


class DocumentValidatorBase(abc.ABC):
    """
    문서 유효성 검증 인터페이스

    문서의 유효성을 검증하고 대응하는 인터페이스를 정의합니다.
    """

    @abc.abstractmethod
    def validate(self, request: ChatRequest) -> Optional[ChatResponse]:
        """
        요청의 문서를 검증하고 필요한 경우 오류 응답을 반환합니다.

        Args:
            request (ChatRequest): 검증할 채팅 요청

        Returns:
            Optional[ChatResponse]: 유효성 검증 실패 시 응답, 성공 시 None
        """
        pass


class DocumentProcessorBase(abc.ABC):
    """
    문서 프로세서 기본 인터페이스

    모든 문서 프로세서 구현체가 준수해야 하는 기본 인터페이스입니다.
    """

    @abc.abstractmethod
    def convert_payload_to_document(self, request: ChatRequest) -> List[Document]:
        """
        채팅 요청의 페이로드를 Document 객체 목록으로 변환합니다.

        Args:
            request (ChatRequest): 페이로드를 포함한 채팅 요청

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        pass

    @abc.abstractmethod
    def convert_document_to_payload(self, documents: List[Document]) -> List[PayloadReq]:
        """
        Document 객체 목록을 페이로드 객체 목록으로 변환합니다.

        Args:
            documents (List[Document]): 변환할 Document 객체 목록

        Returns:
            List[PayloadReq]: 변환된 페이로드 객체 목록
        """
        pass

    @abc.abstractmethod
    def validate_retrieval_documents(self, request: ChatRequest) -> Optional[ChatResponse]:
        """
        채팅 요청의 검색 문서 유효성을 검증합니다.

        Args:
            request (ChatRequest): 검증할 채팅 요청

        Returns:
            Optional[ChatResponse]: 유효성 검증 실패 시 응답, 성공 시 None
        """
        pass
