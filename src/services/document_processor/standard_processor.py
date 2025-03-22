"""
표준 문서 프로세서 모듈

표준 문서 처리 기능을 제공합니다.
"""

import logging
from typing import List, Optional, Any

from langchain_core.documents import Document
from src.schema.chat_req import ChatRequest, PayloadReq
from src.schema.chat_res import ChatResponse
from src.services.document_processor.base import DocumentProcessorBase
from src.services.document_processor.converters import RequestToDocumentConverter, DocumentToPayloadConverter
from src.services.document_processor.validators import DocumentValidator

# 로거 설정
logger = logging.getLogger(__name__)


class StandardDocumentProcessor(DocumentProcessorBase):
    """
    표준 문서 프로세서 클래스

    문서 변환 및 유효성 검증을 위한 기본 구현을 제공합니다.
    """

    def __init__(self, settings: Any):
        """
        표준 문서 프로세서 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings

        # 컴포넌트 초기화
        self._request_converter = RequestToDocumentConverter()
        self._document_converter = DocumentToPayloadConverter()
        self._document_validator = DocumentValidator(settings)

        logger.debug("표준 문서 프로세서가 초기화되었습니다")

    def convert_payload_to_document(self, request: ChatRequest) -> List[Document]:
        """
        채팅 요청의 페이로드를 Document 객체 목록으로 변환합니다.

        Args:
            request (ChatRequest): 페이로드를 포함한 채팅 요청

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        return self._request_converter.convert(request)

    def convert_document_to_payload(self, documents: List[Document]) -> List[PayloadReq]:
        """
        Document 객체 목록을 페이로드 객체 목록으로 변환합니다.

        Args:
            documents (List[Document]): 변환할 Document 객체 목록

        Returns:
            List[PayloadReq]: 변환된 페이로드 객체 목록
        """
        return self._document_converter.convert(documents)

    def validate_retrieval_documents(self, request: ChatRequest) -> Optional[ChatResponse]:
        """
        채팅 요청의 검색 문서 유효성을 검증합니다.

        Args:
            request (ChatRequest): 검증할 채팅 요청

        Returns:
            Optional[ChatResponse]: 유효성 검증 실패 시 응답, 성공 시 None
        """
        # 필터링이 활성화된 경우에만 유효성 검증 수행
        if getattr(self.settings, 'retriever', None) and getattr(self.settings.retriever, 'filter_flag', False):
            return self._document_validator.validate(request)
        return None
    