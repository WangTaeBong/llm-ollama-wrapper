"""
문서 처리 패키지

문서 변환, 유효성 검증 및 처리를 위한 기능을 제공합니다.
"""

from src.services.document_processor.factory import DocumentProcessorFactory
from src.services.document_processor.exceptions import DocumentProcessorError
from src.services.document_processor.standard_processor import StandardDocumentProcessor


# 기존 DocumentProcessor 클래스와의 호환성을 위한 인터페이스
class DocumentProcessor:
    """
    문서 프로세서 - 기존 API와의 호환성 제공

    기존 DocumentProcessor 클래스의 API를 유지하면서 내부적으로
    새로운 아키텍처를 사용합니다.
    """

    def __init__(self, settings):
        """
        문서 프로세서 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings
        self._processor = DocumentProcessorFactory.create("standard", settings)

    def convert_payload_to_document(self, request):
        """
        채팅 요청의 페이로드를 Document 객체 목록으로 변환합니다.

        Args:
            request: 페이로드를 포함한 채팅 요청

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        return self._processor.convert_payload_to_document(request)

    def convert_document_to_payload(self, documents):
        """
        Document 객체 목록을 페이로드 객체 목록으로 변환합니다.

        Args:
            documents: 변환할 Document 객체 목록

        Returns:
            List[PayloadReq]: 변환된 페이로드 객체 목록
        """
        return self._processor.convert_document_to_payload(documents)

    def validate_retrieval_documents(self, request):
        """
        채팅 요청의 검색 문서 유효성을 검증합니다.

        Args:
            request: 검증할 채팅 요청

        Returns:
            Optional[ChatResponse]: 유효성 검증 실패 시 응답, 성공 시 None
        """
        return self._processor.validate_retrieval_documents(request)


__all__ = [
    'DocumentProcessor',
    'DocumentProcessorFactory',
    'DocumentProcessorError',
    'StandardDocumentProcessor'
]
