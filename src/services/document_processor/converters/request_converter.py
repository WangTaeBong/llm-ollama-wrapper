"""
요청-문서 변환 모듈

채팅 요청과 문서 객체 간의 변환 기능을 제공합니다.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from src.schema.chat_req import ChatRequest
from src.services.document_processor.base import DocumentConverterBase
from src.services.document_processor.exceptions import DocumentConversionError

# 로거 설정
logger = logging.getLogger(__name__)


class RequestToDocumentConverter(DocumentConverterBase[ChatRequest]):
    """
    요청을 Document 객체로 변환하는 클래스

    채팅 요청의 페이로드를 Document 객체로 변환하는 기능을 제공합니다.
    """

    def convert(self, request: ChatRequest) -> List[Document]:
        """
        채팅 요청의 페이로드를 Document 객체 목록으로 변환합니다.

        Args:
            request (ChatRequest): 페이로드를 포함한 채팅 요청

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        session_id = self._get_session_id(request)

        if not request.chat.payload:
            logger.debug(f"[{session_id}] 변환할 페이로드가 없습니다")
            return []

        try:
            documents = [
                Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.doc_name,
                        "doc_page": doc.doc_page
                    }
                )
                for doc in request.chat.payload if doc and doc.content
            ]

            logger.debug(f"[{session_id}] {len(documents)}개의 문서로 변환 완료")
            return documents

        except Exception as e:
            logger.warning(f"[{session_id}] 페이로드에서 문서 변환 실패: {e}")
            raise DocumentConversionError(f"페이로드 변환 실패: {str(e)}")

    @staticmethod
    def _get_session_id(request: Optional[ChatRequest]) -> str:
        """
        요청에서 세션 ID를 안전하게 추출합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 세션 ID 또는 기본값
        """
        if request and hasattr(request, 'meta') and hasattr(request.meta, 'session_id'):
            return request.meta.session_id
        return "unknown"
