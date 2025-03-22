"""
문서-페이로드 변환 모듈

Document 객체와 페이로드 객체 간의 변환 기능을 제공합니다.
"""

import logging
from typing import List

from langchain_core.documents import Document
from src.schema.chat_req import PayloadReq
from src.services.document_processor.base import DocumentConverterBase
from src.services.document_processor.exceptions import DocumentConversionError

# 로거 설정
logger = logging.getLogger(__name__)


class DocumentToPayloadConverter(DocumentConverterBase[List[Document]]):
    """
    Document 객체를 페이로드 객체로 변환하는 클래스

    Document 객체 목록을 페이로드 객체 목록으로 변환하는 기능을 제공합니다.
    """

    @classmethod
    def convert(cls, documents: List[Document]) -> List[PayloadReq]:
        """
        Document 객체 목록을 페이로드 객체 목록으로 변환합니다.

        Args:
            documents (List[Document]): 변환할 Document 객체 목록

        Returns:
            List[PayloadReq]: 변환된 페이로드 객체 목록
        """
        if not documents:
            return []

        try:
            payload_list = [
                PayloadReq(
                    doc_name=doc.metadata.get("source", ""),  # 올바른 키 "source" 사용
                    doc_page=doc.metadata.get("doc_page", ""),
                    content=doc.page_content
                )
                for doc in documents if doc
            ]

            logger.debug(f"{len(payload_list)}개의 페이로드로 변환 완료")
            return payload_list

        except Exception as e:
            logger.warning(f"문서에서 페이로드 변환 실패: {e}")
            raise DocumentConversionError(f"문서 변환 실패: {str(e)}")
