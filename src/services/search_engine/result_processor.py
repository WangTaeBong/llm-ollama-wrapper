"""
검색 결과 처리 모듈

검색 결과를 처리하고 Document 객체로 변환하는 기능을 제공합니다.
"""

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document

# 로거 설정
logger = logging.getLogger(__name__)


class ResultProcessor:
    """
    검색 결과 처리 클래스

    검색 결과를 Document 객체로 변환하고 필터링하는 기능을 제공합니다.
    """

    @classmethod
    def convert_to_documents(cls, results: List[Dict[str, Any]]) -> List[Document]:
        """
        검색 결과를 Document 객체 목록으로 변환합니다.

        Args:
            results: 검색 결과 목록

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        documents = []
        invalid_count = 0

        for result in results:
            # 결과 유효성 검사
            if not result or not isinstance(result, dict):
                invalid_count += 1
                continue

            # 필수 필드 확인
            body = result.get('body', '')
            if not body:
                logger.debug("내용이 없는 검색 결과를 건너뜁니다")
                continue

            # Document 객체 생성
            doc = Document(
                page_content=body,
                metadata={
                    'source': result.get('title', 'Unknown'),
                    'doc_page': result.get('href', '#'),
                }
            )
            documents.append(doc)

        # 유효하지 않은 결과 로깅
        if invalid_count > 0:
            logger.debug(f"{invalid_count}개의 유효하지 않은 검색 결과를 건너뜁니다")

        return documents

    @classmethod
    def filter_documents(cls, documents: List[Document], min_content_length: int = 10) -> List[Document]:
        """
        Document 객체 목록을 필터링합니다.

        Args:
            documents: 필터링할 Document 객체 목록
            min_content_length: 최소 내용 길이

        Returns:
            List[Document]: 필터링된 Document 객체 목록
        """
        filtered = [
            doc for doc in documents
            if doc.page_content and len(doc.page_content) >= min_content_length
        ]

        filtered_count = len(documents) - len(filtered)
        if filtered_count > 0:
            logger.debug(f"{filtered_count}개의 짧은 내용을 가진 문서를 필터링했습니다")

        return filtered
