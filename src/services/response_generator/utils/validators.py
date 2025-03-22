# src/services/response_generator/utils/validators.py
"""
유효성 검사 유틸리티 모듈

데이터 유효성 검사를 위한 도구와 유틸리티를 제공합니다.
"""

import logging
import re
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class DocumentValidator:
    """
    문서 유효성 검사 클래스

    문서 콘텐츠 및 메타데이터의 유효성을 검사합니다.
    """

    def __init__(self):
        """
        DocumentValidator 초기화
        """
        # 유효성 검사를 위한 정규식 패턴
        self._patterns = {
            'url': re.compile(
                r'https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9@:%_+.~#?&/=]*'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'whitespace': re.compile(r'\s+')
        }

    def validate_documents(self, documents: List[Document]) -> List[Document]:
        """
        문서 목록의 유효성을 검사하고 필터링합니다.

        Args:
            documents: 검사할 문서 목록

        Returns:
            List[Document]: 유효한 문서 목록
        """
        if not documents:
            return []

        valid_docs = []
        for doc in documents:
            if self.is_valid_document(doc):
                valid_docs.append(doc)

        if len(valid_docs) < len(documents):
            logger.debug(f"{len(documents) - len(valid_docs)}개의 유효하지 않은 문서가 필터링되었습니다.")

        return valid_docs

    @classmethod
    def is_valid_document(cls, document: Document) -> bool:
        """
        단일 문서의 유효성을 검사합니다.

        Args:
            document: 검사할 문서

        Returns:
            bool: 문서가 유효하면 True, 아니면 False
        """
        try:
            # 콘텐츠 검사
            if not document.page_content or len(document.page_content.strip()) < 10:
                return False

            # 필수 메타데이터 검사
            if not document.metadata:
                return False

            source = document.metadata.get("source") or document.metadata.get("doc_name")
            if not source:
                return False

            return True
        except Exception as e:
            logger.warning(f"문서 유효성 검사 중 오류: {e}")
            return False

    @classmethod
    def clean_document_content(cls, content: str) -> str:
        """
        문서 콘텐츠를 정리합니다.

        공백 정규화, 특수 문자 제거 등을 수행합니다.

        Args:
            content: 정리할 문서 콘텐츠

        Returns:
            str: 정리된 콘텐츠
        """
        if not content:
            return ""

        try:
            # 공백 정규화
            content = self._patterns['whitespace'].sub(' ', content)
            # 앞뒤 공백 제거
            content = content.strip()
            # 줄 바꿈 문자 통합
            content = content.replace("\r\n", "\n")

            return content
        except Exception as e:
            logger.warning(f"콘텐츠 정리 중 오류: {e}")
            return content

    @classmethod
    def sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        문서 메타데이터를 정리합니다.

        Args:
            metadata: 정리할 메타데이터

        Returns:
            Dict[str, Any]: 정리된 메타데이터
        """
        if not metadata:
            return {}

        try:
            sanitized = {}
            for key, value in metadata.items():
                # 문자열 값을 정리
                if isinstance(value, str):
                    sanitized[key] = value.strip()
                else:
                    sanitized[key] = value

            return sanitized
        except Exception as e:
            logger.warning(f"메타데이터 정리 중 오류: {e}")
            return metadata.copy() if metadata else {}
