"""
문서 저장소 모듈

Document 객체의 저장 및 중복 제거를 관리합니다.
"""

import hashlib
import logging
from typing import Dict, List, Set, Optional, DefaultDict
from collections import defaultdict

from langchain_core.documents import Document

from src.services.retrieval.base import DocumentStoreBase

# 로거 설정
logger = logging.getLogger(__name__)


class DocumentStore(DocumentStoreBase):
    """
    문서 저장소 클래스

    Document 객체의 효율적인 저장 및 중복 제거 기능을 제공합니다.
    """

    def __init__(self):
        """문서 저장소 초기화"""
        self._documents: DefaultDict[str, List[Document]] = defaultdict(list)
        self._document_keys: Set[str] = set()

    def add(self, document: Document) -> bool:
        """
        단일 문서 추가

        Args:
            document: 추가할 Document 객체

        Returns:
            bool: 추가 성공 시 True, 중복인 경우 False
        """
        doc_key = self._create_document_key(document)

        # 중복 확인
        if doc_key in self._document_keys:
            return False

        # 문서 추가
        doc_name = document.metadata.get("doc_name", "default")
        self._documents[doc_name].append(document)
        self._document_keys.add(doc_key)
        return True

    def add_batch(self, documents: List[Document]) -> int:
        """
        문서 배치 추가

        Args:
            documents: 추가할 Document 객체 목록

        Returns:
            int: 성공적으로 추가된 문서 수
        """
        added_count = 0
        for doc in documents:
            if self.add(doc):
                added_count += 1
        return added_count

    def get_all(self) -> List[Document]:
        """
        모든 저장된 문서 가져오기

        Returns:
            List[Document]: 저장된 모든 Document 객체
        """
        return [doc for docs in self._documents.values() for doc in docs]

    def get_by_doc_name(self, doc_name: str) -> List[Document]:
        """
        이름으로 문서 가져오기

        Args:
            doc_name: 문서 이름

        Returns:
            List[Document]: 지정된 이름의 문서 목록
        """
        return self._documents.get(doc_name, [])

    def clear(self) -> None:
        """저장소의 모든 문서를 지웁니다."""
        self._documents.clear()
        self._document_keys.clear()

    @classmethod
    def _create_document_key(cls, document: Document) -> str:
        """
        문서의 고유 키 생성

        Args:
            document: Document 객체

        Returns:
            str: 고유 문서 키
        """
        return f"{document.metadata.get('doc_name', 'default')}_{document.page_content}"
