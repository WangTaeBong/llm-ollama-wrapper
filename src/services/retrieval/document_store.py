"""
문서 저장소 모듈
=============

검색 결과 문서를 효율적으로 저장하고 관리하는 기능을 제공합니다.

기능:
- 문서 저장 및 검색
- 중복 제거 및 정렬
- 메타데이터 관리
"""

import hashlib
import logging
from collections import defaultdict
from typing import Dict, List, Set

from langchain_core.documents import Document

# 로거 설정
logger = logging.getLogger(__name__)


class DocumentStore:
    """
    문서 저장소 클래스

    검색된 문서를 저장하고 효율적으로 관리합니다.
    """

    def __init__(self):
        """문서 저장소 초기화"""
        self._documents: Dict[str, List[Document]] = defaultdict(list)
        self._document_keys: Set[str] = set()

    def add(self, document: Document) -> bool:
        """
        단일 문서 추가

        Args:
            document: 추가할 Document 객체

        Returns:
            bool: 문서가 추가되었으면 True, 중복이면 False
        """
        doc_key = self._create_document_key(document)

        # 중복 확인
        if doc_key in self._document_keys:
            return False

        # 문서 추가
        doc_name = document.metadata.get("doc_name", document.metadata.get("source", "default"))
        self._documents[doc_name].append(document)
        self._document_keys.add(doc_key)
        return True

    def add_batch(self, documents: List[Document]) -> int:
        """
        문서 배치 추가

        Args:
            documents: 추가할 Document 객체 리스트

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
        모든 저장된 문서 반환

        Returns:
            List[Document]: 저장된 모든 Document 객체
        """
        return [doc for docs in self._documents.values() for doc in docs]

    def get_by_doc_name(self, doc_name: str) -> List[Document]:
        """
        문서 이름으로 문서 조회

        Args:
            doc_name: 조회할 문서 이름

        Returns:
            List[Document]: 해당 이름의 Document 객체 리스트
        """
        return self._documents.get(doc_name, [])

    def get_by_source(self, source: str) -> List[Document]:
        """
        소스별 문서 조회

        Args:
            source: 조회할 소스명

        Returns:
            List[Document]: 해당 소스의 Document 객체 리스트
        """
        results = []
        for docs in self._documents.values():
            for doc in docs:
                if doc.metadata.get("source_type") == source:
                    results.append(doc)
        return results

    def clear(self) -> None:
        """문서 저장소 비우기"""
        self._documents.clear()
        self._document_keys.clear()

    def size(self) -> int:
        """
        저장된 문서 총 개수 반환

        Returns:
            int: 문서 개수
        """
        return sum(len(docs) for docs in self._documents.values())

    @classmethod
    def _create_document_key(cls, document: Document) -> str:
        """
        문서의 고유 키 생성

        Args:
            document: Document 객체

        Returns:
            str: 고유 문서 키
        """
        content_hash = hashlib.md5(document.page_content.encode('utf-8')).hexdigest()
        source = document.metadata.get("source", "")
        page = document.metadata.get("doc_page", "")

        return f"{source}_{page}_{content_hash}"
