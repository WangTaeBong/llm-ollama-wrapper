"""
문서 검색 서비스 모듈

채팅 시스템을 위한 문서 검색 및 처리 기능을 제공합니다.
비동기 처리, 캐싱, 필터링 기능을 통해 효율적인
문서 검색과 관리를 지원합니다.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import List, Dict, Any, Optional, Set

from langchain_core.documents import Document

from src.common.config_loader import ConfigLoader
from src.schema.chat_req import ChatRequest
from src.services.document_processor import DocumentProcessor
from src.services.search_engine import SearchEngine

# 로거 설정
logger = logging.getLogger(__name__)


class RetrieverService:
    """
    문서 검색 서비스 클래스

    채팅 콘텍스트를 위한 문서 검색 및 처리를 수행합니다.
    주요 기능:
    - 문서 검색 및 변환
    - 웹 검색 결과 통합
    - 문서 필터링 및 품질 향상
    - 캐싱 및 성능 최적화
    - 비동기 처리 지원
    """

    # 클래스 수준 캐시 (인스턴스 간 공유)
    _document_cache: Dict[str, List[Document]] = {}
    _cache_ttl: Dict[str, float] = {}  # 캐시 만료 시간
    _default_ttl = 300  # 기본 캐시 유효 시간(초)

    def __init__(self, request: ChatRequest, settings=None):
        """
        검색 서비스 초기화

        Args:
            request (ChatRequest): 채팅 요청 인스턴스
            settings (Any, optional): 설정 객체, 제공되지 않으면 전역 설정 사용
        """
        self.request = request
        self.documents: List[Document] = []
        self.session_id = request.meta.session_id

        # 설정 의존성 주입
        if settings is None:
            self.settings = ConfigLoader().get_settings()
        else:
            self.settings = settings

        # 컴포넌트 초기화
        self.document_processor = DocumentProcessor(self.settings)
        self.search_engine = None  # 지연 초기화

        # 성능 측정 및 메트릭
        self.start_time = time.time()
        self.metrics = {
            "retrieval_time": 0.0,
            "web_search_time": 0.0,
            "filter_time": 0.0,
            "document_count": 0,
            "filtered_count": 0
        }

        # 중복 검사를 위한 문서 ID 세트
        self._document_ids: Set[str] = set()

        # 이 요청의 캐시 키 생성
        self.cache_key = self._create_cache_key()

        # 캐시에서 문서 로드 시도
        self._load_from_cache()

    def _create_cache_key(self) -> str:
        """
        요청 데이터로부터 캐시 키를 생성합니다.

        Returns:
            str: 고유한 캐시 키
        """
        cache_data = {
            "rag_sys_info": self.request.meta.rag_sys_info,
            "session_id": self.session_id,
            "query": self.request.chat.user
        }

        # JSON 문자열로 변환 후 해시 생성
        key_str = json.dumps(cache_data, sort_keys=True)
        return f"doc_cache:{hashlib.md5(key_str.encode('utf-8')).hexdigest()}"

    def _load_from_cache(self) -> bool:
        """
        캐시에서 문서를 로드합니다.

        Returns:
            bool: 캐시 적중 여부
        """
        # 캐시에 해당 키가 있고 만료되지 않았는지 확인
        if (self.cache_key in self._document_cache and
                self.cache_key in self._cache_ttl and
                time.time() < self._cache_ttl[self.cache_key]):
            self.documents = self._document_cache[self.cache_key]
            self.metrics["document_count"] = len(self.documents)

            # 문서 ID 세트 업데이트
            self._document_ids = {self._get_document_id(doc) for doc in self.documents}

            logger.debug(f"[{self.session_id}] 캐시에서 {len(self.documents)}개 문서 로드됨")
            return True

        return False

    def _save_to_cache(self, ttl: Optional[int] = None) -> None:
        """
        문서를 캐시에 저장합니다.

        Args:
            ttl (Optional[int]): 캐시 유효 시간(초), 기본값 사용 시 None
        """
        if not self.documents:
            return

        # 캐시에 저장
        self._document_cache[self.cache_key] = self.documents

        # 만료 시간 설정
        expiration = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache_ttl[self.cache_key] = expiration

        logger.debug(f"[{self.session_id}] {len(self.documents)}개 문서 캐시 저장됨")

        # 오래된 캐시 정리 (10% 확률로 실행)
        if hash(self.cache_key) % 10 == 0:
            self._cleanup_cache()

    @classmethod
    def _cleanup_cache(cls) -> None:
        """오래된 캐시 항목을 정리합니다."""
        current_time = time.time()
        expired_keys = [
            key for key in cls._cache_ttl
            if current_time > cls._cache_ttl[key]
        ]

        # 만료된 항목 삭제
        for key in expired_keys:
            cls._cache_ttl.pop(key, None)
            cls._document_cache.pop(key, None)

        if expired_keys:
            logger.debug(f"{len(expired_keys)}개의 만료된 캐시 항목 정리됨")

    def retrieve_documents(self) -> List[Document]:
        """
        페이로드를 문서로 변환하여 검색합니다.

        Returns:
            List[Document]: 검색된 문서 목록

        Raises:
            Exception: 문서 검색 실패 시
        """
        # 이미 문서가 로드된 경우 캐시된 결과 반환
        if self.documents:
            return self.documents

        start_time = time.time()

        try:
            logger.debug(f"[{self.session_id}] 문서 검색 시작")

            # DocumentProcessor를 사용하여 페이로드를 문서로 변환
            documents = self.document_processor.convert_payload_to_document(self.request)

            # 중복 제거 및 문서 ID 추적을 위한 처리
            for doc in documents:
                doc_id = self._get_document_id(doc)
                if doc_id not in self._document_ids:
                    self.documents.append(doc)
                    self._document_ids.add(doc_id)

            elapsed = time.time() - start_time
            self.metrics["retrieval_time"] = elapsed
            self.metrics["document_count"] = len(self.documents)

            logger.debug(
                f"[{self.session_id}] {len(self.documents)}개 문서 검색됨 - {elapsed:.4f}초 소요"
            )

            # 결과 캐싱
            self._save_to_cache()

            return self.documents

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{self.session_id}] 문서 검색 중 오류: {str(e)} - {elapsed:.4f}초 소요",
                exc_info=True
            )
            raise

    async def retrieve_documents_async(self) -> List[Document]:
        """
        비동기적으로 문서를 검색합니다.
        asyncio.to_thread를 사용하여 동기 메서드를 비동기적으로 실행합니다.

        Returns:
            List[Document]: 검색된 문서 목록
        """
        # 이미 문서가 로드된 경우 캐시된 결과 반환
        if self.documents:
            return self.documents

        try:
            return await asyncio.to_thread(self.retrieve_documents)
        except Exception as e:
            logger.error(f"[{self.session_id}] 비동기 문서 검색 중 오류: {str(e)}", exc_info=True)
            return []

    async def add_web_search_results(self) -> List[Document]:
        """
        설정에 따라 웹 검색 결과를 문서에 비동기적으로 추가합니다.

        Returns:
            List[Document]: 웹 검색 결과가 추가된 문서 목록
        """
        # 웹 검색이 비활성화되었거나 문서가 없는 경우
        if not getattr(self.settings, 'web_search', {}).get('enabled', False):
            return self.documents

        start_time = time.time()

        try:
            # 검색 엔진 지연 초기화
            if self.search_engine is None:
                self.search_engine = SearchEngine(self.settings)

            # 사용자 쿼리로 웹 검색 수행
            query = self.request.chat.user

            # 비동기 웹 검색 실행
            logger.debug(f"[{self.session_id}] 웹 검색 시작: {query}")
            web_results = await self.search_engine.websearch_duckduckgo_async(query)

            # 중복 제거하며 검색 결과 추가
            added_count = 0
            for doc in web_results:
                doc_id = self._get_document_id(doc)
                if doc_id not in self._document_ids:
                    # 웹 검색 결과임을 표시하는 메타데이터 추가
                    doc.metadata["source_type"] = "web_search"

                    self.documents.append(doc)
                    self._document_ids.add(doc_id)
                    added_count += 1

            elapsed = time.time() - start_time
            self.metrics["web_search_time"] = elapsed
            self.metrics["document_count"] = len(self.documents)

            logger.debug(
                f"[{self.session_id}] 웹 검색 완료: {added_count}개 추가됨 - {elapsed:.4f}초 소요"
            )

            # 결과 캐싱
            self._save_to_cache()

            return self.documents

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{self.session_id}] 웹 검색 중 오류: {str(e)} - {elapsed:.4f}초 소요",
                exc_info=True
            )
            return self.documents

    def filter_documents(self, min_content_length: int = 10, max_documents: Optional[int] = None) -> List[Document]:
        """
        검색된 문서를 필터링하여 품질을 향상시킵니다.

        Args:
            min_content_length (int): 최소 내용 길이
            max_documents (Optional[int]): 최대 문서 수, None이면 모든 문서 반환

        Returns:
            List[Document]: 필터링된 문서 목록
        """
        if not self.documents:
            return []

        start_time = time.time()
        original_count = len(self.documents)

        try:
            filtered_docs = []
            for doc in self.documents:
                # 내용 길이 검사
                if not doc.page_content or len(doc.page_content.strip()) < min_content_length:
                    continue

                # 메타데이터 검사
                if not doc.metadata or not doc.metadata.get("source"):
                    # 소스 메타데이터 없으면 최소한의 정보 추가
                    if not doc.metadata:
                        doc.metadata = {}
                    doc.metadata["source"] = "unknown"

                filtered_docs.append(doc)

            # 최대 문서 수 제한 적용
            if max_documents and len(filtered_docs) > max_documents:
                filtered_docs = filtered_docs[:max_documents]

            # 필터링 결과 업데이트
            self.documents = filtered_docs
            self._document_ids = {self._get_document_id(doc) for doc in filtered_docs}

            elapsed = time.time() - start_time
            self.metrics["filter_time"] = elapsed
            self.metrics["filtered_count"] = original_count - len(filtered_docs)
            self.metrics["document_count"] = len(filtered_docs)

            logger.debug(
                f"[{self.session_id}] 문서 필터링: {original_count}개 → {len(filtered_docs)}개 "
                f"({elapsed:.4f}초 소요)"
            )

            # 결과 캐싱
            self._save_to_cache()

            return self.documents

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"[{self.session_id}] 문서 필터링 중 오류: {str(e)} - {elapsed:.4f}초 소요",
                exc_info=True
            )
            return self.documents

    def sort_documents_by_relevance(self, query: Optional[str] = None) -> List[Document]:
        """
        문서를 관련성에 따라 정렬합니다.

        Args:
            query (Optional[str]): 정렬 기준 쿼리, None이면 요청의 사용자 쿼리 사용

        Returns:
            List[Document]: 정렬된 문서 목록
        """
        if not self.documents or len(self.documents) <= 1:
            return self.documents

        if query is None:
            query = self.request.chat.user

        try:
            # 간단한 관련성 점수 계산 함수
            def relevance_score(doc: Document) -> float:
                # 문서 내용이 없으면 최저 점수
                if not doc.page_content:
                    return 0.0

                content = doc.page_content.lower()
                query_terms = set(query.lower().split())

                # 쿼리 단어 포함 여부로 점수 계산
                term_matches = sum(1 for term in query_terms if term in content)
                term_score = term_matches / max(1, len(query_terms))

                # 문서 길이도 고려 (너무 짧거나 긴 문서 패널티)
                length = len(content)
                length_score = min(1.0, length / 1000) if length < 1000 else 2000 / max(length, 1)

                # 출처 유형에 따른 가중치
                source_weight = 1.0
                if doc.metadata.get("source_type") == "web_search":
                    source_weight = 0.7  # 웹 검색 결과는 약간 낮은 가중치

                return (term_score * 0.7 + length_score * 0.3) * source_weight

            # 관련성 점수로 정렬
            self.documents.sort(key=relevance_score, reverse=True)

            logger.debug(f"[{self.session_id}] {len(self.documents)}개 문서를 관련성으로 정렬함")
            return self.documents

        except Exception as e:
            logger.error(f"[{self.session_id}] 문서 정렬 중 오류: {str(e)}", exc_info=True)
            return self.documents

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        검색 서비스의 성능 지표를 가져옵니다.

        Returns:
            Dict[str, Any]: 성능 지표
        """
        end_time = time.time()

        # 총 소요 시간 계산
        total_time = end_time - self.start_time

        # 지표 업데이트
        self.metrics.update({
            "total_time": total_time,
            "document_count": len(self.documents),
        })

        return self.metrics

    @staticmethod
    def _get_document_id(doc: Document) -> str:
        """
        문서의 고유 ID를 생성합니다.

        Args:
            doc (Document): 문서 객체

        Returns:
            str: 문서의 고유 ID
        """
        # 메타데이터에서 소스 정보 추출
        source = doc.metadata.get("source", "") or doc.metadata.get("doc_name", "")
        page = doc.metadata.get("doc_page", "")

        # 컨텐츠의 해시 생성 (앞부분 100자만 사용)
        content_sample = doc.page_content[:100] if doc.page_content else ""
        content_hash = hashlib.md5(content_sample.encode('utf-8')).hexdigest()

        return f"{source}:{page}:{content_hash}"

    @classmethod
    def clear_cache(cls) -> None:
        """
        모든 문서 캐시를 초기화합니다.
        서버 재시작이나 테스트 중에 유용합니다.
        """
        cls._document_cache.clear()
        cls._cache_ttl.clear()
        logger.info("문서 캐시가 초기화되었습니다")
