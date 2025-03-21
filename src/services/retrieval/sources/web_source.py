"""
웹 검색 소스 모듈
==============

웹 검색 엔진을 통해 문서를 검색하는 기능을 제공합니다.

기능:
- DuckDuckGo 기반 웹 검색
- 검색 결과 처리 및 문서 변환
- 캐싱 및 오류 처리
"""

import asyncio
import logging
import time
from typing import List, Optional

from duckduckgo_search import DDGS
from langchain_core.documents import Document
from tenacity import retry, stop_after_attempt, wait_exponential

from ..base import RetrievalSourceBase
from ..cache import CacheManager
from ..factory import RetrievalSourceFactory

# 로거 설정
logger = logging.getLogger(__name__)


class WebRetrievalSource(RetrievalSourceBase):
    """
    웹 기반 검색 소스 구현

    DuckDuckGo 검색 엔진을 사용하여 웹 검색 결과를 제공합니다.
    """

    def __init__(self, settings):
        """
        웹 검색 소스 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.web_settings = getattr(settings, 'web_search', None)

        # 기본 설정 초기화
        self.enabled = getattr(self.web_settings, 'use_flag', False) if self.web_settings else False
        self.region = getattr(self.web_settings, 'region', 'wt-wt') if self.web_settings else 'wt-wt'
        self.max_results = getattr(self.web_settings, 'max_results', 10) if self.web_settings else 10

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager(ttl=1800)  # 30분 캐시

        # 초기화 상태 플래그
        self.is_initialized = False

    async def initialize(self) -> bool:
        """
        웹 검색 소스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 웹 검색이 활성화되지 않았으면 초기화 성공으로 간주
        if not self.enabled:
            logger.info("웹 검색 기능이 비활성화되어 있습니다.")
            self.is_initialized = True
            return True

        # 웹 검색 설정 확인
        if not self.web_settings:
            logger.error("웹 검색 설정이 없습니다.")
            return False

        # DuckDuckGo 간단한 테스트 검색
        try:
            with DDGS() as ddgs:
                # 간단한 테스트 쿼리 실행 (결과는 사용하지 않음)
                list(ddgs.text("test query", region=self.region, max_results=1))

            self.is_initialized = True
            logger.info(f"웹 검색 소스 초기화 완료 (region: {self.region}, max_results: {self.max_results})")
            return True
        except Exception as e:
            logger.error(f"웹 검색 소스 초기화 실패: {str(e)}")
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        웹 검색 수행

        Args:
            query: 검색 쿼리
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if not self.is_initialized:
            logger.error("웹 검색 소스가 초기화되지 않았습니다.")
            return []

        # 웹 검색이 비활성화되어 있으면 빈 결과 반환
        if not self.enabled:
            return []

        # 쿼리가 비어있으면 빈 결과 반환
        if not query or not query.strip():
            logger.warning("빈 검색 쿼리가 제공되었습니다.")
            return []

        session_id = kwargs.get('session_id', 'unknown')
        start_time = time.time()

        try:
            # 캐시 키 생성
            cache_key = self.cache_manager.create_key(f"websearch_{query}")

            # 캐시 확인
            cached_results = self.cache_manager.get(cache_key)
            if cached_results is not None:
                logger.debug(f"[{session_id}] 웹 검색 캐시 히트: {query}")
                return cached_results

            # 웹 검색 수행
            logger.debug(f"[{session_id}] 웹 검색 시작: {query}")
            documents = await self._perform_web_search(query)

            # 캐시에 결과 저장
            self.cache_manager.set(cache_key, documents)

            # 메트릭 업데이트
            elapsed = time.time() - start_time
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["document_count"] += len(documents)

            logger.debug(f"[{session_id}] 웹 검색 완료: {elapsed:.4f}초, {len(documents)}개 문서 검색됨")

            return documents

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] 웹 검색 중 오류(elapsed: {elapsed:.4f}): {str(e)}")
            self.metrics["error_count"] += 1
            return []

    async def _perform_web_search(self, query: str) -> List[Document]:
        """
        실제 웹 검색 수행 내부 메서드

        Args:
            query: 검색 쿼리

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        try:
            # DuckDuckGo 검색
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region=self.region, max_results=self.max_results))

            # 결과를 Document 객체로 변환
            documents = []
            for result in results:
                if not result or not isinstance(result, dict):
                    continue

                body = result.get('body', '')
                if not body:
                    continue

                # Document 객체 생성
                doc = Document(
                    page_content=body,
                    metadata={
                        'source': result.get('title', 'Unknown'),
                        'doc_page': result.get('href', '#'),
                        'source_type': 'web'
                    }
                )
                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"웹 검색 엔진 오류: {str(e)}")
            return []

    def get_name(self) -> str:
        """
        검색 소스 이름 반환

        Returns:
            str: 검색 소스 이름
        """
        return "web"


# 팩토리에 소스 등록
RetrievalSourceFactory.register_source("web", WebRetrievalSource)
