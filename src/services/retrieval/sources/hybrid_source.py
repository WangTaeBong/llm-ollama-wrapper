"""
하이브리드 검색 소스 모듈
======================

API와 웹 검색을 통합하여 포괄적인 검색 결과를 제공합니다.

기능:
- API 검색과 웹 검색 결과 통합
- 문서 중복 제거 및 정렬
- 검색 전략 관리
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple

from langchain_core.documents import Document

from ..base import RetrievalSourceBase
from ..factory import RetrievalSourceFactory
from ..document_store import DocumentStore
from .api_source import APIRetrievalSource
from .web_source import WebRetrievalSource
from src.schema.retriever_req import RetrieverRequest

# 로거 설정
logger = logging.getLogger(__name__)


class HybridRetrievalSource(RetrievalSourceBase):
    """
    하이브리드 검색 소스 구현

    API와 웹 검색을 결합하여 더 포괄적인 검색 결과를 제공합니다.
    """

    def __init__(self, settings):
        """
        하이브리드 검색 소스 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)

        # 검색 소스 구성 요소
        self.api_source = APIRetrievalSource(settings)
        self.web_source = WebRetrievalSource(settings)

        # 문서 저장소 초기화
        self.document_store = DocumentStore()

        # 검색 전략 설정
        self.web_search_enabled = getattr(settings.web_search, 'use_flag', False)
        self.document_add_type = getattr(settings.web_search, 'document_add_type', 1)

        # 초기화 상태 플래그
        self.is_initialized = False

    async def initialize(self) -> bool:
        """
        하이브리드 검색 소스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        if self.is_initialized:
            return True

        # API 소스 초기화
        api_initialized = await self.api_source.initialize()

        # 웹 검색 초기화 (웹 검색이 활성화된 경우만)
        web_initialized = True
        if self.web_search_enabled:
            web_initialized = await self.web_source.initialize()

        # API 초기화 실패시 전체 실패 처리
        if not api_initialized:
            logger.error("API 검색 소스 초기화 실패")
            return False

        # 웹 검색 초기화 실패는 경고로 처리 (웹 검색 없이도 계속 진행)
        if self.web_search_enabled and not web_initialized:
            logger.warning("웹 검색 소스 초기화 실패, API 검색만 사용합니다.")
            self.web_search_enabled = False

        self.is_initialized = True
        logger.info(f"하이브리드 검색 소스 초기화 완료 (웹 검색: {self.web_search_enabled})")
        return True

    async def retrieve(self, query: str, request_data: Optional[RetrieverRequest] = None, **kwargs) -> List[Document]:
        """
        하이브리드 검색 수행

        Args:
            query: 검색 쿼리
            request_data: API 요청 데이터
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if not self.is_initialized:
            logger.error("하이브리드 검색 소스가 초기화되지 않았습니다.")
            return []

        session_id = request_data.meta.session_id if request_data and hasattr(request_data, 'meta') else kwargs.get(
            'session_id', 'unknown')
        start_time = time.time()

        try:
            # 검색 작업 목록 초기화
            tasks = []
            task_types = []

            # document_add_type이 0이고 웹 검색이 활성화된 경우 API 검색 건너뜀
            if not (self.web_search_enabled and self.document_add_type == 0):
                # API 검색 작업 추가
                tasks.append(self.api_source.retrieve(query, request_data=request_data, **kwargs))
                task_types.append("api")
                logger.debug(f"[{session_id}] API 검색 작업 추가됨")
            else:
                logger.debug(f"[{session_id}] 웹 검색 설정에 따라 API 검색을 건너뜁니다 (document_add_type={self.document_add_type})")

            # 웹 검색이 활성화된 경우 웹 검색 작업 추가
            if self.web_search_enabled:
                tasks.append(self.web_source.retrieve(query, session_id=session_id, **kwargs))
                task_types.append("web")
                logger.debug(f"[{session_id}] 웹 검색 작업 추가됨")

            # 모든 검색 작업 병렬 실행
            results = await asyncio.gather(*tasks)

            # 최종 문서 목록 초기화
            final_documents = []

            # 검색 전략에 따른 결과 처리
            for i, task_type in enumerate(task_types):
                if task_type == "api":
                    api_documents = results[i]
                    # API 결과를 기본으로 설정
                    final_documents = api_documents
                elif task_type == "web" and results[i]:
                    web_documents = results[i]
                    # 웹 검색 통합 전략 적용
                    if self.web_search_enabled:
                        if self.document_add_type == 1 and final_documents:
                            # 기존 API 문서에 웹 검색 결과 추가
                            final_documents.extend(web_documents)
                        elif self.document_add_type == 0:
                            # 웹 검색 결과만 사용
                            final_documents = web_documents

            # 메트릭 업데이트
            elapsed = time.time() - start_time
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["document_count"] += len(final_documents)

            logger.debug(f"[{session_id}] 하이브리드 검색 완료: {elapsed:.4f}초, {len(final_documents)}개 문서 검색됨")

            return final_documents

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] 하이브리드 검색 중 오류(elapsed: {elapsed:.4f}): {str(e)}")
            self.metrics["error_count"] += 1
            return []

    def get_name(self) -> str:
        """
        검색 소스 이름 반환

        Returns:
            str: 검색 소스 이름
        """
        return "hybrid"

    def get_metrics(self) -> Dict[str, Any]:
        """
        종합 메트릭 정보 반환

        Returns:
            Dict[str, Any]: 검색 메트릭 정보
        """
        # 기본 메트릭 정보 가져오기
        metrics = super().get_metrics()

        # 구성 요소 메트릭 추가
        metrics["api_metrics"] = self.api_source.get_metrics()
        metrics["web_metrics"] = self.web_source.get_metrics()

        return metrics


# 팩토리에 소스 등록
RetrievalSourceFactory.register_source("hybrid", HybridRetrievalSource)
