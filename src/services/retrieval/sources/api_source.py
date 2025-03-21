"""
API 기반 검색 소스 모듈
====================

외부 API를 통해 문서를 검색하는 기능을 제공합니다.

기능:
- API 기반 문서 검색
- 응답 처리 및 문서 변환
- 캐싱 및 오류 처리
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document

from ..base import RetrievalSourceBase
from ..cache import CacheManager
from ..factory import RetrievalSourceFactory
from src.common.restclient import rc
from src.schema.retriever_req import RetrieverRequest

# 로거 설정
logger = logging.getLogger(__name__)


class APIRetrievalSource(RetrievalSourceBase):
    """
    API 기반 검색 소스 구현

    외부 API를 통해 문서를 검색하고 결과를 처리합니다.
    """

    def __init__(self, settings):
        """
        API 검색 소스 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.api_settings = getattr(settings, 'api', None)
        self.retriever_settings = getattr(settings, 'retriever', None)

        # API 설정 초기화
        self.url = getattr(self.api_settings, 'retrival_api', '') if self.api_settings else ''
        self.headers = {"content-type": "application/json;charset=utf-8"}

        # 캐시 관리자 초기화
        self.cache_manager = CacheManager(ttl=3600)

        # 초기화 상태 플래그
        self.is_initialized = False

        # 타임아웃 설정
        self.timeout = 60.0

    async def initialize(self) -> bool:
        """
        API 검색 소스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        if self.is_initialized:
            return True

        if not self.url:
            logger.error("API URL이 설정되지 않았습니다.")
            return False

        # 간단한 상태 점검은 실제 구현에서 추가
        self.is_initialized = True
        logger.info(f"API 검색 소스 초기화 완료 (URL: {self.url})")
        return True

    async def retrieve(self, query: str, request_data: Optional[RetrieverRequest] = None, **kwargs) -> List[Document]:
        """
        API를 통한 문서 검색

        Args:
            query: 검색 쿼리
            request_data: API 요청 데이터
            **kwargs: 추가 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        if not self.is_initialized:
            logger.error("API 검색 소스가 초기화되지 않았습니다.")
            return []

        session_id = request_data.meta.session_id if request_data and hasattr(request_data, 'meta') else "unknown"
        start_time = time.time()

        try:
            # 요청 데이터 복사 및 쿼리 업데이트
            request_data_copy = request_data.copy()
            request_data_copy.chat.user = query

            # 캐시 키 생성
            cache_key = self.cache_manager.create_key({
                'url': self.url,
                'request': request_data_copy.dict()
            })

            # 캐시 확인
            cached_response = self.cache_manager.get(cache_key)
            if cached_response:
                logger.debug(f"[{session_id}] API 캐시 사용: {query}")
                documents = self._extract_documents(cached_response, session_id)
                return documents

            # API 호출
            logger.debug(f"[{session_id}] API 호출 시작: {query}")
            response = await asyncio.wait_for(
                rc.restapi_post_async(self.url, request_data_copy.dict()),
                timeout=self.timeout
            )

            # 응답 상태 확인
            if response.get("status", 200) != 200:
                logger.error(f"[{session_id}] API 호출 실패: {response.get('status')}")
                return []

            # 캐시에 응답 저장
            self.cache_manager.set(cache_key, response)

            # 문서 추출
            documents = self._extract_documents(response, session_id)

            # 메트릭 업데이트
            elapsed = time.time() - start_time
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed
            self.metrics["document_count"] += len(documents)

            logger.debug(f"[{session_id}] API 검색 완료: {elapsed:.4f}초, {len(documents)}개 문서 검색됨")

            return documents

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.warning(f"[{session_id}] API 호출 타임아웃 (>{self.timeout}초, elapsed: {elapsed:.4f}): {query}")
            self.metrics["error_count"] += 1
            return []
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id}] API 호출 중 오류(elapsed: {elapsed:.4f}): {str(e)}")
            self.metrics["error_count"] += 1
            return []

    @classmethod
    def _extract_documents(cls, response: Dict[str, Any], session_id: str = None) -> List[Document]:
        """
        API 응답에서 문서 추출

        Args:
            response: API 응답 데이터
            session_id: 세션 ID

        Returns:
            List[Document]: 추출된 Document 객체 리스트
        """
        documents = []

        try:
            # 페이로드 추출
            payload = response.get("chat", {}).get("payload", [])

            for item in payload:
                # 필요한 데이터 확인
                content = item.get("content")
                if not content:
                    continue

                # 메타데이터 추출
                metadata = {
                    key: item.get(key, "")
                    for key in ["doc_name", "doc_page"]
                    if key in item
                }

                # 소스 타입 추가
                metadata["source_type"] = "api"

                # Document 객체 생성 및 추가
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            return documents

        except Exception as e:
            logger.error(f"[{session_id}] 응답 처리 중 오류: {str(e)}")
            return []

    def get_name(self) -> str:
        """
        검색 소스 이름 반환

        Returns:
            str: 검색 소스 이름
        """
        return "api"


# 팩토리에 소스 등록
RetrievalSourceFactory.register_source("api", APIRetrievalSource)
