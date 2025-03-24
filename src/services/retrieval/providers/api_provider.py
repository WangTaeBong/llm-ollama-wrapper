"""
API 제공자 모듈

외부 API를 통한 문서 검색 기능을 제공합니다.
"""

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document

from src.common.restclient import rc
from src.services.retrieval.base import DataProviderBase, CacheManagerBase

# 로거 설정
logger = logging.getLogger(__name__)


class APIProvider(DataProviderBase):
    """
    API 기반 데이터 제공자

    외부 API에서 문서를 검색하는 기능을 제공합니다.
    """

    def __init__(self, url: str, headers: Dict[str, str], cache_manager: CacheManagerBase, timeout: float = 60.0):
        """
        API 제공자 초기화

        Args:
            url: API 엔드포인트 URL
            headers: API 요청 헤더
            cache_manager: 캐시 관리자
            timeout: API 호출 제한 시간(초)
        """
        self.url = url
        self.headers = headers
        self.cache_manager = cache_manager
        self.timeout = timeout

    async def fetch_documents(self, query: str, **kwargs) -> List[Document]:
        """
        API를 호출하여 문서를 가져옵니다.

        Args:
            query: 검색 쿼리
            **kwargs: 추가 매개변수, request_data가 포함되어야 함

        Returns:
            List[Document]: 검색된 문서 목록
        """
        # request_data를 kwargs에서 가져옵니다
        request_data = kwargs.get('request_data')
        if not request_data:
            logger.error("request_data가 제공되지 않았습니다")
            return []

        # 요청 데이터 복사 및 쿼리 업데이트
        request_data_copy = request_data.copy()
        request_data_copy.chat.user = query

        # 로깅용 세션 ID 추출
        session_id = request_data_copy.meta.session_id

        # 캐시 키 생성
        cache_key = self.cache_manager.create_key({
            'url': self.url,
            'request': request_data_copy.dict()
        })

        # 캐시 확인
        cached_response = self.cache_manager.get(cache_key)
        if cached_response:
            logger.debug(f"[{session_id}] 쿼리 캐시 사용: {query}")
            return self._extract_documents_from_response(cached_response, session_id)

        start_time = time.time()
        logger.debug(f"[{session_id}] API 호출 시작: {query}")

        try:
            # 제한 시간 적용 API 호출
            response = await asyncio.wait_for(
                rc.restapi_post_async(self.url, request_data_copy.dict()),
                timeout=self.timeout
            )

            # 응답 상태 확인
            if response.get("status", 200) != 200:
                logger.error(f"[{session_id}] API 호출 실패: 상태 코드 {response.get('status')}")
                return []

            # 캐시에 응답 저장
            self.cache_manager.set(cache_key, response)

            api_time = time.time() - start_time
            logger.debug(f"[{session_id}] API 호출 완료: {api_time:.4f}초 소요")

            # 문서 추출 및 반환
            return self._extract_documents_from_response(response, session_id)

        except asyncio.TimeoutError:
            logger.warning(f"[{session_id}] API 호출 시간 초과(> {self.timeout}초): {query}")
            return []
        except Exception as e:
            logger.error(f"[{session_id}] API 호출 중 오류: {e}")
            return []

    @classmethod
    def _extract_documents_from_response(cls, response: Dict[str, Any], session_id: str = None) -> List[Document]:
        """
        API 응답에서 문서를 추출하여 Document 객체로 변환합니다.

        Args:
            response: API 응답 데이터
            session_id: 세션 ID (로깅용)

        Returns:
            List[Document]: 변환된 Document 객체 목록
        """
        documents = []

        try:
            # 페이로드 추출
            payload = response.get("chat", {}).get("payload", [])

            for item in payload:
                # 필요한 데이터가 있는지 확인
                content = item.get("content")
                if not content:
                    continue

                # 메타데이터 추출
                metadata = {
                    "doc_name": item.get("doc_name", ""),
                    "doc_page": item.get("doc_page", "")
                }

                # Document 객체 생성 및 추가
                documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            logger.debug(f"[{session_id}] 응답에서 {len(documents)}개 문서 추출")
            return documents

        except Exception as e:
            logger.error(f"[{session_id}] 응답 처리 중 오류: {e}")
            return []
