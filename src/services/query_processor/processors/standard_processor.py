"""
표준 쿼리 프로세서 모듈

기본적인 쿼리 처리 기능을 제공합니다.
"""

import logging
from src.services.query_processor.base import QueryProcessorBase
from src.services.query_processor.cache_manager import QueryCache

# 로거 설정
logger = logging.getLogger(__name__)


class StandardQueryProcessor(QueryProcessorBase):
    """
    표준 쿼리 프로세서 클래스

    기본적인 쿼리 정제 및 필터링 기능을 제공합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        표준 쿼리 프로세서 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)
        """
        super().__init__(settings, query_check_json_dict)

        # 캐시 인스턴스 가져오기
        self.cache = QueryCache.get_instance()

        # logger.debug("표준 쿼리 프로세서가 초기화되었습니다")

    def process_query(self, query: str) -> str:
        """
        쿼리를 정제하고 필터링하는 통합 메서드

        Args:
            query: 처리할 쿼리

        Returns:
            str: 처리된 쿼리
        """
        # 캐시 키 생성
        cache_key = f"std_process:{QueryCache.create_key(query)}"

        # 캐시에서 결과 확인
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 정제 및 필터링 수행
        cleaned = self.clean_query(query)
        filtered = self.filter_query(cleaned)

        # 결과 캐싱
        self.cache.set(cache_key, filtered)

        return filtered
