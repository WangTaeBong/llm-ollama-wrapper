"""
표준 쿼리 프로세서 모듈

기본적인 쿼리 처리 기능을 제공합니다.
"""

import logging
import re
from typing import Optional, Set

from src.services.query_processor.base import QueryProcessorBase

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

        # 컴파일된 정규 표현식 패턴 (생성자에서 미리 컴파일하여 성능 최적화)
        self._clean_query_pattern = re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]')
        self._ko_jamo_pattern = re.compile(r'([ㄱ-ㅎㅏ-ㅣ\s]+)')
        self._arabia_num_pattern = re.compile(r'([0-9]+)')
        self._wild_char_pattern = re.compile(r'([^\w\s]+)')

        logger.debug("표준 쿼리 프로세서가 초기화되었습니다")

    def clean_query(self, query: str) -> str:
        """
        사용자 입력 쿼리에서 불필요한 특수 문자와 기호를 제거합니다.

        Args:
            query (str): 정제할 원본 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        return self._clean_query_pattern.sub('', query.lower())

    def filter_query(self, query: str) -> str:
        """
        쿼리에서 불필요한 문자를 제거하기 위한 필터를 적용합니다.

        Args:
            query (str): 필터링할 원본 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        if not query:
            return ""

        try:
            # 설정에 따라 필터링 패턴 적용
            if self.settings.query_filter.ko_jamo:
                query = self._ko_jamo_pattern.sub('', query)
            if self.settings.query_filter.arabia_num:
                query = self._arabia_num_pattern.sub('', query)
            if self.settings.query_filter.wild_char:
                query = self._wild_char_pattern.sub('', query)
            return query
        except AttributeError as e:
            # 설정 접근 관련 오류만 로깅
            logger.warning(f"filter_query에서 설정 접근 오류: {e}")
            return query
