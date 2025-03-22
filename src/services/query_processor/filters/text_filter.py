"""
텍스트 필터 모듈

텍스트 기반 쿼리 필터링 기능을 제공합니다.
"""

import logging
import re
from typing import Pattern, List, Dict

# 로거 설정
logger = logging.getLogger(__name__)


class TextFilter:
    """
    텍스트 필터 클래스

    텍스트 기반 쿼리 필터링 기능을 제공합니다.
    """

    def __init__(self, settings):
        """
        텍스트 필터 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings

        # 필터링 패턴 컴파일
        self._compile_patterns()

        logger.debug("텍스트 필터가 초기화되었습니다")

    def _compile_patterns(self) -> None:
        """필터링에 사용할 정규 표현식 패턴을 컴파일합니다."""
        self._patterns: Dict[str, Pattern] = {
            'ko_jamo': re.compile(r'([ㄱ-ㅎㅏ-ㅣ\s]+)'),
            'arabia_num': re.compile(r'([0-9]+)'),
            'wild_char': re.compile(r'([^\w\s]+)'),
            'clean': re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]')
        }

    def filter_query(self, query: str) -> str:
        """
        쿼리를 필터링합니다.

        Args:
            query: 필터링할 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        if not query:
            return ""

        try:
            result = query

            # 설정에 따라 필터링 적용
            if getattr(self.settings.query_filter, 'ko_jamo', False):
                result = self._patterns['ko_jamo'].sub('', result)

            if getattr(self.settings.query_filter, 'arabia_num', False):
                result = self._patterns['arabia_num'].sub('', result)

            if getattr(self.settings.query_filter, 'wild_char', False):
                result = self._patterns['wild_char'].sub('', result)

            return result
        except Exception as e:
            logger.warning(f"쿼리 필터링 중 오류 발생: {e}")
            return query

    def clean_query(self, query: str) -> str:
        """
        쿼리를 정제합니다.

        Args:
            query: 정제할 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        return self._patterns['clean'].sub('', query.lower())
