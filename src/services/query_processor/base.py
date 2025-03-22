"""
쿼리 프로세서 기본 인터페이스

모든 쿼리 프로세서 구현의 기본이 되는 추상 클래스를 제공합니다.
"""

import abc
import logging
import re
from typing import Optional, Dict, Any

# 로거 설정
logger = logging.getLogger(__name__)


class QueryProcessorBase(abc.ABC):
    """
    쿼리 프로세서 기본 인터페이스

    모든 쿼리 프로세서 구현체가 준수해야 하는 인터페이스를 정의합니다.
    공통 기능을 제공하여 코드 중복을 최소화합니다.
    """

    def __init__(self, settings, query_check_json_dict=None):
        """
        쿼리 프로세서 초기화

        Args:
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전 (선택 사항)
        """
        self.settings = settings
        self.query_check_json_dict = query_check_json_dict

        # 모든 프로세서에서 공통으로 사용하는 정규식 패턴 초기화
        self._compile_common_patterns()

        # logger.debug(f"{self.__class__.__name__} 초기화됨")

    def _compile_common_patterns(self):
        """공통 정규식 패턴을 컴파일합니다."""
        self._patterns = {
            'clean': re.compile(r'[~!@#$%^&*()=+\[\]{}:?,<>/\-_.]'),
            'ko_jamo': re.compile(r'([ㄱ-ㅎㅏ-ㅣ\s]+)'),
            'arabia_num': re.compile(r'([0-9]+)'),
            'wild_char': re.compile(r'([^\w\s]+)')
        }

    def clean_query(self, query: str) -> str:
        """
        쿼리를 정제합니다. 기본 구현은 일반적인 특수 문자를 제거합니다.

        Args:
            query: 정제할 쿼리

        Returns:
            str: 정제된 쿼리
        """
        if not query:
            return ""
        return self._patterns['clean'].sub('', query.lower())

    def filter_query(self, query: str) -> str:
        """
        쿼리를 필터링합니다. 설정에 따라 필터링 패턴을 적용합니다.

        Args:
            query: 필터링할 쿼리

        Returns:
            str: 필터링된 쿼리
        """
        if not query:
            return ""

        try:
            result = query

            # 설정에 따라 필터링 패턴 적용
            filter_config = getattr(self.settings, 'query_filter', None)
            if filter_config:
                if getattr(filter_config, 'ko_jamo', False):
                    result = self._patterns['ko_jamo'].sub('', result)
                if getattr(filter_config, 'arabia_num', False):
                    result = self._patterns['arabia_num'].sub('', result)
                if getattr(filter_config, 'wild_char', False):
                    result = self._patterns['wild_char'].sub('', result)

            return result
        except AttributeError as e:
            logger.warning(f"설정 접근 오류: {e}")
            return query

    def get_cached_setting(self, setting_path: str, default=None) -> Any:
        """
        설정값을 안전하게 가져오는 유틸리티 메서드

        Args:
            setting_path: '.'으로 구분된 설정 경로 (예: 'web_search.use_flag')
            default: 설정이 없을 경우 반환할 기본값

        Returns:
            Any: 설정값 또는 기본값
        """
        parts = setting_path.split('.')
        obj = self.settings

        try:
            for part in parts:
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return default

    @property
    def processor_name(self) -> str:
        """
        프로세서 이름을 반환합니다.

        Returns:
            str: 프로세서 이름
        """
        return self.__class__.__name__
