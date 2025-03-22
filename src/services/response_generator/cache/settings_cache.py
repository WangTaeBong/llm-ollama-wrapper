# src/services/response_generator/cache/settings_cache.py
"""
설정 캐싱 모듈

자주 사용되는 설정 값의 효율적인 캐싱 메커니즘을 제공합니다.
"""

import logging
import time
from typing import Dict, Any, List, Optional

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class SettingsCache:
    """
    설정 캐싱 클래스

    자주 사용되는 설정 값을 메모리에 캐싱하여 반복적인 설정 접근 성능을 최적화합니다.
    깊은 객체 경로에 대한 접근을 단순화하고 오류를 방지합니다.
    """

    def __init__(self, settings, ttl: int = 3600):
        """
        SettingsCache 초기화

        Args:
            settings: 시스템 설정 객체
            ttl: 캐시 Time-To-Live(초)
        """
        self.settings = settings
        self.ttl = ttl
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._known_paths: Dict[str, str] = {
            'source_rag_target': 'prompt.source_type',
            'none_source_rag_target': 'prompt.none_source_type',
            'faq_category_rag_target_list': 'prompt.faq_type',
        }

    def load_settings(self, settings_keys: List[str]) -> None:
        """
        지정된 설정 키 목록을 캐싱합니다.

        Args:
            settings_keys: 캐싱할 설정 키 목록
        """
        for key in settings_keys:
            self.get_setting(key)

    def get_setting(self, setting_name: str, default_value: Any = None) -> Any:
        """
        캐시된 설정 값을 반환하거나 필요한 경우 로드합니다.

        Args:
            setting_name: 설정의 캐시 키
            default_value: 설정을 찾을 수 없는 경우의 기본값

        Returns:
            Any: 설정 값 또는 기본값
        """
        # 캐시 만료 확인
        current_time = time.time()
        if setting_name in self._cache and current_time - self._timestamps.get(setting_name, 0) < self.ttl:
            return self._cache[setting_name]

        # 캐시 미스 또는 만료 - 설정에서 로드
        try:
            # 매핑된 경로 찾기
            path = self._known_paths.get(setting_name)
            if not path:
                logger.warning(f"알 수 없는 설정 키: {setting_name}, 기본값 반환")
                return default_value

            # 경로에서 값 추출
            value = self._get_value_from_path(path)

            # 리스트로 변환
            if value and isinstance(value, str) and ',' in value:
                value = value.split(',')

            # 캐시 업데이트
            self._cache[setting_name] = value
            self._timestamps[setting_name] = current_time

            return value
        except Exception as e:
            logger.error(f"설정 로드 중 오류: {e}")
            return default_value

    def _get_value_from_path(self, path: str) -> Any:
        """
        지정된 경로에서 설정 값을 가져옵니다.

        Args:
            path: 점으로 구분된 설정 경로(예: "prompt.source_type")

        Returns:
            Any: 검색된 설정 값

        Raises:
            AttributeError: 경로에 속성이 없는 경우
        """
        parts = path.split('.')
        value = self.settings

        for part in parts:
            value = getattr(value, part)

        return value

    def invalidate(self, setting_name: Optional[str] = None) -> None:
        """
        지정된 설정 또는 모든 설정을 무효화합니다.

        Args:
            setting_name: 무효화할 설정 이름(None이면 모든 설정)
        """
        if setting_name:
            self._cache.pop(setting_name, None)
            self._timestamps.pop(setting_name, None)
        else:
            self._cache.clear()
            self._timestamps.clear()

        logger.debug(f"캐시 무효화: {'전체' if setting_name is None else setting_name}")

    def register_path(self, cache_key: str, settings_path: str) -> None:
        """
        새 설정 경로 매핑을 등록합니다.

        Args:
            cache_key: 캐시 키
            settings_path: 설정 객체 내 경로
        """
        self._known_paths[cache_key] = settings_path
        logger.debug(f"설정 경로 등록: {cache_key} -> {settings_path}")
