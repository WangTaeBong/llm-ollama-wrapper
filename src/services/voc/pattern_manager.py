"""
정규식 패턴 관리 모듈

VOC 처리에 필요한 정규식 패턴을 관리하는 기능을 제공합니다.
"""

import logging
import re
from typing import Dict, Pattern, Any, Optional

# 모듈 레벨 로거 설정
logger = logging.getLogger(__name__)


class PatternManager:
    """
    정규식 패턴 관리 클래스

    VOC 처리에 필요한 정규식 패턴을 컴파일, 캐싱 및 관리합니다.
    """

    def __init__(self, settings: Any) -> None:
        """
        PatternManager 클래스 생성자

        Args:
            settings: 설정 정보를 포함하는 객체
        """
        self.settings = settings

        # 적용할 패턴 가져오기 (설정에서)
        try:
            self.url_pattern = getattr(self.settings.voc, 'gw_doc_id_link_url_pattern', '')
        except AttributeError as e:
            logger.error(f"설정 속성 접근 중 오류 발생: {e}")
            self.url_pattern = ''

        # 컴파일된 패턴 저장
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._init_patterns()

        # logger.debug("PatternManager 인스턴스가 초기화되었습니다")

    def _init_patterns(self) -> None:
        """
        정규식 패턴을 컴파일합니다.
        """
        if not self.url_pattern:
            return

        try:
            # URL 패턴 컴파일
            self._compiled_patterns['url'] = re.compile(self.url_pattern)
        except re.error as e:
            logger.error(f"정규식 컴파일 오류: {e}")
        except Exception as e:
            logger.error(f"패턴 초기화 중 오류 발생: {e}")

    def check_url_pattern(self, text: str) -> bool:
        """
        텍스트에 URL 패턴이 있는지 확인합니다.

        Args:
            text (str): 확인할 텍스트

        Returns:
            bool: URL 패턴이 있으면 True, 아니면 False
        """
        if not text or not isinstance(text, str):
            return False

        pattern = self._compiled_patterns.get('url')
        if not pattern:
            return False

        try:
            return bool(pattern.search(text))
        except Exception as e:
            logger.error(f"URL 패턴 검색 중 오류 발생: {e}")
            return False

    def get_pattern(self, key: str) -> Optional[Pattern]:
        """
        지정된 키에 대한 컴파일된 패턴을 반환합니다.

        Args:
            key (str): 패턴 키

        Returns:
            Optional[Pattern]: 컴파일된 패턴 또는 None
        """
        return self._compiled_patterns.get(key)

    def reload_settings(self) -> None:
        """
        설정을 다시 로드하고 패턴을 재컴파일합니다.
        """
        try:
            # 패턴 가져오기 (설정에서)
            self.url_pattern = getattr(self.settings.voc, 'gw_doc_id_link_url_pattern', '')

            # 패턴 다시 초기화
            self._compiled_patterns.clear()
            self._init_patterns()

            logger.debug("PatternManager 설정이 다시 로드되었습니다")
        except Exception as e:
            logger.error(f"설정 다시 로드 중 오류 발생: {e}")
