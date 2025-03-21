"""
문서 ID 검증 모듈

전자결재 문서 ID의 검증과 관련된 기능을 제공합니다.
"""

import logging
import re
from functools import lru_cache
from typing import Set, Any, List, Optional

# 모듈 레벨 로거 설정
logger = logging.getLogger(__name__)


class DocumentValidator:
    """
    문서 ID 검증 클래스

    전자결재 문서 ID의 유효성을 검사하는 기능을 제공합니다.
    """

    def __init__(self, settings: Any) -> None:
        """
        DocumentValidator 클래스 생성자

        Args:
            settings: 설정 정보를 포함하는 객체
        """
        self.settings = settings

        # 적용할 패턴 가져오기 (설정에서)
        try:
            self.correct_pattern = getattr(self.settings.voc, 'gw_doc_id_link_correct_pattern', '')
        except AttributeError as e:
            logger.error(f"설정 속성 접근 중 오류 발생: {e}")
            self.correct_pattern = ''

        # 정규식 패턴 컴파일
        self._compiled_pattern = None
        self._optimized_pattern = None
        self._init_patterns()

        # 제외할 문서 ID 정의
        self._excluded_doc_ids: Set[str] = {"12345678", "00000000"}

        logger.debug("DocumentValidator 인스턴스가 초기화되었습니다")

    def _init_patterns(self) -> None:
        """
        정규식 패턴을 컴파일합니다.
        """
        if not self.correct_pattern:
            return

        try:
            # 기본 패턴 컴파일
            self._compiled_pattern = re.compile(self.correct_pattern)

            # 최적화된 패턴 컴파일 (URL이 8자리 숫자로 끝나는지 직접 확인)
            optimized_pattern = fr"{self.correct_pattern.rstrip('$')}\/(\d{{8}})(?:$|[?#])"
            self._optimized_pattern = re.compile(optimized_pattern)

            logger.debug("문서 ID 검증 패턴이 성공적으로 컴파일되었습니다")
        except re.error as e:
            logger.error(f"정규식 컴파일 오류: {e}")
        except Exception as e:
            logger.error(f"패턴 초기화 중 오류 발생: {e}")

    def validate_document_id_pattern(self, source: str) -> bool:
        """
        문자열에 유효한 문서 ID 패턴이 있는지 검증합니다.

        Args:
            source (str): 검증할 문자열

        Returns:
            bool: 유효한 패턴이 있으면 True, 아니면 False
        """
        if not source or not isinstance(source, str):
            return False

        if not self._compiled_pattern or not self._optimized_pattern:
            logger.warning("패턴이 초기화되지 않았습니다")
            return False

        try:
            # 최적화된 패턴 사용
            matches = self._optimized_pattern.findall(source)
            return any(self._is_valid_doc_id(doc_id) for doc_id in matches)
        except Exception as e:
            logger.error(f"문서 ID 패턴 검증 중 오류 발생: {e}")
            return False

    @lru_cache(maxsize=64)
    def _is_valid_doc_id(self, doc_id: str) -> bool:
        """
        문서 ID가 유효한지 확인합니다. (캐싱 적용)

        Args:
            doc_id (str): 확인할 문서 ID

        Returns:
            bool: ID가 유효하면 True, 그렇지 않으면 False
        """
        return (doc_id.isdigit() and
                len(doc_id) == 8 and
                doc_id not in self._excluded_doc_ids)

    def find_document_ids(self, text: str) -> List[str]:
        """
        텍스트에서 모든 유효한 문서 ID를 찾습니다.

        Args:
            text (str): 검색할 텍스트

        Returns:
            List[str]: 찾은 유효한 문서 ID 목록
        """
        if not text or not isinstance(text, str):
            return []

        if not self._optimized_pattern:
            return []

        try:
            # 모든 매치 찾기
            matches = self._optimized_pattern.findall(text)

            # 유효한 문서 ID만 필터링
            valid_ids = [doc_id for doc_id in matches if self._is_valid_doc_id(doc_id)]

            return valid_ids
        except Exception as e:
            logger.error(f"문서 ID 검색 중 오류 발생: {e}")
            return []

    def reload_settings(self) -> None:
        """
        설정을 다시 로드하고 패턴을 재컴파일합니다.
        """
        try:
            # 패턴 가져오기 (설정에서)
            self.correct_pattern = getattr(self.settings.voc, 'gw_doc_id_link_correct_pattern', '')

            # 캐시 초기화
            self._is_valid_doc_id.cache_clear()

            # 패턴 다시 초기화
            self._init_patterns()

            logger.debug("DocumentValidator 설정이 다시 로드되었습니다")
        except Exception as e:
            logger.error(f"설정 다시 로드 중 오류 발생: {e}")
