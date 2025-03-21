"""
VOC 링크 처리 모듈

VOC 시스템의 링크 처리 관련 기능을 구현합니다.
"""

import logging
from typing import List, Dict, Any, Optional

from src.services.voc.document_validator import DocumentValidator
from src.services.voc.pattern_manager import PatternManager

# 모듈 레벨 로거 설정
logger = logging.getLogger(__name__)


class VOCLinkProcessor:
    """
    VOC 링크 처리 클래스

    전자결재 문서의 링크 검증, 처리 및 변환 기능을 제공합니다.
    정규식 패턴 캐싱과 성능 최적화 기능을 포함합니다.
    """

    def __init__(self, settings: Any) -> None:
        """
        VOCLinkProcessor 클래스 생성자

        Args:
            settings: 설정 정보를 포함하는 객체
        """
        self.settings = settings

        # 패턴 관리자 초기화
        self.pattern_manager = PatternManager(settings)

        # 문서 검증기 초기화
        self.document_validator = DocumentValidator(settings)

        # 설정 값 캐싱
        self._cached_settings = self._load_cached_settings()

        logger.debug("VOCLinkProcessor 인스턴스가 초기화되었습니다")

    def _load_cached_settings(self) -> Dict[str, str]:
        """
        자주 사용되는 설정 값을 캐싱합니다.

        Returns:
            Dict[str, str]: 캐싱된 설정 값 사전
        """
        try:
            return {
                'check_gw_word': getattr(self.settings.voc, 'check_gw_word_link', ''),
                'check_block_line': getattr(self.settings.voc, 'check_block_line', '')
            }
        except AttributeError as e:
            logger.error(f"설정 속성 접근 중 오류 발생: {e}")
            return {
                'check_gw_word': '',
                'check_block_line': ''
            }

    def is_valid_url_pattern(self, source: str) -> bool:
        """
        문자열에 특정 URL 패턴이 있는지 확인합니다.

        Args:
            source (str): 확인할 문자열

        Returns:
            bool: 패턴이 존재하면 True, 그렇지 않으면 False
        """
        return self.pattern_manager.check_url_pattern(source)

    def is_valid_doc_id_pattern(self, source: str) -> bool:
        """
        URL 패턴이 유효한 문서 ID로 끝나는지 확인합니다.

        Args:
            source (str): 확인할 문자열

        Returns:
            bool: 유효한 패턴이 존재하면 True, 그렇지 않으면 False
        """
        return self.document_validator.validate_document_id_pattern(source)

    def process_voc_document_links(self, text: str) -> str:
        """
        채팅봇 응답에서 전자결재 링크를 검증하고 불필요한 부분을 제거합니다.

        Args:
            text (str): 처리할 원본 응답 텍스트

        Returns:
            str: 처리된 응답 텍스트
        """
        if not text:
            return ""

        try:
            # 응답 텍스트를 줄 단위로 처리
            lines = text.split('\n')
            processed_lines: List[str] = []
            has_valid_link = False

            for line in lines:
                # 빈 줄은 그대로 추가
                if not line:
                    processed_lines.append(line)
                    continue

                # URL 패턴 존재 확인
                if self.is_valid_url_pattern(line):
                    # 올바른 패턴인 경우 추가
                    if self.is_valid_doc_id_pattern(line):
                        processed_lines.append(line)
                        has_valid_link = True
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"유효한 문서 링크 발견: {line[:50]}...")
                    else:
                        # 잘못된 패턴 제외
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"잘못된 링크 패턴 제외: {line[:50]}...")
                else:
                    # URL 패턴이 없는 일반 텍스트는 그대로 추가
                    processed_lines.append(line)

            # 링크가 없는 경우 특정 단어/줄 제거
            if not has_valid_link:
                processed_lines = self._remove_guidance_text(processed_lines)

            # 결과 결합
            result = '\n'.join(processed_lines).strip()

            return result

        except Exception as e:
            logger.error(f"링크 처리 중 오류 발생: {e}")
            return text  # 오류 시 원본 반환

    def _remove_guidance_text(self, lines: List[str]) -> List[str]:
        """
        링크가 없을 때 관련 안내 텍스트를 제거합니다.

        Args:
            lines (List[str]): 처리할 텍스트 라인 목록

        Returns:
            List[str]: 필터링된 텍스트 라인 목록
        """
        check_gw_word = self._cached_settings.get('check_gw_word', '')
        check_block_line = self._cached_settings.get('check_block_line', '')

        if not (check_gw_word or check_block_line):
            return lines

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("문서 링크를 찾을 수 없어 관련 안내 텍스트를 제거합니다")

        # 하나의 반복에서 필터링
        filtered_lines: List[str] = []
        for line in lines:
            has_gw_word = check_gw_word and check_gw_word in line
            has_block_line = check_block_line and check_block_line in line
            if not (has_gw_word or has_block_line):
                filtered_lines.append(line)

        return filtered_lines

    def reload_settings(self) -> bool:
        """
        설정과 패턴을 다시 로드합니다.
        설정이 동적으로 변경된 경우 이 메서드를 호출하세요.

        Returns:
            bool: 다시 로드 작업 성공 여부
        """
        try:
            # 캐싱된 설정 다시 로드
            self._cached_settings = self._load_cached_settings()

            # 패턴 관리자와 문서 검증기 다시 로드
            self.pattern_manager.reload_settings()
            self.document_validator.reload_settings()

            logger.info("VOCLinkProcessor 설정이 다시 로드되었습니다")
            return True
        except Exception as e:
            logger.error(f"설정 다시 로드 중 오류 발생: {e}")
            return False

    # 하위 호환성을 위한 메서드들
    def voc_judge_gw_doc_id_link_pattern(self, source: str) -> bool:
        """하위 호환성을 위한 메서드"""
        return self.is_valid_url_pattern(source)

    def voc_judge_gw_doc_id_link_correct_pattern(self, source: str) -> bool:
        """하위 호환성을 위한 메서드"""
        return self.is_valid_doc_id_pattern(source)

    def make_komico_voc_groupware_docid_url(self, text: str) -> str:
        """하위 호환성을 위한 메서드"""
        return self.process_voc_document_links(text)
