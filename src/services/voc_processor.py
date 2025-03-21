import logging
import re
from functools import lru_cache
from typing import Dict, List, Set, Pattern, Optional, Any, Tuple

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

        # 설정 값 캐싱
        self._cached_settings: Dict[str, str] = self._load_cached_settings()

        # 정규식 패턴 컴파일 및 캐싱
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._init_patterns()

        # 예외 패턴 정의
        self._excluded_doc_ids: Set[str] = {"12345678", "00000000"}

        logger.debug("VOCLinkProcessor 인스턴스가 초기화되었습니다")

    def _load_cached_settings(self) -> Dict[str, str]:
        """
        자주 사용되는 설정 값을 캐싱합니다.

        Returns:
            Dict[str, str]: 캐싱된 설정 값 사전
        """
        try:
            return {
                'url_pattern': getattr(self.settings.voc, 'gw_doc_id_link_url_pattern', ''),
                'correct_pattern': getattr(self.settings.voc, 'gw_doc_id_link_correct_pattern', ''),
                'check_gw_word': getattr(self.settings.voc, 'check_gw_word_link', ''),
                'check_block_line': getattr(self.settings.voc, 'check_block_line', '')
            }
        except AttributeError as e:
            logger.error(f"설정 속성 접근 중 오류 발생: {e}")
            return {
                'url_pattern': '',
                'correct_pattern': '',
                'check_gw_word': '',
                'check_block_line': ''
            }

    def _init_patterns(self) -> None:
        """
        자주 사용되는 정규식 패턴을 컴파일하고 캐싱합니다.
        """
        try:
            # 기본 패턴 가져오기
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            # URL 패턴 컴파일
            if url_pattern:
                self._compiled_patterns['url'] = re.compile(url_pattern)

            # 올바른 형식 패턴 컴파일
            if correct_pattern:
                self._compiled_patterns['correct'] = re.compile(correct_pattern)

                # 최적화된 패턴: URL이 8자리 숫자로 끝나는지 직접 확인
                optimized_pattern = fr"{correct_pattern.rstrip('$')}\/(\d{{8}})(?:$|[?#])"
                self._compiled_patterns['optimized'] = re.compile(optimized_pattern)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{len(self._compiled_patterns)}개의 정규식 패턴이 컴파일되었습니다")
        except re.error as e:
            logger.error(f"정규식 컴파일 오류: {e}")
        except Exception as e:
            logger.error(f"패턴 초기화 중 오류 발생: {e}")

    def is_valid_url_pattern(self, source: str) -> bool:
        """
        문자열에 특정 URL 패턴이 있는지 확인합니다.

        Args:
            source (str): 확인할 문자열

        Returns:
            bool: 패턴이 존재하면 True, 그렇지 않으면 False
        """
        return self._check_pattern_in_string(source, 'url')

    def is_valid_doc_id_pattern(self, source: str) -> bool:
        """
        URL 패턴이 8자리 숫자로 끝나고 예외 값이 아닌지 확인합니다.

        URL은 8자리 숫자로 끝나야 하며 예외 값(12345678, 00000000)이 아니어야 합니다.
        효율적인 검사를 위해 최적화된 정규식을 사용합니다.

        Args:
            source (str): 확인할 문자열

        Returns:
            bool: 유효한 패턴이 존재하면 True, 그렇지 않으면 False
        """
        return self._validate_document_id_pattern(source)

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
            # 설정에서 패턴 가져오기
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            if not url_pattern or not correct_pattern:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning("URL 패턴 설정을 찾을 수 없습니다. 원본 텍스트를 반환합니다.")
                return text

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

        except AttributeError as e:
            logger.error(f"설정 속성 접근 중 오류 발생: {e}")
            return text  # 설정 오류 시 원본 반환
        except Exception as e:
            logger.error(f"링크 처리 중 오류 발생: {e}")
            return text  # 기타 오류 시 원본 반환

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

            # lru_cache 초기화
            self._is_valid_doc_id.cache_clear()

            # 패턴 다시 컴파일
            self._compiled_patterns.clear()
            self._init_patterns()

            logger.info("VOCLinkProcessor 설정이 다시 로드되었습니다")
            return True
        except Exception as e:
            logger.error(f"설정 다시 로드 중 오류 발생: {e}")
            return False

    #
    # 내부 헬퍼 메서드
    #

    def _check_pattern_in_string(self, source: str, pattern_key: str = 'url') -> bool:
        """
        문자열에 특정 패턴이 있는지 확인합니다.

        Args:
            source (str): 확인할 문자열
            pattern_key (str): 사용할 패턴 키 (기본값: 'url')

        Returns:
            bool: 패턴이 존재하면 True, 그렇지 않으면 False
        """
        if not source or not isinstance(source, str):
            return False

        # 컴파일된 패턴 가져오기
        pattern = self._compiled_patterns.get(pattern_key)
        if not pattern:
            # 패턴을 찾을 수 없는 경우 원본 설정 사용
            pattern_str = self._cached_settings.get(f'{pattern_key}_pattern', '')
            if not pattern_str:
                return False

            try:
                return bool(re.search(pattern_str, source))
            except re.error as e:
                logger.error(f"잘못된 정규식 패턴: {pattern_str}, 오류: {e}")
                return False

        try:
            return bool(pattern.search(source))
        except Exception as e:
            logger.error(f"패턴 검색 중 오류 발생: {e}")
            return False

    def _validate_document_id_pattern(self, source: str) -> bool:
        """
        URL 패턴이 8자리 숫자로 끝나고 예외 값이 아닌지 검증합니다.

        Args:
            source (str): 확인할 문자열

        Returns:
            bool: 유효한 패턴이 존재하면 True, 그렇지 않으면 False
        """
        if not source or not isinstance(source, str):
            return False

        try:
            # 최적화된 패턴 사용
            optimized_pattern = self._compiled_patterns.get('optimized')
            if optimized_pattern:
                matches = optimized_pattern.findall(source)
                return any(self._is_valid_doc_id(doc_id) for doc_id in matches)

            # 최적화된 패턴을 사용할 수 없는 경우 원래 방법 사용
            pattern = self._compiled_patterns.get('correct')
            if not pattern:
                pattern_str = self._cached_settings.get('correct_pattern', '')
                if not pattern_str:
                    return False
                matches = re.findall(pattern_str, source)
            else:
                matches = pattern.findall(source)

            # 각 일치 항목 검증
            for match in matches:
                # URL의 마지막 부분 추출
                end_path = match.split('/')[-1]

                # 유효성 검사 (캐싱 적용)
                if self._is_valid_doc_id(end_path):
                    return True

            return False

        except Exception as e:
            logger.error(f"패턴 검증 중 오류 발생: {e}")
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
