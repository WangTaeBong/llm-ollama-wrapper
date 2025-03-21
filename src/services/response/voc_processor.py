"""
VOC 링크 처리 모듈
===============

VOC(Voice of Customer) 관련 링크 처리 기능을 제공합니다.

기능:
- 전자결재 문서 링크 검증
- 링크 패턴 처리 및 변환
- 특수 텍스트 제거
"""

import logging
import re
from functools import lru_cache
from typing import Dict, List, Set, Pattern

from src.services.response.base import ProcessorBase
from src.services.response.factory import ProcessorFactory

# 로거 설정
logger = logging.getLogger(__name__)


class VOCLinkProcessor(ProcessorBase):
    """
    VOC 링크 처리기 구현

    전자결재 링크 검증 및 처리를 담당합니다.
    """

    def __init__(self, settings):
        """
        VOC 링크 처리기 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.settings = settings

        # 설정 값 캐싱
        self._cached_settings: Dict[str, str] = self._load_cached_settings()

        # 정규식 패턴 컴파일 및 캐싱
        self._compiled_patterns: Dict[str, Pattern] = {}
        self._init_patterns()

        # 예외 패턴 정의
        self._excluded_doc_ids: Set[str] = {"12345678", "00000000"}

        logger.debug("VOCLinkProcessor 인스턴스가 초기화되었습니다")

    @classmethod
    async def initialize(cls) -> bool:
        """
        처리기 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        # 이미 초기화 완료됨
        return True

    def _load_cached_settings(self) -> Dict[str, str]:
        """
        자주 사용되는 설정 값 캐싱

        Returns:
            Dict[str, str]: 캐시된 설정 값
        """
        try:
            return {
                'url_pattern': getattr(self.settings.voc, 'gw_doc_id_link_url_pattern', ''),
                'correct_pattern': getattr(self.settings.voc, 'gw_doc_id_link_correct_pattern', ''),
                'check_gw_word': getattr(self.settings.voc, 'check_gw_word_link', ''),
                'check_block_line': getattr(self.settings.voc, 'check_block_line', '')
            }
        except AttributeError as e:
            logger.error(f"설정 접근 중 오류: {e}")
            return {
                'url_pattern': '',
                'correct_pattern': '',
                'check_gw_word': '',
                'check_block_line': ''
            }

    def _init_patterns(self):
        """
        자주 사용되는 정규식 패턴 컴파일 및 캐싱
        """
        try:
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            if url_pattern:
                self._compiled_patterns['url'] = re.compile(url_pattern)
            if correct_pattern:
                self._compiled_patterns['correct'] = re.compile(correct_pattern)

                # 최적화된 패턴: URL이 8자리 숫자로 끝나는지 직접 확인
                optimized_pattern = fr"{correct_pattern.rstrip('$')}\/(\d{{8}})(?:$|[?#])"
                self._compiled_patterns['optimized'] = re.compile(optimized_pattern)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{len(self._compiled_patterns)}개 정규식 패턴이 컴파일되었습니다")
        except re.error as e:
            logger.error(f"정규식 컴파일 오류: {e}")
        except Exception as e:
            logger.error(f"패턴 초기화 중 오류: {e}")

    async def voc_judge_gw_doc_id_link_pattern(self, source: str, pattern_key: str = 'url') -> bool:
        """
        특정 URL 패턴이 문자열에 존재하는지 확인

        Args:
            source: 확인할 문자열
            pattern_key: 사용할 패턴 키 (기본: 'url')

        Returns:
            bool: 패턴이 존재하면 True, 없으면 False
        """
        if not source or not isinstance(source, str):
            return False

        # 컴파일된 패턴 가져오기
        pattern = self._compiled_patterns.get(pattern_key)
        if not pattern:
            # 패턴이 없으면 원본 설정 사용
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
            logger.error(f"패턴 검색 중 오류: {e}")
            return False

    @lru_cache(maxsize=64)
    async def _is_valid_doc_id(self, doc_id: str) -> bool:
        """
        문서 ID 유효성 확인 (캐싱 적용)

        Args:
            doc_id: 확인할 문서 ID

        Returns:
            bool: ID가 유효하면 True, 아니면 False
        """
        return (doc_id.isdigit() and
                len(doc_id) == 8 and
                doc_id not in self._excluded_doc_ids)

    async def voc_judge_gw_doc_id_link_correct_pattern(self, source: str, pattern_key: str = 'correct') -> bool:
        """
        URL 패턴이 8자리 숫자로 끝나고 특정 예외에 해당하지 않는지 확인

        URL은 8자리 숫자로 끝나야 하고 예외 값(12345678, 00000000)이 아니어야 합니다.
        효율적인 확인을 위해 최적화된 정규식을 사용합니다.

        Args:
            source: 확인할 문자열
            pattern_key: 사용할 패턴 키 (기본: 'correct')

        Returns:
            bool: 유효한 패턴이 존재하면 True, 없으면 False
        """
        if not source or not isinstance(source, str):
            return False

        try:
            # 최적화된 패턴 사용
            optimized_pattern = self._compiled_patterns.get('optimized')
            if optimized_pattern:
                matches = optimized_pattern.findall(source)
                return any(await self._is_valid_doc_id(doc_id) for doc_id in matches)

            # 최적화된 패턴이 없으면 원래 방식 사용
            pattern = self._compiled_patterns.get(pattern_key)
            if not pattern:
                pattern_str = self._cached_settings.get(f'{pattern_key}_pattern', '')
                if not pattern_str:
                    return False
                matches = re.findall(pattern_str, source)
            else:
                matches = pattern.findall(source)

            # 각 매치 검증
            for match in matches:
                # URL의 마지막 부분 추출
                end_path = match.split('/')[-1]

                # 유효성 검사 (캐싱 적용)
                if await self._is_valid_doc_id(end_path):
                    return True

            return False

        except Exception as e:
            logger.error(f"패턴 검증 중 오류: {e}")
            return False

    async def make_komico_voc_groupware_docid_url(self, query_answer: str) -> str:
        """
        챗봇 응답의 전자결재 링크 검증 및 불필요한 부분 제거

        Args:
            query_answer: 처리할 원본 응답 텍스트

        Returns:
            str: 처리된 응답 텍스트
        """
        if not query_answer:
            return ""

        try:
            # 설정에서 패턴 가져오기
            url_pattern = self._cached_settings.get('url_pattern', '')
            correct_pattern = self._cached_settings.get('correct_pattern', '')

            if not url_pattern or not correct_pattern:
                logger.warning("URL 패턴 설정을 찾을 수 없습니다. 원본 텍스트를 반환합니다.")
                return query_answer

            # 응답 텍스트를 줄 단위로 처리
            lines = query_answer.split('\n')
            processed_lines: List[str] = []
            is_link_exist = False

            for line in lines:
                # 빈 줄은 그대로 추가
                if not line:
                    processed_lines.append(line)
                    continue

                # URL 패턴 존재 여부 확인
                if await self.voc_judge_gw_doc_id_link_pattern(line):
                    # 올바른 패턴이면 추가
                    if await self.voc_judge_gw_doc_id_link_correct_pattern(line):
                        processed_lines.append(line)
                        is_link_exist = True
                        logger.debug(f"유효한 문서 링크 발견: {line[:50]}...")
                    else:
                        # 잘못된 패턴은 제외
                        logger.debug(f"잘못된 링크 패턴 제외: {line[:50]}...")
                else:
                    # URL 패턴이 없는 일반 텍스트는 그대로 추가
                    processed_lines.append(line)

            # 링크가 없는 경우 특정 단어/줄 제거
            if not is_link_exist:
                check_gw_word = self._cached_settings.get('check_gw_word', '')
                check_block_line = self._cached_settings.get('check_block_line', '')

                if check_gw_word or check_block_line:
                    logger.debug("문서 링크가 없어 관련 안내 텍스트를 제거합니다")

                    # 한 번의 반복으로 필터링
                    filtered_lines: List[str] = []
                    for line in processed_lines:
                        has_gw_word = check_gw_word and check_gw_word in line
                        has_block_line = check_block_line and check_block_line in line
                        if not (has_gw_word or has_block_line):
                            filtered_lines.append(line)

                    processed_lines = filtered_lines

            # 결과 결합
            result = '\n'.join(processed_lines).strip()

            return result

        except AttributeError as e:
            logger.error(f"설정 접근 중 오류: {e}")
            return query_answer  # 설정 오류 시 원본 반환
        except Exception as e:
            logger.error(f"링크 처리 중 오류: {e}")
            return query_answer  # 기타 오류 시 원본 반환

    async def reload_settings(self) -> bool:
        """
        설정 및 패턴 다시 로드

        설정이 동적으로 변경된 경우 호출하세요.

        Returns:
            bool: 다시 로드 성공 여부
        """
        try:
            # 캐시된 설정 다시 로드
            self._cached_settings = self._load_cached_settings()

            # lru_cache 정리
            self._is_valid_doc_id.cache_clear()

            # 패턴 다시 컴파일
            self._compiled_patterns.clear()
            self._init_patterns()

            logger.info("VOCLinkProcessor 설정이 다시 로드되었습니다")
            return True
        except Exception as e:
            logger.error(f"설정 다시 로드 중 오류: {e}")
            return False


# 클래스 등록
ProcessorFactory.register_processor("voc_link", VOCLinkProcessor)
