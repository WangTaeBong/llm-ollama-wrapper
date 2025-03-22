"""
URL 처리 모듈

URL 처리 및 하이퍼링크 변환 기능을 제공합니다.
"""

import logging
import re
from typing import List, Optional

# 로거 설정
logger = logging.getLogger(__name__)


class URLProcessor:
    """
    URL 처리 클래스

    텍스트에서 URL을 감지하고 하이퍼링크로 변환하는 기능을 제공합니다.
    """

    def __init__(self):
        """
        URL 처리기 초기화
        """
        # URL 패턴 컴파일
        self._url_pattern = re.compile(
            r"https?://(?:www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-ㄱ-ㅎ가-힣a-zA-Z0-9@:%_+.~#?&/=]*")
        logger.debug("URL 처리기가 초기화되었습니다")

    def extract_urls(self, text: str) -> List[str]:
        """
        텍스트에서 URL을 추출합니다.

        Args:
            text: URL을 추출할 텍스트

        Returns:
            List[str]: 추출된 URL 목록
        """
        if not text:
            return []

        try:
            return self._url_pattern.findall(text)
        except Exception as e:
            logger.error(f"URL 추출 중 오류 발생: {str(e)}")
            return []

    def convert_urls_to_links(self, text: str, skip_reference_section: bool = True) -> str:
        """
        텍스트에서 URL을 하이퍼링크로 변환합니다.

        Args:
            text: 변환할 텍스트
            skip_reference_section: 참고문헌 섹션 처리 여부

        Returns:
            str: URL이 하이퍼링크로 변환된 텍스트
        """
        if not text:
            return ""

        try:
            # 참고문헌 섹션 검사
            reference_section_start = -1
            if skip_reference_section:
                reference_markers = ["[참고문헌]", "[References]", "[参考文献]"]
                for marker in reference_markers:
                    pos = text.find(marker)
                    if pos != -1:
                        reference_section_start = pos
                        break

            # 참고문헌 섹션이 없으면 전체 텍스트 처리
            if reference_section_start == -1:
                # URL 찾기 및 하이퍼링크로 변환
                matches = self._url_pattern.findall(text)
                for url in matches:
                    text = text.replace(url, f'<a href="{url}" target="_blank">{url}</a>')
                return text

            # 텍스트를 메인 내용과 참고문헌 섹션으로 분할
            main_content = text[:reference_section_start]
            reference_section = text[reference_section_start:]

            # 메인 내용의 URL만 처리
            matches = self._url_pattern.findall(main_content)
            for url in matches:
                main_content = main_content.replace(url, f'<a href="{url}" target="_blank">{url}</a>')

            # 메인 내용과 참고문헌 섹션 결합
            return main_content + reference_section

        except Exception as e:
            logger.error(f"URL을 하이퍼링크로 변환 중 오류 발생: {str(e)}")
            return text  # 오류 시 원본 반환

    def is_valid_url(self, url: str) -> bool:
        """
        URL이 유효한지 확인합니다.

        Args:
            url: 확인할 URL

        Returns:
            bool: URL이 유효하면 True, 그렇지 않으면 False
        """
        if not url:
            return False

        try:
            return bool(self._url_pattern.match(url))
        except Exception as e:
            logger.error(f"URL 유효성 검사 중 오류 발생: {str(e)}")
            return False
