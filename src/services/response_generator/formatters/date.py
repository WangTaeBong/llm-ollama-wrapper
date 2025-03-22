# src/services/response_generator/formatters/date.py
"""
날짜 포맷 모듈

날짜 및 시간 서식 지정 기능을 제공합니다.
"""

import logging
from datetime import datetime
from functools import lru_cache
from time import localtime

# 모듈 로거 설정
logger = logging.getLogger(__name__)


class DateFormatter:
    """
    날짜 포맷 클래스

    날짜와 시간의 지역화 및 포맷팅을 처리합니다.
    """

    def __init__(self):
        """
        DateFormatter 초기화
        """
        # 요일 이름 정의
        self._day_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']

    @lru_cache(maxsize=10)
    def get_formatted_date(self, format_type: str = 'full') -> str:
        """
        현재 날짜와 시간을 지정된 형식으로 포맷팅합니다.

        Args:
            format_type: 날짜 형식 유형('full', 'date_only', 'time_only')

        Returns:
            str: 포맷팅된 날짜 문자열
        """
        try:
            today = datetime.now()
            weekday = self._day_names[localtime().tm_wday]

            if format_type == 'full':
                return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."
            elif format_type == 'date_only':
                return f"{today.strftime('%Y년 %m월 %d일')} {weekday}"
            elif format_type == 'time_only':
                return f"{today.strftime('%H시 %M분')}"
            else:
                return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."
        except Exception as e:
            logger.warning(f"날짜 포맷팅 중 오류: {e}")
            # 더 간단한 형식으로 폴백
            return datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분')

    @classmethod
    def get_time_description(cls, time_context: str = 'now') -> str:
        """
        특정 시간 문맥에 대한 설명을 생성합니다.

        Args:
            time_context: 시간 문맥('now', 'morning', 'afternoon', 'evening')

        Returns:
            str: 시간 문맥 설명
        """
        try:
            current_hour = datetime.now().hour

            if time_context == 'morning' or (time_context == 'now' and 5 <= current_hour < 12):
                return "좋은 아침입니다"
            elif time_context == 'afternoon' or (time_context == 'now' and 12 <= current_hour < 18):
                return "좋은 오후입니다"
            elif time_context == 'evening' or (time_context == 'now' and (18 <= current_hour or current_hour < 5)):
                return "좋은 저녁입니다"
            else:
                return "안녕하세요"
        except Exception as e:
            logger.warning(f"시간 설명 생성 중 오류: {e}")
            return "안녕하세요"
