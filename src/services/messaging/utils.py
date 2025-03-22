"""
메시지 유틸리티 모듈

채팅 메시지 처리를 위한 유틸리티 함수를 제공합니다.
"""

from datetime import datetime
from functools import lru_cache


@lru_cache(maxsize=1)
def get_datetime_format() -> str:
    """
    날짜/시간 형식을 반환합니다.

    캐싱을 통해 반복 호출 성능을 최적화합니다.

    Returns:
        str: 날짜/시간 형식 문자열
    """
    return "%Y-%m-%dT%H:%M:%S.%fZ"


def generate_timestamp() -> str:
    """
    현재 시간의 ISO 8601 형식 타임스탬프를 생성합니다.

    Returns:
        str: ISO 8601 형식의 현재 시간
    """
    return datetime.now().strftime(get_datetime_format())


def parse_timestamp(timestamp_str: str) -> datetime:
    """
    문자열 타임스탬프를 datetime 객체로 변환합니다.

    Args:
        timestamp_str: 변환할 타임스탬프 문자열

    Returns:
        datetime: 변환된 datetime 객체
    """
    return datetime.strptime(timestamp_str, get_datetime_format())
