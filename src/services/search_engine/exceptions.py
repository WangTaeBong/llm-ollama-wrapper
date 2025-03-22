"""
검색 엔진 예외 모듈

검색 엔진 관련 예외 클래스를 제공합니다.
"""


class SearchEngineError(Exception):
    """검색 엔진 관련 기본 예외"""
    pass


class EngineNotFoundError(SearchEngineError):
    """지정된 검색 엔진을 찾을 수 없는 경우의 예외"""
    pass


class SearchTimeoutError(SearchEngineError):
    """검색 시간 초과 예외"""
    pass


class SearchQueryError(SearchEngineError):
    """검색 쿼리 관련 예외"""
    pass
