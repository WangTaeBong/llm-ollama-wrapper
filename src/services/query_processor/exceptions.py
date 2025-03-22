"""
쿼리 프로세서 예외 모듈

쿼리 프로세서 관련 예외 클래스를 제공합니다.
"""


class QueryProcessorError(Exception):
    """쿼리 프로세서 관련 기본 예외"""
    pass


class ProcessorNotFoundError(QueryProcessorError):
    """지정된 쿼리 프로세서를 찾을 수 없는 경우의 예외"""
    pass


class PatternMatchError(QueryProcessorError):
    """패턴 매칭 관련 예외"""
    pass


class FilterError(QueryProcessorError):
    """쿼리 필터링 관련 예외"""
    pass
