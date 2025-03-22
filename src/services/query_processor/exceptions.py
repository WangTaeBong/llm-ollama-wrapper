"""
쿼리 프로세서 예외 모듈

쿼리 프로세서 관련 예외 클래스를 제공합니다.
"""


class QueryProcessorError(Exception):
    """쿼리 프로세서 관련 기본 예외"""

    def __init__(self, message="쿼리 처리 중 오류가 발생했습니다", code=None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ProcessorNotFoundError(QueryProcessorError):
    """지정된 쿼리 프로세서를 찾을 수 없는 경우의 예외"""

    def __init__(self, processor_type):
        super().__init__(f"쿼리 프로세서 유형 '{processor_type}'을(를) 찾을 수 없습니다", "PROCESSOR_NOT_FOUND")


class ConfigurationError(QueryProcessorError):
    """설정 관련 오류"""

    def __init__(self, message="쿼리 프로세서 설정이 올바르지 않습니다"):
        super().__init__(message, "CONFIGURATION_ERROR")


class PatternMatchError(QueryProcessorError):
    """패턴 매칭 관련 예외"""

    def __init__(self, message="패턴 매칭 중 오류가 발생했습니다"):
        super().__init__(message, "PATTERN_MATCH_ERROR")
