"""
문서 처리 예외 모듈

문서 처리와 관련된 예외 클래스들을 제공합니다.
"""


class DocumentProcessorError(Exception):
    """문서 처리 기본 예외 클래스"""

    def __init__(self, message="문서 처리 중 오류가 발생했습니다", code=None):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DocumentConversionError(DocumentProcessorError):
    """문서 변환 중 발생하는 예외"""

    def __init__(self, message="문서 변환 중 오류가 발생했습니다", code="CONVERSION_ERROR"):
        super().__init__(message, code)


class EmptyDocumentError(DocumentProcessorError):
    """빈 문서 또는 문서 없음 예외"""

    def __init__(self, message="문서가 비어 있거나 존재하지 않습니다", code="EMPTY_DOCUMENT"):
        super().__init__(message, code)


class InvalidDocumentFormatError(DocumentProcessorError):
    """유효하지 않은 문서 형식 예외"""

    def __init__(self, message="유효하지 않은 문서 형식입니다", code="INVALID_FORMAT"):
        super().__init__(message, code)


class ConfigurationError(DocumentProcessorError):
    """설정 오류 예외"""

    def __init__(self, message="문서 처리 설정이 올바르지 않습니다", code="CONFIGURATION_ERROR"):
        super().__init__(message, code)
