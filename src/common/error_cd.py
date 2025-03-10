from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


@dataclass(frozen=True)
class ErrorInfo:
    """
    Stores error information as a data class.

    Attributes:
        code (int): Error code
        desc (str): Error description
    """
    code: int
    desc: str

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts error information to dictionary format.

        Returns:
            Dict[str, Any]: Dictionary containing error code and description
        """
        return {"code": self.code, "desc": self.desc}


class ErrorCd(Enum):
    """
    Enum class for managing error codes and descriptions.

    Each error type is defined as an ErrorInfo instance,
    maintaining codes and descriptions in a consistent manner.
    """
    # Success related codes (2xx)
    SUCCESS = ErrorInfo(code=200, desc="Success")
    SUCCESS_NO_DATA = ErrorInfo(code=205, desc="Success (No data available)")
    SUCCESS_NO_EXIST_KNOWLEDGE_DATA = ErrorInfo(code=210, desc="Success (Knowledge does not exist)")

    # Error related codes (4xx, 5xx)
    CHAT_EXCEPTION = ErrorInfo(code=480, desc="An error occurred during the chat services")
    COMMON_EXCEPTION = ErrorInfo(code=500, desc="Internal Server Error")

    @property
    def code(self) -> int:
        """
        Returns the error code.

        Returns:
            int: Error code
        """
        return self.value.code

    @property
    def description(self) -> str:
        """
        Returns the error description.

        Returns:
            str: Error description
        """
        return self.value.desc

    @property
    def error_dict(self) -> Dict[str, Any]:
        """
        Returns error information as a dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing error code and description
        """
        return self.value.to_dict()

    @classmethod
    def get_code(cls, error_type: 'ErrorCd') -> int:
        """
        Returns the code of a given ErrorCd member.

        Args:
            error_type (ErrorCd): ErrorCd member

        Returns:
            int: Error code

        Example:
            >>> ErrorCd.get_code(ErrorCd.SUCCESS)
            200
        """
        return error_type.code

    @classmethod
    def get_description(cls, error_type: 'ErrorCd') -> str:
        """
        Returns the description of a given ErrorCd member.

        Args:
            error_type (ErrorCd): ErrorCd member

        Returns:
            str: Error description

        Example:
            >>> ErrorCd.get_description(ErrorCd.SUCCESS)
            'Success'
        """
        return error_type.description

    @classmethod
    def get_error(cls, error_type: 'ErrorCd') -> Dict[str, Any]:
        """
        Returns the complete error information (code and description) for a given ErrorCd member.

        Args:
            error_type (ErrorCd): ErrorCd member

        Returns:
            Dict[str, Any]: Dictionary containing error code and description

        Example:
            >>> ErrorCd.get_error(ErrorCd.SUCCESS)
            {'code': 200, 'desc': 'Success'}
        """
        return error_type.error_dict
