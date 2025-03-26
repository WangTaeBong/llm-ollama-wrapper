import logging
from typing import Dict, Any, Optional
from src.common.error_cd import ErrorCd
from src.schema.chat_res import ChatResponse, MetaRes, ChatRes


class ErrorHandler:
    def __init__(self, logger=None):
        """
        ErrorHandler 초기화

        Args:
            logger: 로깅을 위한 로거 객체 (미제공 시 기본 로거 사용)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.metrics = {
            "error_count": 0,
            "error_types": {}
        }

    def handle_error(
            self,
            error: Exception,
            session_id: str,
            request=None,
            error_type: str = 'GENERAL_ERROR'
    ) -> ChatResponse:
        """
        공통 에러 처리 메서드

        Args:
            error: 발생한 예외 객체
            session_id: 세션 식별자
            request: 원본 요청 객체 (선택적)
            error_type: 에러 유형 식별자

        Returns:
            ChatResponse: 오류 응답 객체
        """
        # 에러 로깅
        self.logger.error(
            f"[{session_id}] Error occurred: {str(error)}",
            exc_info=True
        )

        # 메트릭 업데이트
        self._update_error_metrics(error)

        # 에러 코드 선택
        error_code = self._get_error_code(error, error_type)

        # 에러 응답 생성
        return self._create_error_response(
            error_code,
            str(error),
            session_id,
            request
        )

    def _update_error_metrics(self, error: Exception):
        """
        에러 메트릭 업데이트

        Args:
            error: 발생한 예외 객체
        """
        self.metrics["error_count"] += 1
        error_type = type(error).__name__
        self.metrics["error_types"][error_type] = \
            self.metrics["error_types"].get(error_type, 0) + 1

    def _get_error_code(self, error: Exception, error_type: str) -> Dict[str, str]:
        """
        예외 유형에 따른 에러 코드 선택

        Args:
            error: 발생한 예외 객체
            error_type: 에러 유형 식별자

        Returns:
            Dict[str, str]: 에러 코드 정보
        """
        error_map = {
            'ValueError': ErrorCd.INVALID_INPUT,
            'KeyError': ErrorCd.MISSING_KEY,
            'TimeoutError': ErrorCd.REQUEST_TIMEOUT,
            'ConnectionError': ErrorCd.CONNECTION_ERROR,
            'GENERAL_ERROR': ErrorCd.CHAT_EXCEPTION
        }

        return ErrorCd.get_error(
            error_map.get(type(error).__name__, error_map.get(error_type, ErrorCd.CHAT_EXCEPTION))
        )

    def _create_error_response(
            self,
            error_code: Dict[str, str],
            error_message: str,
            session_id: str,
            request=None
    ) -> ChatResponse:
        """
        오류 응답 객체 생성

        Args:
            error_code: 에러 코드 정보
            error_message: 오류 메시지
            session_id: 세션 식별자
            request: 원본 요청 객체 (선택적)

        Returns:
            ChatResponse: 오류 응답 객체
        """
        # 요청 객체에서 메타데이터 추출
        company_id = getattr(request.meta, 'company_id', '') if request and hasattr(request, 'meta') else ''
        dept_class = getattr(request.meta, 'dept_class', '') if request and hasattr(request, 'meta') else ''
        rag_sys_info = getattr(request.meta, 'rag_sys_info', '') if request and hasattr(request, 'meta') else ''
        user_query = getattr(request.chat, 'user', '') if request and hasattr(request, 'chat') else ''

        return ChatResponse(
            result_cd=safe_get_error_code(error_code),
            result_desc=error_code.get("desc"),
            meta=MetaRes(
                company_id=company_id,
                dept_class=dept_class,
                session_id=session_id,
                rag_sys_info=rag_sys_info,
            ),
            chat=ChatRes(
                user=user_query,
                system=f"Error: {error_message}",
                category1='',
                category2='',
                category3='',
                info=[]
            )
        )


def safe_get_error_code(error_code_dict: Dict[str, Any]) -> int:
    """
    안전하게 에러 코드를 정수로 변환합니다.

    Args:
        error_code_dict: 에러 코드 딕셔너리

    Returns:
        int: 변환된 에러 코드 (변환 불가능시 기본값 0)
    """
    try:
        return int(error_code_dict.get("code", 0))
    except (TypeError, ValueError):
        return 0
