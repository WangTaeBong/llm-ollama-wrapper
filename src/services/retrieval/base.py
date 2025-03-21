"""
검색 시스템 기본 인터페이스 모듈
===============================

이 모듈은 다양한 검색 소스에 대한 표준 인터페이스를 정의합니다.
모든 검색 소스 구현체는 이 기본 클래스를 상속받아 구현해야 합니다.

기능:
- 동기 및 비동기 문서 검색 인터페이스
- 문서 처리 및 캐싱 지원
- 검색 메트릭 관리
"""

import abc
import logging
from typing import Dict, List, Any

from langchain_core.documents import Document

# 로거 설정
logger = logging.getLogger(__name__)


class RetrievalSourceBase(abc.ABC):
    """
    검색 소스 기본 인터페이스

    모든 검색 소스 구현은 이 클래스를 상속받아 구현해야 합니다.
    """

    def __init__(self, settings: Any):
        """
        검색 소스 초기화

        Args:
            settings: 설정 객체
        """
        self.settings = settings
        self.session_id = None

        # 메트릭 초기화
        self.metrics = {
            "request_count": 0,
            "total_time": 0,
            "error_count": 0,
            "document_count": 0
        }

    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        검색 소스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        pass

    @abc.abstractmethod
    async def retrieve(self, query: str, **kwargs) -> List[Document]:
        """
        주어진 쿼리에 대한 문서 검색

        Args:
            query: 검색 쿼리
            **kwargs: 추가 검색 매개변수

        Returns:
            List[Document]: 검색된 문서 리스트
        """
        pass

    @abc.abstractmethod
    def get_name(self) -> str:
        """
        검색 소스 이름 반환

        Returns:
            str: 검색 소스 이름
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        검색 메트릭 반환

        Returns:
            Dict[str, Any]: 메트릭 정보
        """
        avg_time = 0
        if self.metrics["request_count"] > 0:
            avg_time = self.metrics["total_time"] / self.metrics["request_count"]

        return {
            "request_count": self.metrics["request_count"],
            "error_count": self.metrics["error_count"],
            "avg_response_time": avg_time,
            "total_time": self.metrics["total_time"],
            "document_count": self.metrics["document_count"]
        }
   