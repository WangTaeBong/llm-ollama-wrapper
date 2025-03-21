"""
검색 시스템 팩토리 모듈
=====================

이 모듈은 설정에 따라 적절한 검색 소스 구현체를 생성하는 팩토리 패턴을 구현합니다.
의존성 주입 원칙에 따라 필요한 구성 요소를 검색 소스에 제공합니다.

기능:
- 설정 기반 검색 소스 인스턴스 생성
- 검색 소스 구현체 관리
- 소스 상태 모니터링
"""

import logging
from typing import Dict, Type, Optional, Any

from .base import RetrievalSourceBase

# 로거 설정
logger = logging.getLogger(__name__)


class RetrievalSourceFactory:
    """
    검색 소스 팩토리 클래스

    설정에 따라 적절한 검색 소스 구현체를 생성하고 관리합니다.
    """

    # 사용 가능한 검색 소스 매핑
    _source_registry: Dict[str, Type[RetrievalSourceBase]] = {}

    # 활성화된 소스 인스턴스 (싱글톤)
    _active_source: Optional[RetrievalSourceBase] = None

    @classmethod
    def register_source(cls, name: str, source_class: Type[RetrievalSourceBase]) -> None:
        """
        검색 소스 구현체 등록

        Args:
            name: 소스 이름
            source_class: 검색 소스 클래스
        """
        if not issubclass(source_class, RetrievalSourceBase):
            raise TypeError(f"{source_class.__name__}는 RetrievalSourceBase를 상속받아야 합니다.")

        cls._source_registry[name.lower()] = source_class
        logger.debug(f"검색 소스 '{name}' 등록 완료")

    @classmethod
    async def create_source(cls, settings: Any, source_type: Optional[str] = None) -> RetrievalSourceBase:
        """
        설정 기반으로 검색 소스 생성

        Args:
            settings: 설정 객체
            source_type: 검색 소스 타입 (설정에서 읽지 않고 강제 지정할 경우)

        Returns:
            RetrievalSourceBase: 생성된 검색 소스

        Raises:
            ValueError: 지원되지 않는 검색 소스 요청 시
        """
        # 이미 활성화된 소스가 있으면 재사용
        if cls._active_source is not None:
            return cls._active_source

        # 명시적 소스 타입 또는 설정에서 타입 읽기
        source_name = source_type or getattr(settings.retriever, 'source_type', 'hybrid').lower()

        if not source_name:
            raise ValueError("검색 소스 타입이 설정되지 않았습니다.")

        # 지원되는 소스인지 확인
        if source_name not in cls._source_registry:
            supported = ", ".join(cls._source_registry.keys())
            raise ValueError(
                f"'{source_name}'은 지원되지 않는 검색 소스입니다. 지원되는 소스: {supported}"
            )

        # 소스 생성 및 초기화
        source_class = cls._source_registry[source_name]
        source = source_class(settings)

        # 비동기 초기화 수행
        is_initialized = await source.initialize()
        if not is_initialized:
            raise RuntimeError(f"{source_name} 검색 소스 초기화에 실패했습니다.")

        # 활성화된 소스로 설정
        cls._active_source = source
        logger.info(f"{source_name} 검색 소스 생성 및 초기화 완료")

        return source

    @classmethod
    def get_active_source(cls) -> Optional[RetrievalSourceBase]:
        """
        현재 활성화된 검색 소스 반환

        Returns:
            Optional[RetrievalSourceBase]: 활성화된 소스 또는 None
        """
        return cls._active_source

    @classmethod
    def reset(cls) -> None:
        """
        활성화된 소스 리셋 (주로 테스트용)
        """
        cls._active_source = None
        logger.debug("검색 소스 팩토리 리셋 완료")
