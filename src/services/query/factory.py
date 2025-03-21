"""
쿼리 처리기 팩토리 모듈
===================

설정에 따라 적절한 쿼리 처리기 구현체를 생성하는 팩토리 패턴을 구현합니다.
"""

import logging
from typing import Dict, Type, Optional, Any

from src.services.query.base import QueryProcessorBase

# 로거 설정
logger = logging.getLogger(__name__)


class QueryProcessorFactory:
    """
    쿼리 처리기 팩토리 클래스

    설정에 따라 적절한 쿼리 처리기 구현체를 생성하고 관리합니다.
    """

    # 사용 가능한 처리기 매핑
    _processor_registry: Dict[str, Type[QueryProcessorBase]] = {}

    # 활성화된 처리기 인스턴스
    _active_processor: Optional[QueryProcessorBase] = None

    @classmethod
    def register_processor(cls, name: str, processor_class: Type[QueryProcessorBase]) -> None:
        """
        쿼리 처리기 구현체 등록

        Args:
            name: 처리기 이름
            processor_class: 쿼리 처리기 클래스
        """
        if not issubclass(processor_class, QueryProcessorBase):
            raise TypeError(f"{processor_class.__name__}는 QueryProcessorBase를 상속받아야 합니다.")

        cls._processor_registry[name.lower()] = processor_class
        logger.debug(f"쿼리 처리기 '{name}' 등록 완료")

    @classmethod
    async def create_processor(cls, settings: Any, query_check_dict: Any,
                               processor_type: Optional[str] = None) -> QueryProcessorBase:
        """
        설정 기반으로 쿼리 처리기 생성

        Args:
            settings: 설정 객체
            query_check_dict: 쿼리 체크 딕셔너리
            processor_type: 처리기 타입 (설정에서 읽지 않고 강제 지정할 경우)

        Returns:
            QueryProcessorBase: 생성된 쿼리 처리기
        """
        # 이미 활성화된 처리기가 있으면 재사용
        if cls._active_processor is not None:
            return cls._active_processor

        # 처리기 타입 결정
        processor_name = processor_type or getattr(settings.query, 'processor_type', 'standard').lower()

        # 지원되는 처리기인지 확인
        if processor_name not in cls._processor_registry:
            supported = ", ".join(cls._processor_registry.keys())
            raise ValueError(
                f"'{processor_name}'은 지원되지 않는 쿼리 처리기입니다. 지원되는 처리기: {supported}"
            )

        # 처리기 생성 및 초기화
        processor_class = cls._processor_registry[processor_name]
        processor = processor_class(settings, query_check_dict)

        # 비동기 초기화 수행
        is_initialized = await processor.initialize()
        if not is_initialized:
            raise RuntimeError(f"{processor_name} 쿼리 처리기 초기화에 실패했습니다.")

        # 활성화된 처리기로 설정
        cls._active_processor = processor
        logger.info(f"{processor_name} 쿼리 처리기 생성 및 초기화 완료")

        return processor
