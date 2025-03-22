"""
쿼리 프로세서 팩토리 모듈

설정에 따라 적절한 쿼리 프로세서 인스턴스를 생성하는 팩토리를 제공합니다.
"""

import logging
from typing import Dict, Type, Any

from src.services.query_processor.base import QueryProcessorBase
from src.services.query_processor.processors.standard_processor import StandardQueryProcessor
from src.services.query_processor.processors.faq_processor import FAQQueryProcessor
from src.services.query_processor.processors.pattern_processor import PatternQueryProcessor
from src.services.query_processor.exceptions import ProcessorNotFoundError

# 로거 설정
logger = logging.getLogger(__name__)


class ProcessorFactory:
    """
    쿼리 프로세서 팩토리 클래스

    설정에 따라 적절한 쿼리 프로세서 인스턴스를 생성합니다.
    """

    _registry: Dict[str, Type[QueryProcessorBase]] = {
        "standard": StandardQueryProcessor,
        "faq": FAQQueryProcessor,
        "pattern": PatternQueryProcessor,
    }

    _instances: Dict[str, QueryProcessorBase] = {}

    @classmethod
    def register(cls, name: str, processor_class: Type[QueryProcessorBase]) -> None:
        """
        새로운 쿼리 프로세서를 등록합니다.

        Args:
            name: 프로세서 식별자
            processor_class: 프로세서 클래스
        """
        cls._registry[name.lower()] = processor_class
        logger.info(f"쿼리 프로세서 '{name}'을(를) 등록했습니다")

    @classmethod
    def create(cls, processor_type: str = "standard", settings: Any = None,
               query_check_json_dict=None) -> QueryProcessorBase:
        """
        지정된 유형의 쿼리 프로세서를 생성합니다.

        Args:
            processor_type: 프로세서 유형
            settings: 설정 객체
            query_check_json_dict: 쿼리 패턴 사전

        Returns:
            QueryProcessorBase: 생성된 쿼리 프로세서 인스턴스
        """
        processor_type = processor_type.lower()

        # 인스턴스 캐싱을 위한 키 생성
        instance_key = f"{processor_type}_{id(settings)}"

        # 이미 생성된 인스턴스가 있는지 확인
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # 등록된 프로세서 클래스 확인
        if processor_type not in cls._registry:
            logger.error(f"알 수 없는 쿼리 프로세서 유형: {processor_type}")
            processor_type = "standard"  # 기본값으로 대체

        # 프로세서 인스턴스 생성
        processor_class = cls._registry[processor_type]
        instance = processor_class(settings, query_check_json_dict)

        # 인스턴스 캐싱
        cls._instances[instance_key] = instance

        logger.debug(f"'{processor_type}' 쿼리 프로세서를 생성했습니다")
        return instance

    @classmethod
    def clear_instances(cls) -> None:
        """
        캐시된 모든 인스턴스를 제거합니다.
        설정이 변경되었을 때 호출하세요.
        """
        cls._instances.clear()
        logger.debug("모든 프로세서 인스턴스가 제거되었습니다")
