"""
문서 프로세서 팩토리 모듈

설정에 따라 적절한 문서 프로세서 인스턴스를 생성하는 팩토리를 제공합니다.
"""

import logging
from typing import Dict, Any, Type

from src.services.document_processor.base import DocumentProcessorBase
from src.services.document_processor.exceptions import ConfigurationError
from src.services.document_processor.standard_processor import StandardDocumentProcessor

# 로거 설정
logger = logging.getLogger(__name__)


class DocumentProcessorFactory:
    """
    문서 프로세서 팩토리 클래스

    설정에 따라 적절한 문서 프로세서 인스턴스를 생성합니다.
    """

    _registry: Dict[str, Type[DocumentProcessorBase]] = {
        "standard": StandardDocumentProcessor,
    }

    _instances: Dict[str, DocumentProcessorBase] = {}

    @classmethod
    def register(cls, name: str, processor_class: Type[DocumentProcessorBase]) -> None:
        """
        새로운 문서 프로세서를 등록합니다.

        Args:
            name: 프로세서 식별자
            processor_class: 프로세서 클래스
        """
        cls._registry[name.lower()] = processor_class
        logger.info(f"문서 프로세서 '{name}'을(를) 등록했습니다")

    @classmethod
    def create(cls, processor_type: str = "standard", settings: Any = None) -> DocumentProcessorBase:
        """
        지정된 유형의 문서 프로세서를 생성합니다.

        Args:
            processor_type: 프로세서 유형
            settings: 설정 객체

        Returns:
            DocumentProcessorBase: 생성된 문서 프로세서 인스턴스
        """
        processor_type = processor_type.lower()

        # 인스턴스 캐싱을 위한 키 생성
        instance_key = f"{processor_type}_{id(settings)}"

        # 이미 생성된 인스턴스가 있는지 확인
        if instance_key in cls._instances:
            return cls._instances[instance_key]

        # 등록된 프로세서 클래스 확인
        if processor_type not in cls._registry:
            logger.warning(f"알 수 없는 문서 프로세서 유형: {processor_type}, 기본값 사용")
            processor_type = "standard"  # 기본값으로 대체

        # 프로세서 인스턴스 생성
        processor_class = cls._registry[processor_type]

        if not settings:
            raise ConfigurationError("문서 프로세서 생성에 설정 객체가 필요합니다")

        instance = processor_class(settings)

        # 인스턴스 캐싱
        cls._instances[instance_key] = instance

        logger.debug(f"'{processor_type}' 문서 프로세서를 생성했습니다")
        return instance

    @classmethod
    def clear_instances(cls) -> None:
        """
        캐시된 모든 인스턴스를 제거합니다.
        설정이 변경되었을 때 호출하세요.
        """
        cls._instances.clear()
        logger.debug("모든 프로세서 인스턴스가 제거되었습니다")
