"""
LLM 서비스 팩토리 모듈

다양한 LLM 백엔드(Ollama, vLLM, Gemma)를 위한 핸들러와 관련 객체들을 생성하는
팩토리 패턴을 구현합니다. 이 모듈은 중앙 집중식 객체 생성 메커니즘을 제공하여
의존성 관리와 객체 수명 주기를 단순화합니다.

주요 기능:
- LLM 핸들러 인스턴스 생성 및 캐싱
- 모델 유형 자동 감지 및 적절한 핸들러 선택
- 싱글톤 패턴을 통한 리소스 관리
- 타입별 인터페이스 일관성 보장
"""

import logging
from typing import Dict, Type, Any, Optional, Union, List
import asyncio
from threading import Lock

from src.schema.chat_req import ChatRequest
from src.common.config_loader import ConfigLoader

# 설정 로드 - 싱글톤으로 관리되는 설정 객체
settings = ConfigLoader().get_settings()

# 로거 설정
logger = logging.getLogger(__name__)


class LLMServiceFactory:
    """
    LLM 서비스 팩토리 클래스

    다양한 LLM 백엔드를 위한 핸들러와 서비스 객체를 생성하고 관리합니다.
    싱글톤 패턴을 사용하여 애플리케이션 전체에서 일관된 객체 인스턴스를 제공합니다.
    """

    # 클래스 변수 - 싱글톤 인스턴스
    _instance = None
    _lock = Lock()

    # 핸들러 레지스트리 - 모델 유형별 핸들러 클래스 매핑
    _handler_registry = {}

    # 인스턴스 캐시 - 생성된 핸들러 인스턴스 저장
    _handler_instances = {}

    # 백그라운드 태스크 관리
    _background_tasks = []

    @classmethod
    def get_instance(cls):
        """
        팩토리의 싱글톤 인스턴스를 반환합니다.

        Returns:
            LLMServiceFactory: 팩토리 싱글톤 인스턴스
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
                # 핸들러 레지스트리 초기화
                cls._initialize_registry()
            return cls._instance

    @classmethod
    def _initialize_registry(cls):
        """
        핸들러 레지스트리를 초기화합니다.
        동적 임포트를 사용하여 핸들러 클래스를 등록합니다.
        """
        try:
            # 핸들러 클래스 동적 임포트
            from src.services.llm.handlers.base import BaseLLMHandler
            from src.services.llm.handlers.ollama_handler import OllamaHandler
            from src.services.llm.handlers.vllm_handler import VLLMHandler
            from src.services.llm.handlers.gemma_handler import GemmaHandler

            # 레지스트리에 핸들러 클래스 등록
            cls._handler_registry = {
                "base": BaseLLMHandler,
                "ollama": OllamaHandler,
                "vllm": VLLMHandler,
                "gemma": GemmaHandler
            }
            logger.info("LLM 핸들러 레지스트리 초기화 완료")
        except ImportError as e:
            logger.error(f"핸들러 클래스 임포트 중 오류 발생: {str(e)}")
            # 임시 조치: 기존 구현체로부터 핸들러 임포트
            try:
                from src.services.history.handlers.history_handler import BaseHistoryHandler
                from src.services.history.handlers.ollama_handler import OllamaHistoryHandler
                from src.services.history.handlers.vllm_handler import VLLMHistoryHandler
                from src.services.history.handlers.gemma_handler import GemmaHistoryHandler

                cls._handler_registry = {
                    "base": BaseHistoryHandler,
                    "ollama": OllamaHistoryHandler,
                    "vllm": VLLMHistoryHandler,
                    "gemma": GemmaHistoryHandler
                }
                logger.warning("레거시 LLM 핸들러를 사용하여 레지스트리 초기화")
            except ImportError as e2:
                logger.critical(f"레거시 핸들러도 임포트할 수 없음: {str(e2)}")
                # 기본 레지스트리는 빈 상태로 유지됨

    def create_handler(
            self,
            llm_model: Any,
            request: ChatRequest,
            max_history_turns: int = 10,
            force_new: bool = False
    ) -> Any:
        """
        요청과 모델 유형에 적합한 LLM 핸들러 인스턴스를 생성하거나 캐시에서 가져옵니다.

        Args:
            llm_model: LLM 모델 인스턴스
            request: 채팅 요청 객체
            max_history_turns: 최대 히스토리 턴 수
            force_new: 새 인스턴스를 강제로 생성할지 여부

        Returns:
            Any: 생성된 LLM 핸들러 인스턴스
        """
        # 모델 유형 결정 (Gemma, vLLM, Ollama)
        model_type = self._determine_model_type(request)
        cache_key = f"{model_type}:{request.meta.rag_sys_info}:{request.meta.session_id}"

        # 캐시에서 인스턴스 확인 (force_new가 False인 경우에만)
        if not force_new and cache_key in self._handler_instances:
            logger.debug(f"[{request.meta.session_id}] 캐시된 {model_type} 핸들러 사용")
            return self._handler_instances[cache_key]

        # 핸들러 클래스 가져오기
        handler_class = self._handler_registry.get(model_type)
        if not handler_class:
            logger.warning(f"[{request.meta.session_id}] 알 수 없는 모델 유형: {model_type}, 기본 핸들러 사용")
            handler_class = self._handler_registry.get("base")

            # 기본 핸들러도 없는 경우 예외 발생
            if not handler_class:
                logger.error("기본 핸들러 클래스를 찾을 수 없음")
                raise ValueError("LLM 핸들러를 생성할 수 없음: 유효한 핸들러 클래스가 없음")

        # 새 인스턴스 생성
        logger.debug(f"[{request.meta.session_id}] 새 {model_type} 핸들러 생성")
        handler = handler_class(
            llm_model=llm_model,
            request=request,
            max_history_turns=max_history_turns
        )

        # 캐시에 저장
        self._handler_instances[cache_key] = handler
        return handler

    def _determine_model_type(self, request: ChatRequest) -> str:
        """
        요청 데이터와 설정에서 모델 유형을 결정합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            str: 모델 유형 ("gemma", "vllm", "ollama", "base" 중 하나)
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # Ollama 백엔드 확인
        if backend == 'ollama':
            return "ollama"

        # vLLM 백엔드인 경우 Gemma 모델 확인
        elif backend == 'vllm':
            if self._is_gemma_model():
                logger.debug(f"[{request.meta.session_id}] Gemma 모델 감지됨")
                return "gemma"
            else:
                return "vllm"

        # 기본값 반환
        logger.warning(f"[{request.meta.session_id}] 알 수 없는 백엔드: {backend}, 기본 핸들러 사용")
        return "base"

    @classmethod
    def _is_gemma_model(cls) -> bool:
        """
        현재 설정된 모델이 Gemma 모델인지 확인합니다.

        Returns:
            bool: Gemma 모델이면 True, 아니면 False
        """
        # LLM 백엔드 확인
        backend = getattr(settings.llm, 'llm_backend', '').lower()

        # OLLAMA 백엔드인 경우
        if backend == 'ollama':
            if hasattr(settings.ollama, 'model_name'):
                model_name = settings.ollama.model_name.lower()
                return 'gemma' in model_name

        # VLLM 백엔드인 경우
        elif backend == 'vllm':
            if hasattr(settings.llm, 'model_type'):
                model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                    settings.llm.model_type).lower()
                return model_type == 'gemma'

        # 기본적으로 False 반환
        return False

    def create_chat_service(self, request: ChatRequest) -> Any:
        """
        채팅 서비스 인스턴스를 생성합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            Any: 생성된 채팅 서비스 인스턴스
        """
        try:
            # 채팅 서비스 클래스 동적 임포트
            from src.services.llm.chat_service import ChatService
            return ChatService(request)
        except ImportError:
            # 임시 조치: 기존 구현체 사용
            from src.services.llm_ollama_process import ChatService
            return ChatService(request)

    def create_llm_service(self, request: ChatRequest) -> Any:
        """
        LLM 서비스 인스턴스를 생성합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            Any: 생성된 LLM 서비스 인스턴스
        """
        try:
            # LLM 서비스 클래스 동적 임포트
            from src.services.llm.service import LLMService
            return LLMService(request)
        except ImportError:
            # 임시 조치: 기존 구현체 사용
            from src.services.llm_ollama_process import LLMService
            return LLMService(request)

    def create_retriever_service(self, request: ChatRequest) -> Any:
        """
        검색기 서비스 인스턴스를 생성합니다.

        Args:
            request: 채팅 요청 객체

        Returns:
            Any: 생성된 검색기 서비스 인스턴스
        """
        try:
            # 검색기 서비스 클래스 동적 임포트
            from src.services.llm.retriever_service import RetrieverService
            return RetrieverService(request)
        except ImportError:
            # 임시 조치: 기존 구현체 사용
            from src.services.llm_ollama_process import RetrieverService
            return RetrieverService(request)

    def register_handler(self, name: str, handler_class: Type) -> None:
        """
        새로운 핸들러 클래스를 등록합니다.

        Args:
            name: 핸들러 이름
            handler_class: 핸들러 클래스
        """
        self._handler_registry[name.lower()] = handler_class
        logger.info(f"새 LLM 핸들러 등록됨: {name}")

    def clear_cache(self) -> None:
        """
        핸들러 캐시를 비웁니다.
        설정이 변경되었거나 메모리를 확보해야 할 때 유용합니다.
        """
        self._handler_instances.clear()
        logger.debug("LLM 핸들러 캐시가 비워졌습니다.")

    def get_handler_class(self, model_type: str) -> Optional[Type]:
        """
        지정된 모델 유형에 대한 핸들러 클래스를 반환합니다.

        Args:
            model_type: 모델 유형

        Returns:
            Optional[Type]: 핸들러 클래스 또는 None
        """
        return self._handler_registry.get(model_type.lower())

    """
    LLM 서비스 팩토리 모듈

    다양한 LLM 백엔드(Ollama, vLLM, Gemma)를 위한 핸들러와 관련 객체들을 생성하는
    팩토리 패턴을 구현합니다. 이 모듈은 중앙 집중식 객체 생성 메커니즘을 제공하여
    의존성 관리와 객체 수명 주기를 단순화합니다.

    주요 기능:
    - LLM 핸들러 인스턴스 생성 및 캐싱
    - 모델 유형 자동 감지 및 적절한 핸들러 선택
    - 싱글톤 패턴을 통한 리소스 관리
    - 타입별 인터페이스 일관성 보장
    """

    import logging
    from typing import Dict, Type, Any, Optional, Union, List
    import asyncio
    from threading import Lock

    from src.schema.chat_req import ChatRequest
    from src.common.config_loader import ConfigLoader

    # 설정 로드 - 싱글톤으로 관리되는 설정 객체
    settings = ConfigLoader().get_settings()

    # 로거 설정
    logger = logging.getLogger(__name__)

    class LLMServiceFactory:
        """
        LLM 서비스 팩토리 클래스

        다양한 LLM 백엔드를 위한 핸들러와 서비스 객체를 생성하고 관리합니다.
        싱글톤 패턴을 사용하여 애플리케이션 전체에서 일관된 객체 인스턴스를 제공합니다.
        """

        # 클래스 변수 - 싱글톤 인스턴스
        _instance = None
        _lock = Lock()

        # 핸들러 레지스트리 - 모델 유형별 핸들러 클래스 매핑
        _handler_registry = {}

        # 인스턴스 캐시 - 생성된 핸들러 인스턴스 저장
        _handler_instances = {}

        # 백그라운드 태스크 관리
        _background_tasks = []

        @classmethod
        def get_instance(cls):
            """
            팩토리의 싱글톤 인스턴스를 반환합니다.

            Returns:
                LLMServiceFactory: 팩토리 싱글톤 인스턴스
            """
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
                    # 핸들러 레지스트리 초기화
                    cls._initialize_registry()
                return cls._instance

        @classmethod
        def _initialize_registry(cls):
            """
            핸들러 레지스트리를 초기화합니다.
            동적 임포트를 사용하여 핸들러 클래스를 등록합니다.
            """
            try:
                # 핸들러 클래스 동적 임포트
                from src.services.llm.handlers.base import BaseLLMHandler
                from src.services.llm.handlers.ollama_handler import OllamaHandler
                from src.services.llm.handlers.vllm_handler import VLLMHandler
                from src.services.llm.handlers.gemma_handler import GemmaHandler

                # 레지스트리에 핸들러 클래스 등록
                cls._handler_registry = {
                    "base": BaseLLMHandler,
                    "ollama": OllamaHandler,
                    "vllm": VLLMHandler,
                    "gemma": GemmaHandler
                }
                logger.info("LLM 핸들러 레지스트리 초기화 완료")
            except ImportError as e:
                logger.error(f"핸들러 클래스 임포트 중 오류 발생: {str(e)}")
                # 임시 조치: 기존 구현체로부터 핸들러 임포트
                try:
                    from src.services.history.handlers.history_handler import BaseHistoryHandler
                    from src.services.history.handlers.ollama_handler import OllamaHistoryHandler
                    from src.services.history.handlers.vllm_handler import VLLMHistoryHandler
                    from src.services.history.handlers.gemma_handler import GemmaHistoryHandler

                    cls._handler_registry = {
                        "base": BaseHistoryHandler,
                        "ollama": OllamaHistoryHandler,
                        "vllm": VLLMHistoryHandler,
                        "gemma": GemmaHistoryHandler
                    }
                    logger.warning("레거시 LLM 핸들러를 사용하여 레지스트리 초기화")
                except ImportError as e2:
                    logger.critical(f"레거시 핸들러도 임포트할 수 없음: {str(e2)}")
                    # 기본 레지스트리는 빈 상태로 유지됨

        def create_handler(
                self,
                llm_model: Any,
                request: ChatRequest,
                max_history_turns: int = 10,
                force_new: bool = False
        ) -> Any:
            """
            요청과 모델 유형에 적합한 LLM 핸들러 인스턴스를 생성하거나 캐시에서 가져옵니다.

            Args:
                llm_model: LLM 모델 인스턴스
                request: 채팅 요청 객체
                max_history_turns: 최대 히스토리 턴 수
                force_new: 새 인스턴스를 강제로 생성할지 여부

            Returns:
                Any: 생성된 LLM 핸들러 인스턴스
            """
            # 모델 유형 결정 (Gemma, vLLM, Ollama)
            model_type = self._determine_model_type(request)
            cache_key = f"{model_type}:{request.meta.rag_sys_info}:{request.meta.session_id}"

            # 캐시에서 인스턴스 확인 (force_new가 False인 경우에만)
            if not force_new and cache_key in self._handler_instances:
                logger.debug(f"[{request.meta.session_id}] 캐시된 {model_type} 핸들러 사용")
                return self._handler_instances[cache_key]

            # 핸들러 클래스 가져오기
            handler_class = self._handler_registry.get(model_type)
            if not handler_class:
                logger.warning(f"[{request.meta.session_id}] 알 수 없는 모델 유형: {model_type}, 기본 핸들러 사용")
                handler_class = self._handler_registry.get("base")

                # 기본 핸들러도 없는 경우 예외 발생
                if not handler_class:
                    logger.error("기본 핸들러 클래스를 찾을 수 없음")
                    raise ValueError("LLM 핸들러를 생성할 수 없음: 유효한 핸들러 클래스가 없음")

            # 새 인스턴스 생성
            logger.debug(f"[{request.meta.session_id}] 새 {model_type} 핸들러 생성")
            handler = handler_class(
                llm_model=llm_model,
                request=request,
                max_history_turns=max_history_turns
            )

            # 캐시에 저장
            self._handler_instances[cache_key] = handler
            return handler

        def _determine_model_type(self, request: ChatRequest) -> str:
            """
            요청 데이터와 설정에서 모델 유형을 결정합니다.

            Args:
                request: 채팅 요청 객체

            Returns:
                str: 모델 유형 ("gemma", "vllm", "ollama", "base" 중 하나)
            """
            # LLM 백엔드 확인
            backend = getattr(settings.llm, 'llm_backend', '').lower()

            # Ollama 백엔드 확인
            if backend == 'ollama':
                return "ollama"

            # vLLM 백엔드인 경우 Gemma 모델 확인
            elif backend == 'vllm':
                if self._is_gemma_model():
                    logger.debug(f"[{request.meta.session_id}] Gemma 모델 감지됨")
                    return "gemma"
                else:
                    return "vllm"

            # 기본값 반환
            logger.warning(f"[{request.meta.session_id}] 알 수 없는 백엔드: {backend}, 기본 핸들러 사용")
            return "base"

        @classmethod
        def _is_gemma_model(cls) -> bool:
            """
            현재 설정된 모델이 Gemma 모델인지 확인합니다.

            Returns:
                bool: Gemma 모델이면 True, 아니면 False
            """
            # LLM 백엔드 확인
            backend = getattr(settings.llm, 'llm_backend', '').lower()

            # OLLAMA 백엔드인 경우
            if backend == 'ollama':
                if hasattr(settings.ollama, 'model_name'):
                    model_name = settings.ollama.model_name.lower()
                    return 'gemma' in model_name

            # VLLM 백엔드인 경우
            elif backend == 'vllm':
                if hasattr(settings.llm, 'model_type'):
                    model_type = settings.llm.model_type.lower() if hasattr(settings.llm.model_type, 'lower') else str(
                        settings.llm.model_type).lower()
                    return model_type == 'gemma'

            # 기본적으로 False 반환
            return False

        def create_chat_service(self, request: ChatRequest) -> Any:
            """
            채팅 서비스 인스턴스를 생성합니다.

            Args:
                request: 채팅 요청 객체

            Returns:
                Any: 생성된 채팅 서비스 인스턴스
            """
            try:
                # 채팅 서비스 클래스 동적 임포트
                from src.services.llm.chat_service import ChatService
                return ChatService(request)
            except ImportError:
                # 임시 조치: 기존 구현체 사용
                from src.services.llm_ollama_process import ChatService
                return ChatService(request)

        def create_llm_service(self, request: ChatRequest) -> Any:
            """
            LLM 서비스 인스턴스를 생성합니다.

            Args:
                request: 채팅 요청 객체

            Returns:
                Any: 생성된 LLM 서비스 인스턴스
            """
            try:
                # LLM 서비스 클래스 동적 임포트
                from src.services.llm.service import LLMService
                return LLMService(request)
            except ImportError:
                # 임시 조치: 기존 구현체 사용
                from src.services.llm_ollama_process import LLMService
                return LLMService(request)

        def create_retriever_service(self, request: ChatRequest) -> Any:
            """
            검색기 서비스 인스턴스를 생성합니다.

            Args:
                request: 채팅 요청 객체

            Returns:
                Any: 생성된 검색기 서비스 인스턴스
            """
            try:
                # 검색기 서비스 클래스 동적 임포트
                from src.services.llm.retriever_service import RetrieverService
                return RetrieverService(request)
            except ImportError:
                # 임시 조치: 기존 구현체 사용
                from src.services.llm_ollama_process import RetrieverService
                return RetrieverService(request)

        def register_handler(self, name: str, handler_class: Type) -> None:
            """
            새로운 핸들러 클래스를 등록합니다.

            Args:
                name: 핸들러 이름
                handler_class: 핸들러 클래스
            """
            self._handler_registry[name.lower()] = handler_class
            logger.info(f"새 LLM 핸들러 등록됨: {name}")

        def clear_cache(self) -> None:
            """
            핸들러 캐시를 비웁니다.
            설정이 변경되었거나 메모리를 확보해야 할 때 유용합니다.
            """
            self._handler_instances.clear()
            logger.debug("LLM 핸들러 캐시가 비워졌습니다.")

        def get_handler_class(self, model_type: str) -> Optional[Type]:
            """
            지정된 모델 유형에 대한 핸들러 클래스를 반환합니다.

            Args:
                model_type: 모델 유형

            Returns:
                Optional[Type]: 핸들러 클래스 또는 None
            """
            return self._handler_registry.get(model_type.lower())

        async def fire_and_forget(self, coro) -> None:
            """
            코루틴을 백그라운드에서 실행하고 오류를 로깅합니다.
            결과를 기다리지 않고 다른 작업을 계속 진행합니다.

            Args:
                coro: 실행할 코루틴
            """

            async def wrapper():
                try:
                    await coro
                except Exception as e:
                    logger.error(f"백그라운드 태스크 오류: {str(e)}", exc_info=True)

            # 태스크 생성 및 추적
            task = asyncio.create_task(wrapper())
            self._background_tasks.append(task)

            # 완료 시 목록에서 제거하는 콜백 추가
            task.add_done_callback(
                lambda t: self._background_tasks.remove(t) if t in self._background_tasks else None
            )

        def cleanup_background_tasks(self) -> None:
            """
            실행 중인 모든 백그라운드 태스크를 정리합니다.
            애플리케이션 종료 시 호출해야 합니다.
            """
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            self._background_tasks.clear()
            logger.debug("모든 백그라운드 태스크가 정리되었습니다.")

        def create_circuit_breaker(self, name: str, **kwargs) -> Any:
            """
            명명된 회로 차단기 인스턴스를 생성합니다.

            Args:
                name: 회로 차단기 이름
                **kwargs: 회로 차단기 구성 매개변수

            Returns:
                Any: 생성된 회로 차단기 인스턴스
            """
            try:
                # 회로 차단기 클래스 동적 임포트
                from src.services.llm.circuit_breaker import CircuitBreaker
                return CircuitBreaker(name=name, **kwargs)
            except ImportError:
                # 임시 조치: 기존 구현체 사용
                from src.services.llm_ollama_process import CircuitBreaker
                return CircuitBreaker(**kwargs)

        def create_stream_manager(self, request: ChatRequest, **kwargs) -> Any:
            """
            스트리밍 관리자 인스턴스를 생성합니다.

            Args:
                request: 채팅 요청 객체
                **kwargs: 추가 구성 매개변수

            Returns:
                Any: 생성된 스트리밍 관리자 인스턴스
            """
            try:
                # 스트리밍 관리자 클래스 동적 임포트
                from src.services.llm.streaming.stream_manager import StreamManager
                return StreamManager(request, **kwargs)
            except ImportError:
                # 임시 조치: 스트리밍 관리자 객체가 없으면 None 반환
                logger.warning("스트리밍 관리자 클래스를 임포트할 수 없습니다.")
                return None
