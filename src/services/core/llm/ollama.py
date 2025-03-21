"""
Ollama LLM 서비스 구현 모듈
========================

이 모듈은 Ollama API를 사용하여 LLM 서비스를 구현합니다.
LLMServiceBase 인터페이스를 구현하여 Ollama 백엔드와의 통신을 담당합니다.

기능:
- Ollama API 연결 및 초기화
- 동기 및 비동기 추론 처리
- 프롬프트 생성 및 응답 처리
- 메트릭 수집 및 모니터링
"""

import asyncio
import inspect
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, AsyncGenerator, Union

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama.llms import OllamaLLM

from src.services.core.base import LLMServiceBase
from src.services.core.factory import LLMServiceFactory

# 로거 설정
logger = logging.getLogger(__name__)


class OllamaLLMService(LLMServiceBase):
    """
    Ollama 백엔드를 사용하는 LLM 서비스 구현

    Ollama API를 통해 LLM 모델과 통신하고, 응답을 생성합니다.
    LLMServiceBase를 상속받아 모든 필수 메서드를 구현합니다.
    """

    def __init__(self, settings):
        """
        Ollama LLM 서비스 초기화

        Args:
            settings: 설정 객체
        """
        super().__init__(settings)
        self.ollama_settings = getattr(settings, 'ollama', None)
        self.llm_settings = getattr(settings, 'llm', None)

        # 초기화 상태 플래그
        self.is_initialized = False

        # 체인 레지스트리 (캐싱용)
        self._chain_registry = {}

        # 세마포어 (동시 요청 제한)
        max_concurrent = getattr(settings.cache, 'max_concurrent_tasks', 5)
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def initialize(self) -> bool:
        """
        Ollama 모델 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        if self.is_initialized:
            return True

        try:
            if not self.ollama_settings:
                logger.error("Ollama 설정이 없습니다.")
                return False

            # access_type 확인 (URL 또는 로컬)
            access_type = getattr(self.ollama_settings, 'access_type', '').lower()

            if access_type == 'url':
                # URL 기반 접근
                base_url = getattr(self.ollama_settings, 'ollama_url', None)
                if not base_url:
                    logger.error("Ollama URL 설정이 없습니다.")
                    return False

                self.model = OllamaLLM(
                    base_url=base_url,
                    model=self.ollama_settings.model_name,
                    mirostat=getattr(self.ollama_settings, 'mirostat', 0),
                    temperature=getattr(self.ollama_settings, 'temperature', 0.7),
                )

            elif access_type == 'local':
                # 로컬 접근
                self.model = OllamaLLM(
                    model=self.ollama_settings.model_name,
                    mirostat=getattr(self.ollama_settings, 'mirostat', 0),
                    temperature=getattr(self.ollama_settings, 'temperature', 0.7),
                )

            else:
                logger.error(f"지원되지 않는 Ollama 접근 방식: {access_type}")
                return False

            logger.info(f"Ollama 모델 '{self.ollama_settings.model_name}' 초기화 완료 (접근 방식: {access_type})")
            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Ollama 모델 초기화 실패: {str(e)}")
            return False

    def build_system_prompt(self,
                            template: Union[str, PromptTemplate],
                            context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트 생성

        Args:
            template: 프롬프트 템플릿 (문자열 또는 PromptTemplate)
            context: 프롬프트 컨텍스트

        Returns:
            str: 완성된 시스템 프롬프트
        """
        try:
            # 템플릿이 PromptTemplate 인스턴스인 경우
            if isinstance(template, PromptTemplate):
                return template.format(**context)

            # 문자열 템플릿인 경우
            return template.format(**context)

        except KeyError as e:
            # 누락된 키가 있는 경우 빈 문자열로 대체
            missing_key = str(e).strip("'")
            logger.warning(f"프롬프트 템플릿에 누락된 키: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt(template, context)

        except Exception as e:
            logger.error(f"시스템 프롬프트 생성 중 오류: {str(e)}")
            # 기본 프롬프트로 대체
            return f"다음 질문에 답변하세요: {context.get('input', '질문 없음')}"

    def _get_or_create_chain(self, prompt_template, session_id=None):
        """
        캐시된 체인을 가져오거나 새로 생성

        Args:
            prompt_template: 프롬프트 템플릿
            session_id: 세션 ID (캐시 키로 사용)

        Returns:
            chain: 생성된 체인
        """
        # 캐시 키 생성
        cache_key = f"{session_id}_{hash(prompt_template)}" if session_id else hash(prompt_template)

        # 캐시에서 체인 조회
        if cache_key in self._chain_registry:
            return self._chain_registry[cache_key]

        # 체인 생성
        chat_prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = create_stuff_documents_chain(self.model, chat_prompt)

        # 캐시에 저장
        self._chain_registry[cache_key] = chain
        return chain

    async def _limited_chain_invoke(self, chain, context, timeout=60):
        """
        세마포어를 통한 제한된 체인 호출

        Args:
            chain: 호출할 체인
            context: 체인 컨텍스트
            timeout: 타임아웃 (초)

        Returns:
            Any: 체인 응답

        Raises:
            asyncio.TimeoutError: 타임아웃 발생 시
            Exception: 기타 오류 발생 시
        """
        start_time = time.time()

        async with self._semaphore:
            try:
                if inspect.iscoroutinefunction(chain.invoke):
                    # 비동기 호출
                    return await asyncio.wait_for(chain.invoke(context), timeout=timeout)
                else:
                    # 동기 호출을 비동기 태스크로 래핑
                    task = asyncio.create_task(chain.invoke(context))
                    return await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                logger.error(f"체인 호출 타임아웃: {elapsed:.2f}초 (제한: {timeout}초)")
                raise
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"체인 호출 실패: {elapsed:.2f}초 후 {type(e).__name__}: {str(e)}")
                raise

    async def ask(self,
                  query: str,
                  documents: List[Document],
                  language: str,
                  context: Optional[Dict[str, Any]] = None) -> str:
        """
        Ollama LLM에 질의하고 응답 반환

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            str: LLM 응답 텍스트
        """
        if not self.is_initialized:
            logger.error("Ollama 모델이 초기화되지 않았습니다.")
            return "죄송합니다. LLM 모델이 초기화되지 않았습니다. 잠시 후 다시 시도해주세요."

        start_time = time.time()
        session_id = context.get("session_id") if context else None

        try:
            # 컨텍스트 준비
            prompt_context = {
                "input": query,
                "context": documents,
                "language": language,
                "today": self._get_today(),
            }

            # 추가 컨텍스트 병합
            if context:
                prompt_context.update(context)

            # 이미지 처리
            if "image" in prompt_context and prompt_context["image"]:
                prompt_context["image_description"] = self._format_image_data(prompt_context["image"])

            # VOC 관련 컨텍스트 추가
            rag_sys_info = prompt_context.get("rag_sys_info")
            if rag_sys_info == "komico_voc":
                voc_settings = getattr(self.settings, 'voc', None)
                if voc_settings:
                    prompt_context.update({
                        "gw_doc_id_prefix_url": getattr(voc_settings, 'gw_doc_id_prefix_url', ''),
                        "check_gw_word_link": getattr(voc_settings, 'check_gw_word_link', ''),
                        "check_gw_word": getattr(voc_settings, 'check_gw_word', ''),
                        "check_block_line": getattr(voc_settings, 'check_block_line', ''),
                    })

            # 프롬프트 템플릿 가져오기
            from src.services.response_generator import ResponseGenerator
            from src.common.query_check_dict import QueryCheckDict

            query_check_dict = QueryCheckDict(self.settings.prompt.llm_prompt_path)
            response_generator = ResponseGenerator(self.settings, query_check_dict)
            prompt_template = response_generator.get_rag_qa_prompt(rag_sys_info or "")

            # 체인 가져오기
            chain = self._get_or_create_chain(prompt_template, session_id)

            # 체인 호출
            timeout = getattr(self.llm_settings, 'timeout', 60)
            result = await self._limited_chain_invoke(chain, prompt_context, timeout=timeout)

            # 메트릭 업데이트
            elapsed = time.time() - start_time
            self.metrics["request_count"] += 1
            self.metrics["total_time"] += elapsed

            logger.info(f"[{session_id or ''}] Ollama 쿼리 완료: {elapsed:.4f}초 소요")

            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id or ''}] Ollama 쿼리 타임아웃: {elapsed:.4f}초 후")
            self.metrics["error_count"] += 1
            return "죄송합니다. 응답 생성 시간이 초과되었습니다. 잠시 후 다시 시도해주세요."

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"[{session_id or ''}] Ollama 쿼리 실패: {elapsed:.4f}초 후: {type(e).__name__}: {str(e)}")
            self.metrics["error_count"] += 1
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def stream_response(self,
                              query: str,
                              documents: List[Document],
                              language: str,
                              context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        스트리밍 모드로 Ollama LLM에 질의하고 응답 생성

        Args:
            query: 사용자 질의
            documents: 검색된 문서 리스트
            language: 응답 언어
            context: 추가 컨텍스트 정보

        Returns:
            AsyncGenerator: 응답 청크를 생성하는 비동기 제너레이터
        """
        session_id = context.get("session_id") if context else None
        logger.debug(f"[{session_id or ''}] Ollama는 현재 스트리밍을 지원하지 않습니다. 대체 응답을 생성합니다.")

        # Ollama는 현재 스트리밍을 직접 지원하지 않으므로 일반 응답을 스트리밍 형식으로 변환
        try:
            # 일반 응답 생성
            full_response = await self.ask(query, documents, language, context)

            # 청크 크기 설정 (문자 단위)
            chunk_size = 20

            # 첫 번째 청크 반환
            first_chunk = full_response[:chunk_size] if len(full_response) > chunk_size else full_response
            yield {
                "text": first_chunk,
                "finished": False
            }

            # 남은 텍스트가 있으면 청크로 분할하여 반환
            remaining_text = full_response[chunk_size:]
            if remaining_text:
                for i in range(0, len(remaining_text), chunk_size):
                    # 응답 청크 생성
                    chunk = remaining_text[i:i + chunk_size]
                    # 마지막 청크인지 확인
                    is_last = (i + chunk_size >= len(remaining_text))

                    # 짧은 지연 추가 (실제 스트리밍 시뮬레이션)
                    await asyncio.sleep(0.05)

                    yield {
                        "text": chunk,
                        "finished": is_last
                    }
            else:
                # 첫 번째 청크가 전체 응답인 경우
                yield {
                    "text": "",
                    "finished": True
                }

            # 전체 응답 반환
            yield {
                "complete_response": full_response
            }

        except Exception as e:
            logger.error(f"[{session_id or ''}] 스트리밍 응답 생성 중 오류: {str(e)}")
            yield {
                "error": True,
                "text": f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                "finished": True
            }

    @staticmethod
    def _format_image_data(image_data: Dict[str, str]) -> str:
        """
        이미지 데이터를 프롬프트에 추가하기 위한 형식으로 변환

        Args:
            image_data: 이미지 데이터 (base64, URL 등)

        Returns:
            str: 포맷된 이미지 정보
        """
        if 'base64' in image_data:
            return "[이미지 데이터가 base64 형식으로 전달되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"
        elif 'url' in image_data:
            return f"[이미지 URL: {image_data.get('url')}]"
        elif 'description' in image_data:
            return f"[이미지 설명: {image_data.get('description')}]"
        else:
            return "[이미지 데이터가 제공되었습니다. 이미지를 분석하여 관련 정보를 제공해주세요.]"

    @staticmethod
    def _get_today() -> str:
        """
        현재 날짜와 요일을 한국어 형식으로 반환

        Returns:
            str: 형식화된 날짜 문자열
        """
        today = datetime.now()
        weekday_names = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        weekday = weekday_names[today.weekday()]

        return f"{today.strftime('%Y년 %m월 %d일')} {weekday} {today.strftime('%H시 %M분')}입니다."


# 팩토리에 서비스 등록
LLMServiceFactory.register_service("ollama", OllamaLLMService)
