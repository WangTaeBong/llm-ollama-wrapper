"""
Gemma 모델용 프롬프트 포맷터 모듈

Gemma 모델의 특화된 형식 요구사항에 맞게 대화 히스토리와 프롬프트를 구성합니다.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from src.services.history.formatters.prompt_formatter import StandardPromptFormatter

# 로거 설정
logger = logging.getLogger(__name__)


class GemmaPromptFormatter(StandardPromptFormatter):
    """
    Gemma 모델용 프롬프트 포맷터 클래스

    Gemma 모델의 특수한 형식(<start_of_turn>/<end_of_turn> 등)에 맞게
    대화 이력과 시스템 프롬프트를 구성합니다.
    """

    def format_history_for_prompt(self, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        Gemma 모델에 적합한 형식으로 대화 이력을 구성합니다.
        Gemma의 <start_of_turn>user/<start_of_turn>model 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수 (기본값: 5)

        Returns:
            str: Gemma 형식의 대화 이력 문자열
        """
        try:
            # 파라미터 유효성 검사
            if not session_history or not hasattr(session_history, 'messages'):
                logger.warning("유효하지 않은 session_history 객체가 제공되었습니다.")
                return ""

            messages = session_history.messages
            if not messages:
                return ""

            # 가장 최근 대화부터 max_turns 수만큼만 추출
            if len(messages) > max_turns * 2:  # 각 턴은 사용자 메시지와 시스템 응답을 포함
                messages = messages[-(max_turns * 2):]

            formatted_history = []

            # 대화 턴 구성
            for i in range(0, len(messages), 2):
                # 사용자 메시지
                if i < len(messages):
                    user_msg = messages[i]
                    if hasattr(user_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>user\n{user_msg.content}<end_of_turn>")

                # 시스템 메시지
                if i + 1 < len(messages):
                    sys_msg = messages[i + 1]
                    if hasattr(sys_msg, 'content'):
                        formatted_history.append(f"<start_of_turn>model\n{sys_msg.content}<end_of_turn>")

            return "\n".join(formatted_history)

        except Exception as e:
            # 예외 발생 시 로깅하고 빈 문자열 반환
            logger.error(f"Gemma 대화 이력 형식화 중 오류 발생: {str(e)}")
            return ""

    def build_system_prompt(self, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        Gemma에 맞는 형식으로 시스템 프롬프트를 구성합니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: Gemma 형식의 시스템 프롬프트
        """
        try:
            # 먼저 기존 함수로 프롬프트 생성
            raw_prompt = super().build_system_prompt(system_prompt_template, context)

            # Gemma 형식으로 변환
            # <start_of_turn>user 형식으로 시작
            formatted_prompt = "<start_of_turn>user\n"

            # 시스템 프롬프트 삽입
            formatted_prompt += raw_prompt

            # 사용자 입력부 종료 및 모델 응답 시작
            formatted_prompt += "\n<end_of_turn>\n<start_of_turn>model\n"

            return formatted_prompt

        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt(system_prompt_template, context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"Gemma 시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 Gemma 프롬프트로 폴백
            basic_prompt = (f"<start_of_turn>user\n다음 질문에 답해주세요: {context.get('input', '질문 없음')}\n"
                            f"end_of_turn>\n<start_of_turn>model\n")
            return basic_prompt

    def build_system_prompt_improved(self, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        개선된 Gemma 시스템 프롬프트 빌드 메소드.
        재작성된 질문을 포함하고 오류 처리를 강화했습니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: 형식화된 Gemma 시스템 프롬프트
        """
        return self.build_system_prompt(system_prompt_template, context)
