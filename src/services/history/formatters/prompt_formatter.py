"""
프롬프트 포맷터 모듈

대화 히스토리 및 컨텍스트를 다양한 LLM 요구사항에 맞게 형식화하는 기능을 제공합니다.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from src.services.history.base import PromptFormatterBase

# 로거 설정
logger = logging.getLogger(__name__)


class StandardPromptFormatter(PromptFormatterBase):
    """
    표준 프롬프트 포맷터 클래스

    대화 이력 및 컨텍스트를 일반적인 LLM 모델 형식으로 변환합니다.
    """

    def format_history_for_prompt(self, session_history: ChatMessageHistory, max_turns: int = 5) -> str:
        """
        형식화된 대화 이력을 생성합니다.
        이전 대화의 맥락을 효과적으로 포착하기 위한 개선된 형식을 사용합니다.

        Args:
            session_history: 채팅 메시지 이력
            max_turns: 포함할 최대 대화 턴 수 (기본값: 5)

        Returns:
            형식화된 대화 이력 문자열
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

            # 대화 이력 프롬프트 헤더 추가
            formatted_history = ["# 이전 대화 내용"]

            # 대화 턴 구성
            turns = []
            current_turn = {"user": None, "assistant": None}

            for msg in messages:
                # 타입 검사 추가
                if hasattr(msg, '__class__') and hasattr(msg.__class__, '__name__'):
                    msg_type = msg.__class__.__name__
                else:
                    msg_type = str(type(msg))

                # HumanMessage 처리
                if isinstance(msg, HumanMessage) or "HumanMessage" in msg_type:
                    # 이전 턴이 있으면 저장
                    if current_turn["user"] is not None and current_turn["assistant"] is not None:
                        turns.append(current_turn)
                        current_turn = {"user": None, "assistant": None}

                    # 현재 사용자 메시지 저장
                    if hasattr(msg, 'content'):
                        current_turn["user"] = msg.content
                    else:
                        # content 속성이 없는 경우 문자열 변환 시도
                        current_turn["user"] = str(msg)

                # AIMessage 처리
                elif isinstance(msg, AIMessage) or "AIMessage" in msg_type:
                    if hasattr(msg, 'content'):
                        current_turn["assistant"] = msg.content
                    else:
                        # content 속성이 없는 경우 문자열 변환 시도
                        current_turn["assistant"] = str(msg)

            # 마지막 턴 저장
            if current_turn["user"] is not None:
                turns.append(current_turn)

            # 턴 수가 많으면 가장 최근 턴 유지
            if len(turns) > max_turns:
                turns = turns[-max_turns:]

            # 형식화된 대화 이력 생성
            for i, turn in enumerate(turns):
                formatted_history.append(f"\n## 대화 {i + 1}")

                if turn["user"]:
                    formatted_history.append(f"User: {turn['user']}")

                if turn["assistant"]:
                    formatted_history.append(f"Assistant: {turn['assistant']}")

            # 개선된 프롬프트 지시문 추가
            formatted_history.append("\n# 현재 질문에 답변할 때 위 대화 내용을 참고하세요.")

            return "\n".join(formatted_history)

        except Exception as e:
            # 예외 발생 시 로깅하고 빈 문자열 반환
            logger.error(f"대화 이력 형식화 중 오류 발생: {str(e)}")
            return ""

    def build_system_prompt(self, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        시스템 프롬프트 템플릿을 컨텍스트 변수로 채워 최종 프롬프트를 생성합니다.
        누락된 변수 처리 및 오류 복구 기능을 포함합니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 컨텍스트 변수

        Returns:
            str: 형식화된 시스템 프롬프트
        """
        try:
            # 템플릿에 재작성된 질문 주입을 위한 토큰 추가
            if "rewritten_question" in context and "{rewritten_question}" not in system_prompt_template:
                # 템플릿에 재작성된 질문 활용 지시문 추가
                insert_point = system_prompt_template.find("{input}")
                if insert_point > 0:
                    instruction = ("\n\n# 재작성된 질문\n"
                                   "다음은 대화 맥락을 고려하여 명확하게 재작성된 질문입니다. 응답 생성 시 참고하세요:\n"
                                   "{rewritten_question}\n\n# 원래 질문\n")
                    system_prompt_template = (system_prompt_template[:insert_point]
                                              + instruction + system_prompt_template[insert_point:])

            # 모든 필수 키가 있는지 확인
            required_keys = set()
            for match in re.finditer(r"{(\w+)}", system_prompt_template):
                required_keys.add(match.group(1))

            # 누락된 키가 있으면 빈 문자열로 대체
            for key in required_keys:
                if key not in context:
                    logger.warning(f"시스템 프롬프트 템플릿에 필요한 키가 누락됨: {key}, 빈 문자열로 대체합니다.")
                    context[key] = ""

            # 템플릿 형식화
            return system_prompt_template.format(**context)

        except KeyError as e:
            # 누락된 키 처리
            missing_key = str(e).strip("'")
            logger.warning(f"시스템 프롬프트 템플릿에 키가 누락됨: {missing_key}, 빈 문자열로 대체합니다.")
            context[missing_key] = ""
            return self.build_system_prompt(system_prompt_template, context)
        except Exception as e:
            # 기타 예외 처리
            logger.error(f"시스템 프롬프트 형식화 중 오류: {e}")
            # 기본 프롬프트로 폴백
            return (f"다음 컨텍스트를 기반으로 질문에 답하세요:\n\n컨텍스트: {context.get('context', '')}\n\n"
                    f"질문: {context.get('input', '질문 없음')}")

    def build_system_prompt_improved(self, system_prompt_template: str, context: Dict[str, Any]) -> str:
        """
        개선된 시스템 프롬프트 빌드 메소드.
        재작성된 질문을 포함하고 오류 처리를 강화했습니다.

        Args:
            system_prompt_template: 프롬프트 템플릿
            context: 템플릿에 적용할 변수들

        Returns:
            str: 형식화된 시스템 프롬프트
        """
        return self.build_system_prompt(system_prompt_template, context)
