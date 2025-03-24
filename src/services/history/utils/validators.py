"""
검증 유틸리티 모듈

대화 히스토리 관련 데이터 검증을 위한 유틸리티 함수를 제공합니다.
"""

import logging
import re
from typing import List, Optional, Set

# 로거 설정
logger = logging.getLogger(__name__)


def extract_important_entities(text: str) -> List[str]:
    """
    텍스트에서 중요 엔티티(고유명사, 전문용어 등)를 추출합니다.
    간단한 구현으로, 대문자로 시작하는 단어와 숫자를 포함하는 단어를 추출합니다.

    Args:
        text (str): 분석할 텍스트

    Returns:
        List[str]: 추출된 중요 엔티티 목록
    """
    if not text:
        return []

    try:
        # 한글 고유명사 추출 패턴 (2글자 이상 연속된 한글)
        korean_entities = re.findall(r'[가-힣]{2,}', text)

        # 영문 대문자로 시작하는 단어 추출
        capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)

        # 숫자가 포함된 단어 추출
        numeric_entities = re.findall(r'\b\w*\d+\w*\b', text)

        # 결과 결합 및 중복 제거
        entities = list(set(korean_entities + capitalized_words + numeric_entities))

        # 불용어 제거 (필요시 확장)
        stopwords = ['그것', '이것', '저것', '그', '이', '저', '그런', '이런', '저런']
        entities = [e for e in entities if e not in stopwords and len(e) > 1]

        return entities
    except Exception as e:
        logger.error(f"엔티티 추출 중 오류: {str(e)}")
        return []


def validate_rewritten_question(
        original_question: str,
        rewritten_question: str,
        important_entities: Optional[List[str]] = None,
        session_id: str = "unknown"
) -> str:
    """
    재작성된 질문의 품질을 검증하고 적합하지 않을 경우 원본 질문을 반환합니다.

    Args:
        original_question (str): 원본 질문
        rewritten_question (str): 재작성된 질문
        important_entities (List[str], optional): 보존해야 할 중요 엔티티 목록
        session_id (str): 로깅용 세션 ID

    Returns:
        str: 검증을 통과한 재작성 질문 또는 원본 질문
    """
    if not original_question or not rewritten_question:
        return original_question

    try:
        # 중요 엔티티가 지정되지 않은 경우, 원본 질문에서 추출
        if important_entities is None:
            important_entities = extract_important_entities(original_question)

        # 1. 핵심 엔티티 보존 검증
        entities_preserved = all(entity in rewritten_question for entity in important_entities)

        # 2. 의미적 일관성 검증 (간소화된 버전)
        # 정교한 임베딩 비교는 추가 라이브러리가 필요하므로, 간단한 단어 중복 기반 유사도 사용
        original_words = set(original_question.lower().split())
        rewritten_words = set(rewritten_question.lower().split())
        word_overlap = len(original_words.intersection(rewritten_words)) / max(len(original_words), 1)

        # 3. 길이 및 복잡성 검증
        length_ratio = len(rewritten_question) / max(len(original_question), 1)
        length_appropriate = 0.7 <= length_ratio <= 2.5

        # 로깅
        logger.debug(
            f"[{session_id}] 질문 재작성 검증: 엔티티 보존={entities_preserved}, "
            f"단어 유사도={word_overlap:.2f}, 길이 비율={length_ratio:.2f}"
        )

        # 판단 로직
        if not entities_preserved:
            logger.warning(f"[{session_id}] 엔티티 보존 실패로 원본 질문 사용")
            return original_question

        if word_overlap < 0.4:  # 단어 기반 유사도 임계값
            logger.warning(f"[{session_id}] 의미 유사도 낮음으로 원본 질문 사용")
            return original_question

        if not length_appropriate:
            logger.warning(f"[{session_id}] 길이 비율 부적절로 원본 질문 사용")
            return original_question

        logger.debug(f"[{session_id}] 재작성된 질문 검증 통과")
        return rewritten_question

    except Exception as e:
        logger.error(f"[{session_id}] 질문 검증 중 오류: {str(e)}")
        return original_question


def is_valid_history_message(message: dict) -> bool:
    """
    히스토리 메시지의 유효성을 검사합니다.

    Args:
        message (dict): 검사할 메시지 딕셔너리

    Returns:
        bool: 메시지가 유효하면 True, 그렇지 않으면 False
    """
    try:
        # 필수 필드 확인
        if not message:
            return False

        if not isinstance(message, dict):
            return False

        if 'role' not in message or 'content' not in message:
            return False

        # 역할 값 검증
        valid_roles = {'HumanMessage', 'AIMessage', 'System', 'Human', 'AI', 'user', 'assistant', 'system'}
        if message['role'] not in valid_roles:
            return False

        # 내용 검증
        if not message['content'] or not isinstance(message['content'], str):
            return False

        return True

    except Exception as e:
        logger.error(f"메시지 검증 중 오류: {str(e)}")
        return False


def is_valid_session_id(session_id: str) -> bool:
    """
    세션 ID의 유효성을 검사합니다.

    Args:
        session_id (str): 검사할 세션 ID

    Returns:
        bool: 세션 ID가 유효하면 True, 그렇지 않으면 False
    """
    if not session_id:
        return False

    # 세션 ID는 알파벳, 숫자, 하이픈, 언더스코어만 포함
    pattern = re.compile(r'^[a-zA-Z0-9\-_]+$')
    return bool(pattern.match(session_id))
