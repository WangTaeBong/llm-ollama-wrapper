"""
메시지 데이터 모델 모듈

채팅 메시지 및 세션 관련 데이터 모델을 정의합니다.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.services.messaging.utils import generate_timestamp


@dataclass
class ChatMessage:
    """채팅 메시지 데이터 클래스"""
    role: str
    content: str
    timestamp: str = field(default_factory=generate_timestamp)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """메시지를 딕셔너리로 변환"""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            **({k: v for k, v in self.metadata.items() if v is not None} if self.metadata else {})
        }


@dataclass
class ChatSession:
    """채팅 세션 데이터 클래스"""
    chat_id: str
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(self, message: ChatMessage) -> None:
        """메시지 추가"""
        self.messages.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """세션을 딕셔너리로 변환"""
        return {
            "chat_id": self.chat_id,
            "messages": [msg.to_dict() for msg in self.messages]
        }
