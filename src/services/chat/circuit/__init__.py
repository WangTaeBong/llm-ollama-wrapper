"""
회로 차단기 관련 패키지

외부 서비스 호출의 안정성을 보장하는 회로 차단기 패턴 구현을 제공합니다.
"""

from src.chat.circuit.circuit_breaker import CircuitBreaker

__all__ = ['CircuitBreaker']