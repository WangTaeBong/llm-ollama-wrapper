"""
Circuit breaker pattern implementation for external service protection.

This module provides a circuit breaker implementation that helps protect
the system from cascading failures when external services experience issues.
"""

import logging
import time
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external service calls.

    Protects system stability by opening the circuit after consecutive failures,
    preventing additional requests, and automatically attempting recovery.

    The circuit has three states:
    - CLOSED: Normal operation, requests are passed through
    - OPEN: Circuit is open, all requests are rejected
    - HALF-OPEN: Test state, limited requests are allowed to check if service is recovered
    """

    def __init__(self, failure_threshold=3, recovery_timeout=60, reset_timeout=300):
        """
        Initialize the circuit breaker.

        Args:
            failure_threshold (int): Number of consecutive failures before opening the circuit.
            recovery_timeout (int): Seconds to wait before attempting recovery.
            reset_timeout (int): Seconds to wait before fully resetting the circuit.
        """
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.reset_timeout = reset_timeout
        self.open_time = None
        self.state = "CLOSED"
        self.half_open_calls = 0
        self.lock = Lock()

    def is_open(self):
        """
        Check if the circuit is open and handle auto-recovery.

        Returns:
            bool: True if requests should be blocked, False if allowed.
        """
        with self.lock:
            if self.state == "CLOSED":
                return False

            if self.state == "OPEN":
                # Check if recovery timeout has elapsed
                if time.time() - self.open_time > self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to half-open state")
                    self.state = "HALF-OPEN"
                    self.half_open_calls = 0
                    return False
                return True

            # In HALF-OPEN state, allow limited calls
            if self.half_open_calls < 1:
                self.half_open_calls += 1
                return False
            return True

    def record_success(self):
        """
        Record a successful call to the service.

        In HALF-OPEN state, this will close the circuit.
        In CLOSED state, this resets the failure counter.
        """
        with self.lock:
            if self.state == "HALF-OPEN":
                logger.info("Circuit breaker closed - service has recovered")
                self.state = "CLOSED"
                self.failure_count = 0
                self.open_time = None
                self.half_open_calls = 0
            elif self.state == "CLOSED":
                self.failure_count = 0

    def record_failure(self):
        """
        Record a failed call to the service.

        In HALF-OPEN state, this will re-open the circuit.
        In CLOSED state, this increases the failure counter and may open the circuit.
        """
        with self.lock:
            if self.state == "HALF-OPEN":
                logger.warning("Circuit breaker keeping open state - service still failing")
                self.state = "OPEN"
                self.open_time = time.time()
            elif self.state == "CLOSED":
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    logger.warning("Circuit breaker opening - failure threshold reached")
                    self.state = "OPEN"
                    self.open_time = time.time()
