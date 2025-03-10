import logging
import time
from collections import defaultdict, deque
from typing import Dict, Deque, Callable, Optional, Set

from fastapi import Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.common.config_loader import ConfigLoader

# Configure logger
logger = logging.getLogger(__name__)

# Load global settings
config_loader = ConfigLoader()
app_settings = config_loader.get_settings()


class RequestLimiterMiddleware(BaseHTTPMiddleware):
    """
    API request rate limiting middleware.

    Controls request frequency per IP to prevent DDoS attacks and limit API usage.
    This middleware tracks requests over a configurable time window and temporarily
    blocks IPs that exceed the defined rate limits.

    Features:
    - Configurable request rate limits and blocking duration
    - Whitelist support for both paths and IP addresses
    - Environment-aware behavior (can disable in development)
    - Proper handling of clients behind proxies
    """

    def __init__(
            self,
            app: ASGIApp,
            rate_limit: int = 100,
            time_window: int = 60,
            blocked_time: int = 300,
            whitelisted_paths: Optional[Set[str]] = None,
            whitelisted_ips: Optional[Set[str]] = None,
            environment: str = config_loader.get_environment().lower(),
    ):
        """
        Initialize the RequestLimiterMiddleware.

        Args:
            app (ASGIApp): The ASGI application
            rate_limit (int): Maximum number of requests allowed within the time window.
                              Defaults to 100 requests.
            time_window (int): Time period in seconds over which to apply rate limiting.
                               Defaults to 60 seconds (1 minute).
            blocked_time (int): Duration in seconds to block IPs that exceed the rate limit.
                                Defaults to 300 seconds (5 minutes).
            whitelisted_paths (Optional[Set[str]]): Set of URL paths exempt from rate limiting.
                                                    Defaults to common API docs and health check endpoints.
            whitelisted_ips (Optional[Set[str]]): Set of IP addresses exempt from rate limiting.
                                                  Defaults to localhost addresses.
            environment (str): Current execution environment (e.g., "development", "production").
                               Determines whether rate limiting is applied.
        """
        super().__init__(app)
        self.rate_limit = rate_limit
        self.time_window = time_window
        self.blocked_time = blocked_time
        self.whitelisted_paths = whitelisted_paths or {"/health", "/metrics", "/docs", "/openapi.json"}
        self.whitelisted_ips = whitelisted_ips or {"127.0.0.1", "::1"}
        self.environment = environment

        # Dictionary to store request history per IP
        # Using deque with a maximum length to automatically remove the oldest requests
        self.request_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=self.rate_limit * 2))

        # Dictionary to store blocked IPs and their release times
        self.blocked_ips: Dict[str, float] = {}

        # Log initialization parameters
        logger.info(
            f"RequestLimiterMiddleware initialized: "
            f"rate_limit={rate_limit}/min, "
            f"blocked_time={blocked_time}s, "
            f"env={environment}"
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Apply rate limiting before processing the request.

        This method is called for every request passing through the middleware.
        It checks if the request should be rate-limited based on client IP,
        path, and previous request history.

        Args:
            request (Request): The incoming HTTP request
            call_next (Callable): Function to call the next middleware in the chain

        Returns:
            Response: HTTP response, either from the application or a rate limit error
        """
        # Bypass rate limiting in development environment (optional behavior)
        if self.environment == "development":
            return await call_next(request)

        # Bypass rate limiting for whitelisted paths
        if any(request.url.path.startswith(path) for path in self.whitelisted_paths):
            return await call_next(request)

        # Get client IP address
        client_ip = self._get_client_ip(request)

        # Bypass rate limiting for whitelisted IPs
        if client_ip in self.whitelisted_ips:
            return await call_next(request)

        # Get current timestamp
        current_time = time.time()

        # Clean up expired IP blocks
        self._clean_expired_blocks(current_time)

        # Check if IP is currently blocked
        if client_ip in self.blocked_ips:
            block_release_time = self.blocked_ips[client_ip]
            remaining_time = int(block_release_time - current_time)

            # Log warning about blocked IP attempting a request
            logger.warning(f"Request attempt from blocked IP: {client_ip}, remaining block time: {remaining_time}s")

            # Return 429 Too Many Requests response
            return self._create_rate_limit_response(remaining_time)

        # Check rate limit for this IP
        if not self._check_rate_limit(client_ip, current_time):
            # Block IP and calculate release time
            self.blocked_ips[client_ip] = current_time + self.blocked_time

            # Log warning about new IP block
            logger.warning(f"IP blocked due to rate limit exceeded: {client_ip}, block duration: {self.blocked_time}s")

            # Return 429 Too Many Requests response
            return self._create_rate_limit_response(self.blocked_time)

        # Process the request normally if rate limit is not exceeded
        return await call_next(request)

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """
        Extract the client's IP address from the request.

        Handles requests behind proxy servers by checking X-Forwarded-For header.
        Falls back to direct client IP if the header is not present.

        Args:
            request (Request): HTTP request object

        Returns:
            str: Client IP address
        """
        # Check X-Forwarded-For header (when behind proxy servers)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Use the first IP in the chain as the client IP
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            # Use directly connected client IP
            client_ip = request.client.host if request.client else "unknown"

        return client_ip

    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """
        Check if the client IP's request rate is within the allowed limit.

        Records the current request timestamp and counts recent requests
        within the configured time window.

        Args:
            client_ip (str): Client IP address
            current_time (float): Current timestamp

        Returns:
            bool: True if request is allowed, False if rate limit is exceeded
        """
        # Add current request to the history
        self.request_history[client_ip].append(current_time)

        # Calculate the start of the time window
        window_start = current_time - self.time_window

        # Count requests within the time window
        recent_requests = sum(1 for timestamp in self.request_history[client_ip] if timestamp > window_start)

        # Log debug information (only for IPs with high request volume)
        if recent_requests > self.rate_limit * 0.7:
            logger.debug(f"Recent requests from IP {client_ip}: {recent_requests}/{self.rate_limit}")

        # Check if request count exceeds the limit
        return recent_requests <= self.rate_limit

    def _clean_expired_blocks(self, current_time: float) -> None:
        """
        Remove expired IP blocks from the blocked list.

        This method is called before processing each request to ensure
        that IPs are unblocked when their penalty time expires.

        Args:
            current_time (float): Current timestamp
        """
        # Identify IPs with expired blocks
        to_remove = [ip for ip, release_time in self.blocked_ips.items() if release_time <= current_time]

        # Remove expired blocks
        for ip in to_remove:
            del self.blocked_ips[ip]
            logger.info(f"IP block released: {ip}")

    def _create_rate_limit_response(self, retry_after: int) -> Response:
        """
        Create a rate limit exceeded response.

        Generates a standard 429 Too Many Requests response with appropriate
        headers and a JSON body explaining the situation.

        Args:
            retry_after (int): Recommended retry time in seconds

        Returns:
            Response: 429 status code response with explanation
        """
        from fastapi.responses import JSONResponse

        # Create 429 Too Many Requests response
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "status": "error",
                "detail": "Too many requests. Please try again later.",
                "retry_after": retry_after
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(self.rate_limit),
                "X-RateLimit-Reset": str(int(time.time()) + retry_after)
            }
        )
