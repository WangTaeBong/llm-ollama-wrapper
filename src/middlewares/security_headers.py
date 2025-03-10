from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds security-related HTTP headers to responses.

    Helps prevent various attacks such as XSS (Cross-Site Scripting),
    clickjacking, MIME sniffing, and other common web vulnerabilities
    by setting appropriate security headers on all responses.

    This middleware implements best practices for web security headers
    as recommended by OWASP and other security standards bodies.
    """

    def __init__(
            self,
            app: ASGIApp,
            content_security_policy: Optional[str] = None,
            include_development_headers: bool = False,
    ):
        """
        Initialize the SecurityHeadersMiddleware.

        Args:
            app (ASGIApp): The ASGI application
            content_security_policy (Optional[str]): Custom Content-Security-Policy header value.
                If not provided, a restrictive default policy will be used.
            include_development_headers (bool): Whether to include development-friendly headers
                such as permissive CORS settings. Should be False in production.
        """
        super().__init__(app)
        self.content_security_policy = content_security_policy
        self.include_development_headers = include_development_headers

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request and add security headers to the response.

        This method intercepts each response after it has been processed by the
        application and adds various security headers before returning it to the client.

        Args:
            request (Request): The incoming HTTP request
            call_next (Callable): Function to call the next middleware in the chain

        Returns:
            Response: HTTP response with added security headers
        """
        # Process the request through the application chain
        response = await call_next(request)

        # Add basic security headers to all responses
        response.headers["X-Content-Type-Options"] = "nosniff"  # Prevents MIME type sniffing
        response.headers["X-Frame-Options"] = "DENY"  # Prevents clickjacking via iframes
        response.headers["X-XSS-Protection"] = "1; mode=block"  # Enables browser XSS filtering
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"  # Controls referrer information

        # Add Strict Transport Security header (HTTPS enforcement) - only for HTTPS requests
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        # Set Content-Security-Policy header
        if self.content_security_policy:
            # Use custom policy if provided
            response.headers["Content-Security-Policy"] = self.content_security_policy
        else:
            # Apply default restrictive CSP policy
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "  # Default fallback for all resource types
                "img-src 'self' data:; "  # Allow images from same origin and data URIs
                "style-src 'self' 'unsafe-inline'; "  # Allow styles from same origin and inline
                "script-src 'self' 'unsafe-inline'; "  # Allow scripts from same origin and inline
                "font-src 'self'; "  # Allow fonts from same origin
                "connect-src 'self'; "  # Allow connections to same origin
                "object-src 'none'; "  # Block <object>, <embed>, and <applet> elements
                "base-uri 'self';"  # Restrict <base> URIs to same origin
            )

        # Skip certain headers for preflight (OPTIONS) requests
        if request.method != "OPTIONS":
            # Add cache control headers for API and admin routes
            # This prevents browsers from caching sensitive data
            if request.url.path.startswith(("/api/", "/admin/")):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"

            # Add development-friendly headers (such as permissive CORS) if enabled
            # WARNING: These headers should never be used in production
            if self.include_development_headers:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"

        # Add Permissions-Policy header (successor to Feature-Policy)
        # Restricts access to browser features that might pose privacy or security risks
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "  # Disable access to user location
            "microphone=(), "  # Disable access to microphone
            "camera=(), "  # Disable access to camera
            "payment=(), "  # Disable access to payment APIs
            "usb=(), "  # Disable access to USB devices
            "accelerometer=(), "  # Disable access to motion sensors
            "gyroscope=(), "  # Disable access to orientation sensors
            "magnetometer=()"  # Disable access to magnetic field sensors
        )

        return response
