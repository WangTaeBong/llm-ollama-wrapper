import logging
import time  # Used to compare the login time with the current time

from fastapi import Request, status, HTTPException
from itsdangerous import URLSafeSerializer, BadSignature

from src.common.config_loader import ConfigLoader

# Logger configuration
logger = logging.getLogger(__name__)

# =============================================================================
# Load configuration and initialize the serializer for secure session management
# =============================================================================

# Load settings using a configuration loader (assumes secure storage of settings)
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Retrieve the secret key from settings for secure session management
SESSION_SECRET = settings.security.session_secret
# Initialize the serializer with the secret key to sign and verify session data
serializer = URLSafeSerializer(SESSION_SECRET)

# Define the maximum allowed session duration in seconds (here, 30 minutes)
MAX_SESSION_DURATION = 30 * 60  # 30 minutes


def is_authenticated(request: Request) -> bool:
    """
    Check if the user is authenticated and if the session is still valid based on the session cookie.

    This function performs the following steps:
      1. Retrieves the session cookie from the incoming request.
      2. Deserializes the cookie to obtain the session data.
      3. Verifies that the 'authenticated' flag in the session data is True.
      4. Checks that a 'login_time' is present in the session data.
      5. Validates that the session has not expired by comparing the current time with the login time.

    Args:
        request (Request): The HTTP request object that contains the cookies.

    Returns:
        bool: True if the session exists, the user is authenticated, and the session is within
              the allowed duration; False otherwise.
    """
    # Retrieve the session cookie from the request; return False if it doesn't exist.
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        return False

    try:
        # Deserialize the session cookie to extract session data.
        session_data = serializer.loads(session_cookie)

        # Check if the session data indicates that the user is authenticated.
        if not session_data.get("authenticated", False):
            return False

        # Retrieve the login time from the session data.
        login_time = session_data.get("login_time")
        if not login_time:
            # Log a warning if the login_time is missing.
            logger.warning("Session authentication failed: 'login_time' is missing in session data.")
            return False

        # Calculate the elapsed time since the login.
        if time.time() - login_time > MAX_SESSION_DURATION:
            # Log a warning if the session has expired.
            logger.warning("Session has expired based on the maximum allowed duration.")
            return False

        # If all checks pass, the session is valid.
        return True

    except BadSignature:
        # Log a warning if the session cookie has an invalid signature (potential tampering).
        logger.warning("Session authentication failed: Invalid signature detected.")
        return False
    except Exception as e:
        # Log any other unexpected errors during the authentication check.
        logger.error(f"Unexpected error during session authentication: {e}")
        return False


async def verify_authentication(request: Request):
    """
    Middleware to verify user authentication before granting access to protected resources.

    This asynchronous function is intended to be used as a dependency in FastAPI routes.
    It calls 'is_authenticated' to determine whether the user has a valid session.
    If the user is not authenticated or the session has expired, it raises an HTTPException
    with a 303 status code and a 'Location' header pointing to the login page. A global
    exception handlers should catch this exception and perform the appropriate redirection.

    Args:
        request (Request): The incoming FastAPI request.

    Raises:
        HTTPException: If the user is not authenticated, with a 303 (See Other) status code and
                       a 'Location' header indicating the URL of the login page.
    """
    if not is_authenticated(request):
        # Log the unauthorized access attempt.
        logger.warning("Unauthorized access attempt detected: redirecting to the login page.")
        # Raise an HTTPException to signal redirection to the login page.
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER,
            detail="Redirecting to login",
            headers={"Location": "/login"}
        )
