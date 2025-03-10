import logging
from typing import Tuple, Union

from fastapi import APIRouter, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.common.config_loader import ConfigLoader
from src.utils.redis_manager import RedisManager

# Load application settings
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Configure logging
logger = logging.getLogger(__name__)

# Configure template directory
templates = Jinja2Templates(directory="templates")

# Define health check router
health_redis_router = APIRouter()


@health_redis_router.get(
    "/health_redis",
    include_in_schema=False,
    response_class=HTMLResponse,
    responses={
        200: {"description": "Healthy Redis connections"},
        503: {"description": "Unhealthy Redis connections"},
        500: {"description": "Internal server error"}
    }
)
async def health_redis(request: Request, format_type: str = "html"):
    """
    Check Redis connection status and display the results.

    This endpoint validates both synchronous and asynchronous Redis connections
    and returns the status in either HTML or JSON format. It's designed for
    monitoring systems, health dashboards, and DevOps integration.

    Args:
        request (Request): The incoming HTTP request
        format_type (str, optional): Response format ('html' or 'json'). Defaults to 'html'

    Returns:
        Union[HTMLResponse, JSONResponse]: Redis status results in the requested format

    Response Codes:
        200: All Redis connections are healthy
        503: One or more Redis connections are unhealthy
        500: Error rendering the response
    """
    # Initialize status code (default to healthy)
    status_code = status.HTTP_200_OK

    # Check Redis connection status
    sync_status, sync_message = await _check_sync_redis()
    async_status, async_message = await _check_async_redis()

    # Determine overall status (unhealthy if either connection fails)
    if not (sync_status and async_status):
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    # Prepare result data structure
    health_data = {
        "redis": {
            "status": "healthy" if (sync_status and async_status) else "unhealthy",
            "sync": {
                "status": "healthy" if sync_status else "unhealthy",
                "message": sync_message
            },
            "async": {
                "status": "healthy" if async_status else "unhealthy",
                "message": async_message
            }
        }
    }

    # Return JSON response if requested
    if format_type.lower() == "json":
        return JSONResponse(content=health_data, status_code=status_code)

    # Prepare and return HTML response
    try:
        content = f"""
            <h2>Redis Health Report</h2>
            <p><strong>Overall Status:</strong> {'Healthy' if (sync_status and async_status) else 'Unhealthy'}</p>
            <p><strong>Async Redis:</strong> {async_message}</p>
            <p><strong>Sync Redis:</strong> {sync_message}</p>
        """

        return templates.TemplateResponse(
            "base.html",
            {
                "request": request,
                "title": "Redis Health Check",
                "content": content
            },
            status_code=status_code
        )
    except Exception as e:
        logger.error(f"Failed to render health check UI: {e}")
        return HTMLResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content="<h2>Error</h2><p>Failed to load health check UI.</p>"
        )


async def _check_async_redis() -> Tuple[bool, str]:
    """
    Check the asynchronous Redis connection status.

    Attempts to ping the Redis server using the asynchronous client
    to verify that the connection is operational.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: Connection status (True for healthy, False for unhealthy)
            - str: Status message with details about the connection state
    """
    try:
        # Use the health check method from RedisManager
        if await RedisManager.async_health_check():
            return True, "Connected (PONG)"
        return False, "Failed to connect"
    except Exception as e:
        logger.error(f"Async Redis health check error: {e}")
        return False, f"Error: {str(e)}"


async def _check_sync_redis() -> Tuple[bool, str]:
    """
    Check the synchronous Redis connection status.

    Attempts to ping the Redis server using the synchronous client
    to verify that the connection is operational.

    Returns:
        Tuple[bool, str]: A tuple containing:
            - bool: Connection status (True for healthy, False for unhealthy)
            - str: Status message with details about the connection state
    """
    try:
        # Use the health check method from RedisManager
        if RedisManager.health_check():
            return True, "Connected (PONG)"
        return False, "Failed to connect"
    except Exception as e:
        logger.error(f"Sync Redis health check error: {e}")
        return False, f"Error: {str(e)}"
