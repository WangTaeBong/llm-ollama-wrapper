import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from src.common.config_loader import ConfigLoader

# Load application settings
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Configure logging
logger = logging.getLogger(__name__)

# Templates directory setup
templates = Jinja2Templates(directory="templates")

# Define health check router
health_router = APIRouter()


@health_router.get(
    "/health",
    response_class=JSONResponse,
    responses={
        200: {"description": "Application is healthy"},
        500: {"description": "Application health check failed"}
    }
)
async def health_check():
    """
    Perform a basic health check to ensure the FastAPI application is running properly.

    This endpoint provides a simple way to verify that the application is responsive
    and can handle requests. It's suitable for integration with load balancers,
    container orchestration systems, and monitoring tools that require a lightweight
    health probe.

    Returns:
        JSONResponse: JSON object containing the status and message
            - status: "ok" or "error" indicating the health state
            - message: A human-readable description of the health status

    Response Codes:
        200: The application is healthy and functioning normally
        500: The application encountered an error during the health check
    """
    logger.info("Health check endpoint accessed.")
    try:
        return JSONResponse(
            status_code=200,
            content={"status": "ok", "message": "MAI-Chat LLM Wrapper is running smoothly."}
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Health check failed due to an internal error."}
        )


@health_router.get(
    "/health_check_ui",
    include_in_schema=False,
    response_class=HTMLResponse,
    responses={
        200: {"description": "HTML health status page"},
        500: {"description": "Failed to render health status page"}
    }
)
async def health_check_ui(request: Request):
    """
    Provide a user-friendly HTML interface for visualizing system health status.

    This endpoint renders a human-readable HTML page showing the current health
    status of the application. It's designed for direct access by developers,
    operations staff, or system administrators who need to quickly check system
    health through a browser.

    Unlike the JSON health check endpoint, this route returns a formatted HTML
    page that includes visual indicators and descriptive text about the system's
    current operational status.

    Args:
        request (Request): The incoming HTTP request, required for template rendering

    Returns:
        HTMLResponse: Rendered HTML page displaying system health information

    Response Codes:
        200: Successfully rendered the health status page
        500: Failed to render the health status page due to an error
    """
    # Determine the correct protocol based on the configuration
    protocol = "https" if settings.ssl.use_https else "http"

    content = f"""
        <h2>System Health: OK</h2>
        <p>The MAI-Chat LLM Wrapper is running smoothly on {protocol}.</p>
    """

    try:
        return templates.TemplateResponse(
            "base.html",
            {
                "request": request,
                "title": "Health Check",
                "content": content
            }
        )
    except Exception as e:
        logger.error(f"Failed to render health check UI: {e}")
        return HTMLResponse(
            status_code=500,
            content="<h2>Error</h2><p>Failed to load health check UI.</p>"
        )
