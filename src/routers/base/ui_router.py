import logging

from fastapi import APIRouter, Request, Depends
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils.auth_utils import verify_authentication  # Authentication utility

# ================================
#         Logging Configuration
# ================================
logger = logging.getLogger(__name__)

# ================================
#         Template Settings
# ================================
templates = Jinja2Templates(directory="templates")  # Setting up Jinja2 templates

# ================================
#         UI Router Setup
# ================================
ui_router = APIRouter()


@ui_router.get(
    "/menu",
    include_in_schema=False,
    dependencies=[Depends(verify_authentication)],
    response_class=HTMLResponse
)
async def menu_page(request: Request):
    """
    Renders the main administration menu page with navigation options.

    This secured endpoint provides a central navigation hub for administrative
    functions. It requires authentication before access is granted, ensuring
    that only authorized personnel can access administrative interfaces.

    The menu presents links to key administrative tools including:
    - API documentation (Swagger UI)
    - System health status
    - Redis connection monitoring

    Args:
        request (Request): The incoming HTTP request object needed for template rendering

    Returns:
        HTMLResponse: A rendered HTML page containing the administration menu
                     with styled navigation buttons

    Security:
        Requires successful authentication via the verify_authentication dependency
    """
    menu_content = """
        <div style="text-align: center;">
            <a href='/custom-swagger' class='btn'>Swagger UI</a><br>
            <a href='/health_check_ui' class='btn'>Health Check</a><br>
            <a href='/health_redis' class='btn'>Redis Health Check</a>
        </div>
    """
    return templates.TemplateResponse(
        "base.html",
        {
            "request": request,
            "title": "MAI Chat LLM Wrapper Admin",
            "content": menu_content
        }
    )


@ui_router.get(
    "/custom-swagger",
    include_in_schema=False,
    dependencies=[Depends(verify_authentication)],
    response_class=HTMLResponse
)
async def custom_swagger_page(request: Request):
    """
    Renders a customized Swagger UI documentation page with enhanced navigation.

    This endpoint provides access to the API documentation through Swagger UI,
    with additional custom elements for improved usability. It dynamically
    determines the OpenAPI schema URL based on the current request context
    and injects a navigation button to return to the main menu.

    The page requires authentication, ensuring that API documentation is only
    accessible to authorized personnel.

    Args:
        request (Request): The incoming HTTP request object used to determine
                          the base URL for the OpenAPI schema

    Returns:
        HTMLResponse: A rendered Swagger UI HTML page with custom navigation elements

    Raises:
        500 Error: If rendering fails due to template errors or other exceptions

    Security:
        Requires successful authentication via the verify_authentication dependency
    """
    try:
        # Dynamically construct the OpenAPI schema URL based on the current request context
        openapi_url = f"{str(request.base_url).rstrip('/')}/openapi.json"

        # Generate the standard Swagger UI HTML content
        html_content = get_swagger_ui_html(
            openapi_url=openapi_url,
            title="Swagger UI",
        ).body.decode('utf-8')

        # Define a "Back to Menu" button with fixed positioning and styling
        button_html = (
            '<a href="/menu" '
            'style="position: fixed; top: 10px; right: 10px; padding: 10px; background: #007bff; '
            'color: white; text-decoration: none; border-radius: 4px; font-weight: bold;">'
            '⬅ Back to Menu</a>'
        )

        # Inject the navigation button into the HTML content just before the closing body tag
        html_content = html_content.replace('</body>', f'{button_html}</body>')

        return HTMLResponse(content=html_content)
    except Exception as e:
        # Log detailed error information for troubleshooting
        logger.error(f"Failed to render Swagger UI: {e}")

        # Return a simple error page to the user
        return HTMLResponse(
            content="<h2>Error</h2><p>Failed to load Swagger UI.</p>",
            status_code=500
        )
