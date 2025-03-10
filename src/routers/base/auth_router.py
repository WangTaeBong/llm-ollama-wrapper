import time

from fastapi import APIRouter, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer, BadSignature

from src.common.config_loader import ConfigLoader

# Load application settings
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Session management: Secret key should be securely stored in the config
SESSION_SECRET = settings.security.session_secret
serializer = URLSafeSerializer(SESSION_SECRET)

# Authentication configuration: Password should also be stored securely
ACCESS_PASSWORD = settings.security.access_password

# Template engine setup
templates = Jinja2Templates(directory="templates")

# Define authentication router
auth_router = APIRouter()


@auth_router.get("/login", include_in_schema=False, response_class=HTMLResponse)
async def login_page(request: Request):
    """
    Serve the login page where users can enter their password.

    Args:
        request (Request): The incoming request object.

    Returns:
        TemplateResponse: The rendered login page.
    """
    return templates.TemplateResponse("login.html", {"request": request})


@auth_router.post("/login", include_in_schema=False)
async def login(request: Request, password: str = Form(...)):
    """
    Process the login form submission.

    This function validates the submitted password against the configured access password.
    If the password is correct, it creates a secure session cookie containing:
      - An "authenticated" flag set to True.
      - A "login_time" timestamp marking the login moment.
    The cookie is configured with security attributes (HTTP-only, secure, and same-site set to "lax")
    to mitigate common web security risks. Upon successful authentication, the user is redirected
    to the '/menu' page. If authentication fails, the login page is re-rendered with an error message.

    Args:
        request (Request): The incoming HTTP request containing the form data.
        password (str): The password submitted through the login form.

    Returns:
        RedirectResponse or TemplateResponse: A redirect to '/menu' on successful login, or
        a re-rendered login page with an error message if the password is invalid.
    """
    # Verify the submitted password against the configured access password.
    if password == ACCESS_PASSWORD:
        # Create session data with authentication status and login timestamp.
        session_data = {
            "authenticated": True,
            "login_time": time.time()  # Record the current time as the login time.
        }
        # Serialize the session data to generate a secure session cookie.
        session_cookie = serializer.dumps(session_data)

        # Create a redirect response to the menu page.
        response = RedirectResponse(url="/menu", status_code=status.HTTP_302_FOUND)

        # Set the session cookie with security attributes.
        response.set_cookie(
            key="session",
            value=session_cookie,
            httponly=True,  # Prevent JavaScript access to the cookie to mitigate XSS attacks.
            secure=False,  # Ensure the cookie is only transmitted over HTTPS.
            samesite="lax"  # Restrict cross-site requests to help prevent CSRF attacks.
        )
        # Return the redirect response with the session cookie.
        return response

    # If the password does not match, re-render the login page with an error message.
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": "Invalid password. Try again."},
    )


@auth_router.get("/logout", include_in_schema=False, response_class=RedirectResponse)
async def logout():
    """
    Handle user logout by clearing the authentication session.

    Returns:
        RedirectResponse: Redirects to the login page after clearing the session cookie.
    """
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("session")  # Remove authentication cookie
    return response


@auth_router.get("/session-check", include_in_schema=False, response_class=HTMLResponse)
async def session_check(request: Request):
    """
    Check if the user is authenticated by verifying the session cookie.

    Args:
        request (Request): The incoming request object.

    Returns:
        dict: A JSON response indicating authentication status.
    """
    session_cookie = request.cookies.get("session")
    if not session_cookie:
        return {"authenticated": False}

    try:
        session_data = serializer.loads(session_cookie)
        if session_data.get("authenticated"):
            return {"authenticated": True}
    except BadSignature:
        return {"authenticated": False}

    return {"authenticated": False}
