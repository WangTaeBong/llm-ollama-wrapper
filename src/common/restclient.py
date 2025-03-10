import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator

import aiohttp
import requests
from requests.adapters import HTTPAdapter, Retry

from src.common.config_loader import ConfigLoader

# ================================
#         Configuration
# ================================

# Load settings from configuration file
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Configure logging for this module
logger = logging.getLogger(__name__)


class RestClient:
    """
    A comprehensive REST client for making synchronous and asynchronous HTTP requests.

    This client provides a robust interface for API communication with features including:
    - Automatic retry mechanisms for transient failures
    - Consistent error handling across sync and async operations
    - Configurable timeout settings
    - Support for default and custom headers
    - SSL verification configuration
    - Logging capabilities with data truncation for sensitive information

    The client supports both synchronous operations (using requests) and
    asynchronous operations (using aiohttp) for integration with FastAPI applications.
    """

    _aio_session: Optional[aiohttp.ClientSession] = None  # Shared session for FastAPI async operations
    ssl_enabled = settings.ssl.use_https  # Global SSL verification setting

    def __init__(self, default_headers: Optional[Dict[str, str]] = None):
        """
        Initialize a new RestClient instance with customizable headers.

        Sets up an internal requests.Session with retry configuration to handle
        transient network failures and server errors automatically.

        Args:
            default_headers (Optional[Dict[str, str]]): Default HTTP headers to include
                in all requests. Defaults to Content-Type: application/json if not provided.
        """
        self.default_headers = default_headers or {"Content-Type": "application/json; charset=utf-8"}
        self.session = requests.Session()

        # Configure automatic retries for resilient HTTP requests
        retries = Retry(
            total=3,  # Total number of retry attempts
            backoff_factor=0.3,  # Exponential backoff multiplier
            status_forcelist=[500, 502, 503, 504],  # Retry on these server error status codes
            allowed_methods={"GET", "POST"}  # HTTP methods to retry
        )
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    @classmethod
    def truncate_log_data(cls, data_str, max_length=200):
        """
        Truncate log data to limit its length for better log readability and privacy.

        Prevents excessively long payloads or responses from flooding log files
        while preserving the start of the data for debugging purposes.

        Args:
            data_str (str): The string data to be truncated.
            max_length (int): Maximum allowed length before truncation. Defaults to 200 characters.

        Returns:
            str: Truncated string with length information if truncation occurred.
        """
        if data_str and len(data_str) > max_length:
            return f"{data_str[:max_length]}... [total {len(data_str)} characters]"
        return data_str

    @staticmethod
    def _prepare_body(body: Any) -> str:
        """
        Serialize the request body to a JSON string for HTTP transmission.

        Handles multiple input types:
        - Dictionary objects are serialized directly to JSON
        - Objects with a .json() method (like Pydantic models) are processed through that method first

        Args:
            body (Any): The data to serialize into the request body.
                        Can be a dictionary or an object with a .json() method.

        Returns:
            str: The JSON-encoded string representation of the request body.

        Raises:
            ValueError: If the body cannot be properly serialized to JSON or is in an unsupported format.
        """
        try:
            if isinstance(body, dict):
                return json.dumps(body, ensure_ascii=False)
            elif hasattr(body, "json") and callable(body.json):
                data = json.loads(body.json())
                return json.dumps(data, ensure_ascii=False)
            else:
                raise ValueError("The body must be a dictionary or have a callable `json()` method.")
        except Exception as e:
            raise ValueError(f"Failed to prepare body: {e}") from e

    def restapi_post(self, url: str, body: Any, headers: Optional[Dict[str, str]] = None,
                     timeout: int = 120) -> requests.Response:
        """
        Send a synchronous HTTP POST request to the specified endpoint.

        This method handles the complete request lifecycle including:
        - JSON serialization of the request body
        - Header configuration and merging with defaults
        - Timeout management
        - SSL verification according to global settings
        - Error handling for timeouts, connection failures, and HTTP status errors

        Args:
            url (str): The full URL endpoint to send the request to.
            body (Any): The request payload to be serialized as JSON.
            headers (Optional[Dict[str, str]]): Custom headers to be sent with this specific request.
                                              These will override any default headers with the same name.
            timeout (int): Maximum time in seconds to wait for server response before raising a timeout error.
                           Default is 120 seconds.

        Returns:
            requests.Response: The complete HTTP response object if the request is successful.

        Raises:
            RuntimeError: Encapsulates various network and HTTP errors with descriptive messages:
                - Timeout errors when the server doesn't respond within the timeout period
                - Connection errors when the server is unreachable
                - HTTP status errors (4xx, 5xx) from the server
                - Other unexpected request failures
        """
        try:
            # Prepare the request body by converting to JSON format
            body_data = self._prepare_body(body)

            # Send the POST request with appropriate headers and settings
            response = self.session.post(
                url,
                headers=headers or self.default_headers,
                data=body_data,
                timeout=timeout,
                verify=self.ssl_enabled
            )

            # Raise an HTTPError for 4xx and 5xx status codes
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            # Handle timeout errors when the server takes too long to respond
            raise RuntimeError(f"⏳ Request Timeout: POST {url} exceeded {timeout} seconds.")
        except requests.exceptions.ConnectionError:
            # Handle network-related errors (e.g., no internet, server down)
            raise RuntimeError(f"❌ Connection Error: Unable to reach {url}. Check network or server status.")
        except requests.exceptions.RequestException as e:
            # Handle any other unexpected request failures
            raise RuntimeError(f"🚨 Unexpected Error in POST {url}: {str(e)}") from e

    def restapi_get(self, url: str, timeout: int = 60) -> requests.Response:
        """
        Send a synchronous HTTP GET request to the specified endpoint.

        This method handles the complete request lifecycle including:
        - Header configuration with defaults
        - Timeout management
        - SSL verification according to global settings
        - Automatic redirect following
        - Error handling for timeouts, connection failures, and HTTP status errors

        Args:
            url (str): The full URL endpoint to send the request to.
            timeout (int): Maximum time in seconds to wait for server response before raising a timeout error.
                           Default is 60 seconds.

        Returns:
            requests.Response: The complete HTTP response object if the request is successful.

        Raises:
            RuntimeError: Encapsulates various network and HTTP errors with descriptive messages:
                - Timeout errors when the server doesn't respond within the timeout period
                - Connection errors when the server is unreachable
                - HTTP status errors (4xx, 5xx) from the server
                - Other unexpected request failures
        """
        try:
            # Send the GET request with appropriate headers and settings
            response = self.session.get(
                url,
                headers=self.default_headers,
                allow_redirects=True,
                timeout=timeout,
                verify=self.ssl_enabled
            )

            # Raise an HTTPError for 4xx and 5xx status codes
            response.raise_for_status()
            return response

        except requests.exceptions.Timeout:
            # Handle timeout errors when the server takes too long to respond
            raise RuntimeError(f"⏳ Request Timeout: GET {url} exceeded {timeout} seconds.")
        except requests.exceptions.ConnectionError:
            # Handle network-related errors (e.g., no internet, server down)
            raise RuntimeError(f"❌ Connection Error: Unable to reach {url}. Check network or server status.")
        except requests.exceptions.RequestException as e:
            # Handle any other unexpected request failures
            raise RuntimeError(f"🚨 Unexpected Error in GET {url}: {str(e)}") from e

    async def restapi_post_async(self, url: str, body: Any) -> Dict[str, Any]:
        """
        Send an asynchronous HTTP POST request using the FastAPI-managed aiohttp session.

        This method provides asynchronous HTTP capabilities that integrate with FastAPI's
        event loop and connection pool management. It includes comprehensive error handling
        and logging for debugging purposes.

        Note: The global aiohttp.ClientSession must be initialized before using this method,
        typically through the set_global_session class method during application startup.

        Args:
            url (str): The full URL endpoint to send the request to.
            body (Any): The request payload to be serialized as JSON.

        Returns:
            Dict[str, Any]: The JSON-decoded response body if the request is successful.

        Raises:
            RuntimeError: Encapsulates various network and HTTP errors with descriptive messages:
                - Session initialization errors if _aio_session is not set
                - Timeout errors when the server doesn't respond within 120 seconds
                - Connection errors when the server is unreachable
                - HTTP status errors (4xx, 5xx) from the server
                - JSON parsing errors for invalid response formats
                - Other unexpected failures during the async request lifecycle
        """
        if RestClient._aio_session is None:
            raise RuntimeError("🚨 aiohttp.ClientSession is not initialized. Ensure FastAPI is running.")

        try:
            # Prepare the request body as a JSON string
            body_data_str = self._prepare_body(body)

            # Convert back to Python object since aiohttp's json parameter expects a Python object (e.g., dict)
            # not a pre-serialized JSON string
            body_data = json.loads(body_data_str)

            # Extract session ID for logging purposes, with fallback paths for different payload structures
            session_id = body_data.get('meta', {}).get('session_id', 'unknown_session')
            # If the value is 'unknown_session', try an alternative path
            if session_id == 'unknown_session':
                session_id = body_data.get('session_id', 'still_unknown')

            # Log the request with truncated body for debugging (ensures non-ASCII characters display correctly)
            logger.debug(f"[{session_id}] Sending request to {url} with body: "
                         f"{self.truncate_log_data(body_data_str)}")

            # Send an asynchronous POST request using the aiohttp session
            async with RestClient._aio_session.post(
                    url,
                    json=body_data,
                    headers=self.default_headers,
                    timeout=120,
                    ssl=self.ssl_enabled
            ) as response:
                # Raise an error for HTTP status codes in the 4xx and 5xx range
                response.raise_for_status()
                # Parse and return the JSON response body
                return await response.json()

        except aiohttp.ClientResponseError as http_err:
            # Handle HTTP response errors (e.g., 404 Not Found, 500 Internal Server Error)
            raise RuntimeError(f"🔴 HTTP Error: POST {url} returned status {http_err.status} - {http_err.message}")
        except aiohttp.ClientConnectorError:
            # Handle connection errors when the server is unreachable
            raise RuntimeError(f"❌ Connection Error: Unable to reach {url}. Check network or server status.")
        except asyncio.TimeoutError:
            # Handle timeout errors when the server takes too long to respond
            raise RuntimeError(f"⏳ Request Timeout: POST {url} exceeded 120 seconds.")
        except aiohttp.ClientError as e:
            # Handle any other client-side aiohttp error
            raise RuntimeError(f"🚨 Async Client Error in POST {url}: {str(e)}") from e
        except Exception as e:
            # Handle unexpected errors during the request lifecycle
            raise RuntimeError(f"🚨 Unexpected error during async POST {url}: {str(e)}") from e

    async def restapi_stream_async(self, session_id: str, url: str, body: Any) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send an asynchronous streaming HTTP POST request using the FastAPI-managed aiohttp session.
        """
        if RestClient._aio_session is None:
            raise RuntimeError("🚨 aiohttp.ClientSession is not initialized. Ensure FastAPI is running.")

        try:
            # Prepare the request body as a JSON string
            body_data_str = self._prepare_body(body)

            # Convert back to Python object since aiohttp's json parameter expects a Python object
            body_data = json.loads(body_data_str)

            if session_id is None:
                session_id = body_data.get('session_id', 'still_unknown')

            # Log the request with truncated body for debugging
            logger.debug(f"[{session_id}] Sending streaming request to {url} with body: "
                         f"{self.truncate_log_data(body_data_str)}")

            # Send an asynchronous streaming POST request using the aiohttp session
            async with RestClient._aio_session.post(
                    url,
                    json=body_data,
                    headers=self.default_headers,
                    timeout=120,
                    ssl=self.ssl_enabled
            ) as response:
                # Raise an error for HTTP status codes in the 4xx and 5xx range
                response.raise_for_status()

                # 디버그 로깅 추가
                logger.debug(f"[{session_id}] Response headers: {response.headers}")
                logger.debug(f"[{session_id}] Response content type: {response.headers.get('Content-Type', 'Unknown')}")

                # Buffer to accumulate chunks
                buffer = bytearray()

                # Process streaming response
                async for chunk in response.content:
                    buffer.extend(chunk)

                    # Process complete lines
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        chunk_str = line.decode('utf-8').strip()

                        # data: 접두사 제거
                        if chunk_str.startswith('data: '):
                            chunk_str = chunk_str.replace('data: ', '')

                            try:
                                chunk_data = json.loads(chunk_str)

                                # 로깅 (디버그 레벨)
                                logger.debug(
                                    f"[{session_id}] Streaming chunk received: {self.truncate_log_data(chunk_str)}")

                                yield chunk_data

                                # [DONE] 청크로 스트리밍 종료 확인
                                if chunk_str == '[DONE]':
                                    break

                            except json.JSONDecodeError:
                                # [DONE] 케이스는 무시
                                if chunk_str == '[DONE]':
                                    logger.debug(f"[{session_id}] Received [DONE] marker, ending stream")
                                    break
                                logger.warning(f"[{session_id}] Invalid JSON chunk: {chunk_str}")

                # Process any remaining buffer
                if buffer:
                    try:
                        chunk_str = buffer.decode('utf-8').strip()
                        if chunk_str.startswith('data: '):
                            chunk_str = chunk_str.replace('data: ', '')
                            chunk_data = json.loads(chunk_str)
                            yield chunk_data
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass

        except aiohttp.ClientResponseError as http_err:
            # Handle HTTP response errors
            raise RuntimeError(f"🔴 HTTP Error: POST {url} returned status {http_err.status} - {http_err.message}")
        except aiohttp.ClientConnectorError:
            # Handle connection errors when the server is unreachable
            raise RuntimeError(f"❌ Connection Error: Unable to reach {url}. Check network or server status.")
        except asyncio.TimeoutError:
            # Handle timeout errors when the server takes too long to respond
            raise RuntimeError(f"⏳ Request Timeout: POST {url} exceeded 120 seconds.")
        except aiohttp.ClientError as e:
            # Handle any other client-side aiohttp error
            raise RuntimeError(f"🚨 Async Client Error in streaming POST {url}: {str(e)}") from e
        except Exception as e:
            # Handle unexpected errors during the request lifecycle
            raise RuntimeError(f"🚨 Unexpected error during async streaming POST {url}: {str(e)}") from e

    @classmethod
    def set_global_session(cls, session: aiohttp.ClientSession):
        """
        Set the shared aiohttp.ClientSession for all asynchronous requests.

        This method should be called during application startup (e.g., in FastAPI's
        startup event handler) to initialize the global session used by all async
        request methods.

        Args:
            session (aiohttp.ClientSession): A properly configured aiohttp session instance
                that will be shared across all RestClient instances for async operations.
        """
        cls._aio_session = session


# ================================
#       Global Instance
# ================================

# Initialize a global RestClient instance for application-wide use
rc = RestClient()
