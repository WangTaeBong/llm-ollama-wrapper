import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, AsyncGenerator
from contextlib import asynccontextmanager

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


# vLLM 스트리밍 응답 표준 포맷을 처리하기 위한 보조 함수
def _parse_vllm_streaming_chunk(raw_chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    vLLM 스트리밍 응답 형식을 표준화된 내부 형식으로 변환합니다.

    Args:
        raw_chunk: vLLM API에서 반환된 원시 청크 데이터

    Returns:
        Dict[str, Any]: 표준화된 청크 데이터
    """
    # vLLM 응답 형식 처리
    if 'text' in raw_chunk:
        # 생성된 텍스트 포함
        return {
            'new_text': raw_chunk['text'],
            'finished': raw_chunk.get('finished', False)
        }
    elif 'generated_text' in raw_chunk:
        # 전체 생성 텍스트 포함
        return {
            'new_text': raw_chunk['generated_text'],
            'finished': True
        }
    elif 'delta' in raw_chunk:
        # OpenAI 호환 델타 형식
        return {
            'new_text': raw_chunk['delta'].get('content', ''),
            'finished': raw_chunk.get('finished', False)
        }
    elif 'error' in raw_chunk:
        # 오류 형식
        return {
            'error': True,
            'message': raw_chunk.get('message', 'Unknown error'),
            'finished': True
        }
    else:
        # 기타 형식
        return raw_chunk


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
                session_id = body_data.get('request_id', 'still_unknown')

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

    @classmethod
    async def restapi_stream_async(cls, session_id: str, url: str, data) -> AsyncGenerator[Dict, None]:
        """
        Makes a streaming POST request to a REST API endpoint with improved batching and buffering.

        Optimized for:
        1. Better error handling with detailed context
        2. Adaptive timeouts based on connection state
        3. Stream processing optimization with batched reading
        4. Performance monitoring with detailed tracing

        Args:
            session_id: Session identifier for logging and correlation
            url: Target API endpoint
            data: Request data object

        Yields:
            Dict: Streaming response chunks
        """
        logger.debug(f"[{session_id}] Starting optimized streaming request to {url}")
        start_time = time.time()
        received_bytes = 0
        chunk_count = 0

        # Timeout settings - we need to be generous for streaming
        connect_timeout = 5.0  # Connection timeout
        read_timeout = 30.0  # Read timeout for chunks
        total_timeout = 300.0  # Overall request timeout (5 minutes max for long generations)

        # SSE parsing constants
        sse_prefix = "data: "
        sse_done_marker = "[DONE]"

        # Create timeout configuration
        timeout = aiohttp.ClientTimeout(
            connect=connect_timeout,
            sock_read=read_timeout,
            total=total_timeout
        )

        try:
            # 중요: ClientSession 생성 시 명시적으로 close_timeout을 길게 설정
            async with aiohttp.ClientSession(timeout=timeout,
                                             connector=aiohttp.TCPConnector(force_close=False),
                                             raise_for_status=True) as session:
                # Convert request data to JSON
                request_data = data.dict() if hasattr(data, "dict") else data

                logger.debug(f"[{session_id}] Sending streaming request to {url}")

                # Measure connection setup time
                connection_start = time.time()

                # Make HTTP request with detailed headers
                headers = {
                    "Content-Type": "application/json; charset=utf-8",
                    "Accept": "text/event-stream",
                    "X-Request-ID": session_id,
                }

                # Start request with initial timeout
                async with session.post(url, json=request_data, headers=headers) as response:
                    # Measure time to first byte
                    ttfb = time.time() - connection_start
                    logger.debug(f"[{session_id}] Connection established, TTFB: {ttfb:.4f}s")

                    # Check response status
                    if response.status != 200:
                        error_msg = await response.text()
                        logger.error(f"[{session_id}] Stream request failed: HTTP {response.status}, {error_msg}")
                        yield {"error": True, "message": f"HTTP error {response.status}: {error_msg}"}
                        return

                    # Set up buffer and state for SSE processing
                    buffer = ""

                    # Read response in efficient chunks
                    reader = response.content
                    is_first_chunk = True

                    # Use content.read() with a chunk size instead of iter_chunked
                    # This avoids coroutine issues in some aiohttp versions
                    while True:
                        raw_chunk = await reader.read(8192)  # 8KB chunks

                        # Exit loop when no more data
                        if not raw_chunk:
                            break

                        # Track first chunk received
                        if is_first_chunk:
                            logger.debug(
                                f"[{session_id}] First chunk received after {time.time() - connection_start:.4f}s")
                            is_first_chunk = False

                        # Track metrics
                        chunk_size = len(raw_chunk)
                        received_bytes += chunk_size
                        chunk_count += 1

                        # Convert bytes to string and add to buffer
                        text_chunk = raw_chunk.decode('utf-8')
                        buffer += text_chunk

                        # Process complete SSE events from buffer
                        while "\n\n" in buffer:
                            # Split at the first double newline (SSE event separator)
                            event, buffer = buffer.split("\n\n", 1)
                            event = event.strip()

                            # Skip empty events
                            if not event:
                                continue

                            # Process "data:" prefixed lines
                            if event.startswith(sse_prefix):
                                data_str = event[len(sse_prefix):]

                                # Check for SSE completion marker
                                if data_str == sse_done_marker:
                                    logger.debug(f"[{session_id}] SSE stream complete marker received")
                                    yield sse_done_marker
                                    continue

                                # Parse JSON data
                                try:
                                    data_obj = json.loads(data_str)
                                    yield data_obj
                                except json.JSONDecodeError as e:
                                    logger.warning(
                                        f"[{session_id}] JSON decode error: {str(e)}, data: {data_str[:100]}...")
                                    # Return raw data on parse error
                                    yield {"error": False, "new_text": data_str, "raw": True}
                            else:
                                # Non-data event (comments, etc.)
                                logger.debug(f"[{session_id}] Non-data SSE event: {event[:50]}...")

                    # Process any remaining data in buffer
                    if buffer.strip():
                        if buffer.startswith(sse_prefix):
                            data_str = buffer[len(sse_prefix):]
                            try:
                                data_obj = json.loads(data_str)
                                yield data_obj
                            except json.JSONDecodeError:
                                # Return raw data on parse error
                                yield {"error": False, "new_text": data_str, "raw": True}
                        else:
                            # Handle non-SSE format data
                            logger.debug(f"[{session_id}] Non-SSE data in final buffer")
                            yield {"error": False, "new_text": buffer, "raw": True}

                # Log streaming completion metrics
                duration = time.time() - start_time
                logger.info(
                    f"[{session_id}] Stream completed: {received_bytes} bytes, {chunk_count} chunks, "
                    f"{duration:.2f}s, {received_bytes / 1024 / duration:.2f} KB/s"
                )

        except asyncio.TimeoutError as e:
            logger.error(f"[{session_id}] Stream request timeout after {time.time() - start_time:.2f}s")
            yield {"error": True, "message": f"Request timed out: {str(e)}"}

        except aiohttp.ClientError as e:
            logger.error(f"[{session_id}] HTTP error during streaming: {str(e)}")
            yield {"error": True, "message": f"HTTP error: {str(e)}"}

        except Exception as e:
            logger.error(f"[{session_id}] Unexpected error during streaming: {str(e)}", exc_info=True)
            yield {"error": True, "message": f"Error: {str(e)}"}

        finally:
            logger.debug(f"[{session_id}] Stream request completed")

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
