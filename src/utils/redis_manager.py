import logging
from typing import Optional

import redis
from redis.asyncio import Redis as AsyncRedis, ConnectionPool as AsyncConnectionPool
from redis.exceptions import RedisError, ConnectionError, TimeoutError
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from src.common.config_loader import ConfigLoader

# Module level logger configuration
logger = logging.getLogger(__name__)

# Load configuration
config_loader = ConfigLoader()
settings = config_loader.get_settings()

# Configure retry decorator for handling connection and timeout errors
retry_decorator = retry(
    wait=wait_fixed(2),  # Wait 2 seconds between retries
    stop=stop_after_attempt(3),  # Maximum 3 retry attempts
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # Retry only on ConnectionError or TimeoutError
    reraise=True,  # Re-raise the exception after all retry attempts fail
)


class RedisManager:
    """
    Manager class for synchronous and asynchronous Redis connections

    This class is implemented as a singleton pattern and uses connection pools
    for efficient resource management. It includes automatic retry mechanisms for
    connection failures and manages connection states.
    """

    # Singleton instances and connection pools
    _sync_instance: Optional[redis.StrictRedis] = None
    _async_instance: Optional[AsyncRedis] = None
    _sync_pool = None
    _async_pool = None

    # Logging level configuration
    _debug_mode = False

    @classmethod
    def set_debug_mode(cls, enabled: bool = False):
        """
        Enable/disable debug mode

        Args:
            enabled (bool): Whether to enable debug mode (default: False)
        """
        cls._debug_mode = enabled

    @classmethod
    def _initialize_sync_pool(cls):
        """
        Initialize synchronous Redis connection pool

        Creates a synchronous Redis connection pool based on configured environment variables.
        Does not reinitialize if already initialized.
        """
        if cls._sync_pool is None:
            cls._sync_pool = redis.ConnectionPool(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                db=settings.redis.database,
                decode_responses=True,  # Decode responses to Python strings
                max_connections=10,  # Maximum number of connections in the pool
                socket_timeout=5,  # Socket operation timeout (seconds)
                retry_on_timeout=True,  # Retry on timeout
                health_check_interval=30,  # Connection health check interval (seconds)
            )
            if cls._debug_mode:
                logger.debug("Initialized synchronous Redis connection pool")

    @classmethod
    def _initialize_async_pool(cls):
        """
        Initialize asynchronous Redis connection pool

        Creates an asynchronous (asyncio compatible) Redis connection pool based on configured
        environment variables. Does not reinitialize if already initialized.
        """
        if cls._async_pool is None:
            cls._async_pool = AsyncConnectionPool(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password,
                db=settings.redis.database,
                decode_responses=True,  # Decode responses to Python strings
                max_connections=10,  # Maximum number of connections in the pool
                socket_timeout=5,  # Socket operation timeout (seconds)
                socket_connect_timeout=3,  # Connection attempt timeout (seconds)
                health_check_interval=30,  # Connection health check interval (seconds)
            )
            if cls._debug_mode:
                logger.debug("Initialized asynchronous Redis connection pool")

    @classmethod
    @retry_decorator
    def get_sync_connection(cls) -> Optional[redis.StrictRedis]:
        """
        Return synchronous Redis connection instance

        Creates a singleton instance of the synchronous Redis client and verifies the connection.
        If a connection error occurs, it retries and, after all retries fail, logs the error and returns None.

        Returns:
            Optional[redis.StrictRedis]: Redis client instance or None if connection fails

        Raises:
            RedisError: If connection fails after retries
        """
        if cls._sync_instance is None:
            cls._initialize_sync_pool()
            try:
                cls._sync_instance = redis.StrictRedis(connection_pool=cls._sync_pool)
                cls._sync_instance.ping()  # Test connection
                logger.info("Successfully connected to Redis (sync)")
            except RedisError as e:
                logger.error(f"Redis connection error (sync): {e}")
                cls._sync_instance = None
                raise  # Re-raise the exception to activate the retry mechanism
        return cls._sync_instance

    @classmethod
    @retry_decorator
    async def get_async_connection(cls) -> Optional[AsyncRedis]:
        """
        Return asynchronous Redis connection instance

        Creates a singleton instance of the asynchronous Redis client and verifies the connection.
        If a connection error occurs, it retries and, after all retries fail, logs the error and returns None.
        Optimized for reusability and minimal overhead.

        Returns:
            Optional[AsyncRedis]: Asynchronous Redis client instance or None if connection fails

        Raises:
            RedisError: If connection fails after retries
        """
        if cls._async_instance is None:
            cls._initialize_async_pool()
            try:
                cls._async_instance = AsyncRedis(connection_pool=cls._async_pool)
                await cls._async_instance.ping()  # Test connection
                logger.info("Successfully connected to Redis (async)")
            except RedisError as e:
                logger.error(f"Redis connection error (async): {e}")
                cls._async_instance = None
                raise  # Re-raise the exception to activate the retry mechanism
        return cls._async_instance

    @classmethod
    def close_sync_connection(cls):
        """
        Close synchronous Redis connection pool

        Disconnects all connections in the synchronous connection pool and resets the singleton instance.
        """
        if cls._sync_instance:
            try:
                cls._sync_pool.disconnect()
                if cls._debug_mode:
                    logger.debug("Closed Redis connection pool (sync)")
                else:
                    logger.info("Closed Redis connection pool (sync)")
            except Exception as e:
                logger.warning(f"Error closing Redis connection pool (sync): {e}")
            finally:
                cls._sync_instance = None
                cls._sync_pool = None

    @classmethod
    async def close_async_connection(cls):
        """
        Close asynchronous Redis connection pool

        Disconnects all connections in the asynchronous connection pool and resets the singleton instance.
        """
        if cls._async_instance:
            try:
                await cls._async_pool.disconnect()
                if cls._debug_mode:
                    logger.debug("Closed Redis connection pool (async)")
                else:
                    logger.info("Closed Redis connection pool (async)")
            except Exception as e:
                logger.warning(f"Error closing Redis connection pool (async): {e}")
            finally:
                cls._async_instance = None
                cls._async_pool = None

    @classmethod
    async def close_all_connections(cls):
        """
        Close all Redis connections

        Disconnects all connections in both synchronous and asynchronous connection pools
        and resets the singleton instances. Used for resource cleanup when the application terminates.
        """
        await cls.close_async_connection()
        cls.close_sync_connection()
        logger.info("All Redis connections closed")

    @classmethod
    def health_check(cls) -> bool:
        """
        Check synchronous Redis connection status

        Verifies that the current Redis connection is functioning properly.

        Returns:
            bool: True if the connection status is normal, False otherwise
        """
        try:
            connection = cls.get_sync_connection()
            if connection:
                connection.ping()
                if cls._debug_mode:
                    logger.debug("Redis health check passed (sync)")
                return True
            return False
        except RedisError:
            logger.warning("Redis health check failed (sync)")
            return False

    @classmethod
    async def async_health_check(cls) -> bool:
        """
        Check asynchronous Redis connection status

        Verifies that the current asynchronous Redis connection is functioning properly.

        Returns:
            bool: True if the connection status is normal, False otherwise
        """
        try:
            connection = await cls.get_async_connection()
            if connection:
                await connection.ping()
                if cls._debug_mode:
                    logger.debug("Redis health check passed (async)")
                return True
            return False
        except RedisError:
            logger.warning("Redis health check failed (async)")
            return False
