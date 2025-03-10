import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

from jsonschema import validate, ValidationError

# Logger configuration
logger = logging.getLogger(__name__)


class QueryCheckDict:
    """
    A class that processes JSON data and supports nested dictionary value access and validation.

    Main features:
    - Loading and validating JSON configuration files
    - Accessing data using nested keys (with caching functionality)
    - Providing default values when data is missing

    Attributes:
        config_path (Path): Path to the JSON configuration file
        schema (Optional[Dict]): Schema for validating JSON data
        json_data_dict (Dict): Loaded JSON data
        _path_cache (Dict): Cache for results of frequently used paths

    Example:
        >>> config = QueryCheckDict("config.json")
        >>> value = config.get_nested_value(["category", "subcategory", "key"])
        >>> prompt = config.get_prompt_data("en", "greeting", "formal")
    """

    def __init__(self, config_path: Union[str, Path], schema: Optional[Dict] = None):
        """
        Initialization method for QueryCheckDict class.

        Args:
            config_path (Union[str, Path]): Path to the JSON configuration file
            schema (Optional[Dict]): Schema for validating JSON data

        Raises:
            FileNotFoundError: When the file does not exist at the specified path
            ValueError: When a serious error occurs during initialization
        """
        self.config_path = Path(config_path)
        self.schema = schema
        self.json_data_dict = self._read_and_validate_config()

        # Initialize path cache
        self._path_cache: Dict[str, Any] = {}

        # Explicit error handling when data loading fails
        if not self.json_data_dict and not self._is_empty_config_valid():
            raise ValueError(f"Failed to successfully load '{config_path}' file.")

        # Pre-cache frequently used paths
        self._init_cache()

    @classmethod
    def _is_empty_config_valid(cls) -> bool:
        """
        Check if an empty configuration is valid in the current context.

        Returns:
            bool: True if empty config is allowed, False otherwise
        """
        # Determine if empty configuration is valid based on implementation
        # By default, empty configuration is not valid
        return False

    def _read_and_validate_config(self) -> Dict:
        """
        Read the JSON configuration file and validate it if a schema is provided.

        Returns:
            Dict: JSON data as a dictionary

        Raises:
            FileNotFoundError: When the file cannot be found
            json.JSONDecodeError: When the JSON file has an invalid format
            ValidationError: When the JSON data does not match the provided schema
        """
        try:
            # Check if file exists
            if not self.config_path.exists():
                logger.error(f"File does not exist: {self.config_path}")
                raise FileNotFoundError(f"File not found: {self.config_path}")

            # Read file
            with open(self.config_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Validate data if schema is provided
            if self.schema:
                self._validate_config(data)

            return data

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format in file '{self.config_path}': {e}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error occurred while processing config file '{self.config_path}': {e}")
            return {}

    def _validate_config(self, data: Dict) -> None:
        """
        Validate JSON data based on the schema.

        Args:
            data (Dict): JSON data to validate

        Raises:
            ValidationError: When data does not match the schema
        """
        try:
            validate(instance=data, schema=self.schema)
            logger.debug("JSON schema validation successful")
        except ValidationError as e:
            logger.error(f"JSON schema validation failed: {e}")
            raise

    def _init_cache(self) -> None:
        """
        Pre-cache frequently used paths.
        Caching is performed for first-level and second-level paths.
        """
        if not self.json_data_dict:
            return

        # Cache top-level keys (typically language codes)
        for key1, value1 in self.json_data_dict.items():
            if not isinstance(value1, dict):
                continue

            # Cache second-level keys (e.g., greetings, farewells, etc.)
            for key2, value2 in value1.items():
                cache_key = f"{key1}.{key2}"
                self._path_cache[cache_key] = value2

    def get_nested_value_raw(self, keys: List[str], default: Any = None) -> Any:
        """
        Basic nested value lookup method that does not use caching.
        Used internally only.

        Args:
            keys (List[str]): List representing nested keys
            default (Any, optional): Default value to return if key is not found

        Returns:
            Any: Value corresponding to the key or the default value
        """
        if not keys:
            return default

        data = self.json_data_dict
        for key in keys:
            if not isinstance(data, dict) or key not in data:
                return default
            data = data[key]

        return data

    @lru_cache(maxsize=128)
    def get_nested_value_cached(self, keys: Tuple[str, ...], default: Any = None) -> Any:
        """
        Method to retrieve a value from a nested dictionary using LRU caching.

        Args:
            keys (Tuple[str, ...]): Tuple representing nested keys (immutable type used for caching)
            default (Any, optional): Default value to return if key is not found

        Returns:
            Any: Value corresponding to the key or the default value
        """
        if not keys:
            return default

        # Check if already in cache
        cache_key = ".".join(keys)
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Access data if not in cache
        result = self.get_nested_value_raw(list(keys), default)

        # Cache important paths (only for key lengths <= 3 to prevent cache overflow)
        if len(keys) <= 3:
            self._path_cache[cache_key] = result

        return result

    def get_nested_value(self, keys: List[str], default: Any = None) -> Any:
        """
        Method to safely retrieve a value from a nested dictionary.
        Uses caching internally.

        Args:
            keys (List[str]): List representing nested keys
            default (Any, optional): Default value to return if key is not found

        Returns:
            Any: Value corresponding to the key or the default value

        Example:
            >>> config = QueryCheckDict("config.json")
            >>> value = config.get_nested_value(["user", "preferences", "theme"], "light")
        """
        if not keys:
            return default

        # Convert list to tuple to make it a cacheable type
        return self.get_nested_value_cached(tuple(keys), default)

    def get_dict_data(self, lang_key: str, dict_key: str, default: Any = "") -> Any:
        """
        Retrieve a value from JSON data using two keys.
        Utilizes internal caching to optimize performance.

        Args:
            lang_key (str): First key (language key)
            dict_key (str): Second key (data key)
            default (Any, optional): Default value to return if key is not found

        Returns:
            Any: Value corresponding to the key or the default value

        Example:
            >>> config = QueryCheckDict("translations.json")
            >>> greeting = config.get_dict_data("en", "welcome_message", "Welcome!")
        """
        # Try direct lookup from cache
        cache_key = f"{lang_key}.{dict_key}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Access data if not in cache
        result = self.get_nested_value_raw([lang_key, dict_key], default)

        # Cache the result
        self._path_cache[cache_key] = result
        return result

    def get_prompt_data(self, search_key1: str, search_key2: str, rag_sys_info: str, default: Any = "") -> Any:
        """
        Retrieve a value from nested JSON data using three keys.
        Utilizes internal caching to optimize performance.

        Args:
            search_key1 (str): First key (main category key)
            search_key2 (str): Second key (subcategory key)
            rag_sys_info (str): Third key (information key)
            default (Any, optional): Default value to return if key is not found

        Returns:
            Any: Value corresponding to the key or the default value

        Example:
            >>> config = QueryCheckDict("prompts.json")
            >>> prompt = config.get_prompt_data("chatbot", "greeting", "formal", "Hello, how may I assist you today?")
        """
        # Create cache key
        cache_key = f"{search_key1}.{search_key2}.{rag_sys_info}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Access data if not in cache
        result = self.get_nested_value_raw([search_key1, search_key2, rag_sys_info], default)

        # Cache the result
        self._path_cache[cache_key] = result
        return result

    def reload_config(self) -> bool:
        """
        Reload the configuration file and reset the cache.
        Used when the file has been changed externally.

        Returns:
            bool: Whether the reload was successful

        Example:
            >>> config = QueryCheckDict("config.json")
            >>> # After the file has been changed externally
            >>> if config.reload_config():
            >>>     print("Configuration has been successfully reloaded.")
        """
        try:
            self.json_data_dict = self._read_and_validate_config()
            # Reset cache
            self._path_cache.clear()
            get_nested_value_cached.cache_clear()  # Clear lru_cache
            # Reinitialize cache
            self._init_cache()
            return True
        except Exception as e:
            logger.error(f"Failed to reload configuration file: {e}")
            return False

    def get_all_data(self) -> Dict:
        """
        Return a copy of all loaded JSON data.

        Returns:
            Dict: A copy of the entire configuration data

        Example:
            >>> config = QueryCheckDict("config.json")
            >>> all_data = config.get_all_data()
        """
        return self.json_data_dict.copy() if self.json_data_dict else {}


# Define lru_cache at function level (referenced in internal methods)
@lru_cache(maxsize=128)
def get_nested_value_cached(json_data_dict: Dict, keys: Tuple[str, ...], default: Any = None) -> Any:
    """
    Nested value lookup function utilizing LRU caching.
    Defined outside the class to be referenced by instance methods.
    """
    if not keys:
        return default

    data = json_data_dict
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]

    return data
