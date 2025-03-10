import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.common.config_loader import ConfigLoader
from src.common.json_config_loader import JsonConfigLoader

# Logging configuration
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Singleton-style manager that loads and caches configuration files.

    Attributes:
        _config (Any): General configuration cache
        _history_prompt_config (Dict[str, Any]): History prompt configuration cache
        _base_dir (Path): Base directory for configuration files
    """
    _config = None
    _history_prompt_config = None
    _base_dir = Path(__file__).resolve().parent

    @staticmethod
    def get_config() -> Any:
        """
        Loads and caches general configuration.

        Returns:
            Any: Loaded configuration object

        Raises:
            Exception: When an error occurs during configuration loading
        """
        if ConfigManager._config is None:
            try:
                logger.debug("Starting to load general configuration")
                config_loader = ConfigLoader()
                ConfigManager._config = config_loader.get_settings()
                logger.info(f"General configuration loading complete (environment: {config_loader.environment})")
            except FileNotFoundError as e:
                logger.error(f"Configuration file not found: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error occurred while loading configuration: {str(e)}")
                raise
        return ConfigManager._config

    @staticmethod
    def get_history_prompt_config() -> Dict[str, Any]:
        """
        Loads and caches history prompt configuration.

        Returns:
            Dict[str, Any]: Loaded history prompt configuration

        Raises:
            FileNotFoundError: When prompt configuration file cannot be found
            Exception: When an error occurs during configuration loading
        """
        if ConfigManager._history_prompt_config is None:
            try:
                logger.debug("Starting to load history prompt configuration")
                config = ConfigManager.get_config()

                # Safely extract history_prompt_path from configuration
                history_prompt_path = get_safe_value(
                    config,
                    ["prompt", "history_prompt_path"],
                    os.path.join(ConfigManager._base_dir, "../../config/prompts/default_history_prompt.json")
                )

                # Convert relative path to absolute path if needed
                if not os.path.isabs(history_prompt_path):
                    history_prompt_path = os.path.join(ConfigManager._base_dir, history_prompt_path)

                json_config_loader = JsonConfigLoader(history_prompt_path)
                ConfigManager._history_prompt_config = json_config_loader.get_settings()
                logger.info(f"History prompt configuration loading complete: {history_prompt_path}")
            except FileNotFoundError as e:
                logger.error(f"History prompt configuration file not found: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error occurred while loading history prompt configuration: {str(e)}")
                raise
        return ConfigManager._history_prompt_config

    @staticmethod
    def reset_cache() -> None:
        """
        Resets all cached configurations.
        """
        logger.debug("Resetting configuration cache")
        ConfigManager._config = None
        ConfigManager._history_prompt_config = None

    @staticmethod
    def get_section_value(section: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Gets configuration value for the specified section and key.

        Args:
            section (str): Configuration section name
            key (str): Configuration key name
            default (Any, optional): Default value to return when value is not found

        Returns:
            Any: Configuration value or default value
        """
        try:
            config = ConfigManager.get_config()
            section_obj = getattr(config, section.lower(), None)
            if section_obj is None:
                return default
            return getattr(section_obj, key, default)
        except Exception as e:
            logger.warning(f"Error occurred while getting key {key} from section {section}: {str(e)}")
            return default


class PromptManager:
    """
    Singleton-style manager that manages and caches ChatPromptTemplate instances.

    Attributes:
        _contextualize_q_prompt (ChatPromptTemplate): Template cache for query contextualization
    """
    _contextualize_q_prompt = None

    @staticmethod
    def get_contextualize_q_prompt() -> ChatPromptTemplate:
        """
        Creates or retrieves from cache a ChatPromptTemplate for query contextualization.

        This prompt template includes:
        - System prompt loaded from history prompt configuration
        - Placeholder for chat history
        - Human prompt containing user input

        Returns:
            ChatPromptTemplate: Cached or newly created prompt template

        Raises:
            KeyError: When required prompt configuration is missing
            Exception: When other errors occur
        """
        if PromptManager._contextualize_q_prompt is None:
            try:
                logger.debug("Starting to create contextualization prompt template")
                # Load history prompt configuration
                history_prompt_config = ConfigManager.get_history_prompt_config()

                # Safely extract required prompts
                system_prompt = get_safe_key(
                    history_prompt_config,
                    ["prompts", "contextualize_q_system_prompt"]
                )

                if not system_prompt:
                    logger.warning("System prompt is missing, using default")
                    system_prompt = ("Default system prompt. "
                                     "Answer the user's question considering the previous conversation.")

                # Create ChatPromptTemplate instance
                PromptManager._contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])
                logger.info("Contextualization prompt template creation complete")
            except KeyError as e:
                logger.error(f"Required prompt configuration missing: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error occurred while creating prompt template: {str(e)}")
                raise
        return PromptManager._contextualize_q_prompt

    @staticmethod
    def reset_cache() -> None:
        """
        Resets all cached prompt templates.
        """
        logger.debug("Resetting prompt template cache")
        PromptManager._contextualize_q_prompt = None


def get_safe_key(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    """
    Safely retrieves nested values from a dictionary.

    This function uses a list of keys to navigate through a dictionary, and returns
    the default value if a key is missing or if a non-dictionary value is encountered
    during navigation.

    Args:
        data (Dict[str, Any]): Dictionary to search
        keys (List[str]): List of keys specifying the path to the desired value
        default (Any, optional): Value to return if path doesn't exist. Defaults to None

    Returns:
        Any: Value at specified path or default value
    """
    current = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key, {})

    # Return default value if final value is an empty dictionary
    if current == {}:
        return default

    return current


def get_safe_value(obj: Any, attrs: List[str], default: Any = None) -> Any:
    """
    Safely retrieves nested attribute values from an object.

    This function uses a list of attributes to navigate through an object, and returns
    the default value if an attribute is missing. Works with dynamic object settings
    returned by ConfigLoader.

    Args:
        obj (Any): Object to search
        attrs (List[str]): List of attributes specifying the path to the desired value
        default (Any, optional): Value to return if path doesn't exist. Defaults to None

    Returns:
        Any: Value at specified path or default value
    """
    current = obj
    for attr in attrs:
        if current is None:
            return default
        current = getattr(current, attr, None)
        if current is None:
            return default

    return current
