import os
from configparser import ConfigParser
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml  # Requires PyYAML (install via pip install pyyaml)


def _is_list_type_key(key: str) -> bool:
    """
    Check if a value is a list type based on the key name.

    Args:
        key (str): Configuration key to check

    Returns:
        bool: True if the key indicates a list type, False otherwise
    """
    # Key name patterns that indicate lists
    list_key_patterns = [
        "_hosts", "_origins", "_domains", "_urls", "_paths",
        "_allowed", "_denied", "_list", "_items", "_headers",
        "_keys", "_names", "urls", "hosts", "origins", "domains"
    ]

    lower_key = key.lower()
    return any(pattern in lower_key for pattern in list_key_patterns)


def _parse_list_value(value: str) -> List[str]:
    """
    Convert a comma-separated string into a list of strings.

    Args:
        value (str): Comma-separated string

    Returns:
        List[str]: List of individual items
    """
    if not value or value.strip() == "":
        return []

    # Split by commas and trim each item
    items = [item.strip() for item in value.split(",")]

    # Remove empty items
    return [item for item in items if item]


class ConfigLoader:
    """
    ConfigLoader: A utility class for loading and parsing application configuration files.
    """

    def __init__(self):
        """
        Initialize ConfigLoader:
          - Detect the execution environment
          - Load the relevant configuration files
        """
        self.base_dir = Path(__file__).resolve().parent
        self.environment = self.get_environment()
        self.config = {}  # Final configuration is stored as a dictionary
        self.settings = None  # Cached settings object
        self._load_config()

    def get_environment(self) -> str:
        """
        Detect the application's execution environment.

        Returns:
            str: Environment name (e.g., 'development', 'production')

        Raises:
            FileNotFoundError: If environment configuration file is missing
            KeyError: If 'ENVIRONMENT' section or 'env' key is missing
        """
        # Use APP_ENV environment variable if available
        if env_var := os.getenv("APP_ENV"):
            return env_var.strip().lower()

        # Look for mai-chat-llm-env.yaml or mai-chat-llm-env.ini in environments directory
        # YAML file takes precedence if both exist
        base_env_dir = self.base_dir / "../../config/environments"
        yaml_env_file = base_env_dir / "mai-chat-llm-env.yaml"
        ini_env_file = base_env_dir / "mai-chat-llm-env.ini"

        if yaml_env_file.exists():
            with open(yaml_env_file, 'r', encoding='utf-8') as f:
                env_config = yaml.safe_load(f)
            try:
                return env_config['ENVIRONMENT']['env'].strip().lower()
            except KeyError:
                raise KeyError("Missing 'ENVIRONMENT' section or 'env' key in mai-chat-llm-env.yaml")
        elif ini_env_file.exists():
            parser = ConfigParser()
            parser.read(ini_env_file, encoding='utf-8')
            try:
                return parser['ENVIRONMENT']['env'].strip().lower()
            except KeyError:
                raise KeyError("Missing 'ENVIRONMENT' section or 'env' key in mai-chat-llm-env.ini")
        else:
            raise FileNotFoundError(f"Could not find environment configuration file in {base_env_dir}")

    def _load_config(self):
        """
        Load both common and environment-specific configuration files.
        Supports both INI and YAML formats. If both formats exist for the same configuration base,
        the YAML file takes precedence.
        """
        base_env_dir = self.base_dir / "../../config/environments"
        # Define configuration file bases
        file_bases = [
            "mai-chat-llm-env",
            f"mai-chat-llm-{self.environment}"
        ]
        config_files = []
        for base in file_bases:
            # Prioritize YAML over INI if both exist
            yaml_path = base_env_dir / f"{base}.yaml"
            ini_path = base_env_dir / f"{base}.ini"
            if yaml_path.exists():
                config_files.append(yaml_path)
            elif ini_path.exists():
                config_files.append(ini_path)

        if not config_files:
            raise FileNotFoundError("Could not find valid configuration files.")

        # Merge configuration files into a single dictionary
        merged_config: Dict[str, Dict[str, Any]] = {}
        for config_path in config_files:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                # Assume yaml_config is a dictionary with each top-level key being a section
                for section, section_dict in yaml_config.items():
                    if not isinstance(section_dict, dict):
                        continue  # Skip sections that are not in dict format
                    if section in merged_config:
                        merged_config[section].update(section_dict)
                    else:
                        merged_config[section] = section_dict
            elif config_path.suffix.lower() == ".ini":
                parser = ConfigParser()
                parser.read(config_path, encoding='utf-8')
                for section in parser.sections():
                    section_dict = {}
                    for key in parser[section]:
                        section_dict[key] = parser[section][key]
                    if section in merged_config:
                        merged_config[section].update(section_dict)
                    else:
                        merged_config[section] = section_dict
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        self.config = merged_config

    def _parse_section(self, section: str) -> Any:
        """
        Parse a configuration section and return as an object with attributes.

        Args:
            section (str): Name of the section to parse

        Returns:
            Any: Object containing key-value pairs as attributes

        Raises:
            ValueError: If section is not found in the configuration
        """
        if section not in self.config:
            raise ValueError(f"Section '{section}' not found in configuration.")

        class Section:
            def __init__(self, items: Dict[str, Any]):
                for key, value in items.items():
                    setattr(self, key, value)

        # Convert values for each key using _parse_value
        section_data = {key: self._parse_value(section, key) for key in self.config[section]}
        return Section(section_data)

    def _parse_value(self, section: str, key: str) -> Any:
        """
        Convert configuration values to appropriate data types.
        Specifically handles comma-separated lists and applies type-specific handling based on key names.

        Args:
            section (str): Section name
            key (str): Configuration key

        Returns:
            Any: Converted configuration value
        """
        value = self.config[section][key]

        # Values read from YAML may already be of the correct type
        if isinstance(value, (list, dict, bool, int, float)):
            return value

        # Handle string values
        if isinstance(value, str):
            lower_value = value.lower()

            # Handle boolean values
            if lower_value in {"true", "yes", "on", "1"}:
                return True
            if lower_value in {"false", "no", "off", "0"}:
                return False

            # Special handling based on specific key patterns
            if _is_list_type_key(key):
                return _parse_list_value(value)

            # Handle numeric types
            try:
                if '.' in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value

        return value

    def get_section_value(self, section: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration value for the specified section and key.

        Args:
            section (str): Configuration section name
            key (str): Configuration key name
            default (Any, optional): Default value to return when value is not present

        Returns:
            Any: Configuration value or default
        """
        if section not in self.config or key not in self.config[section]:
            return default

        return self._parse_value(section, key)

    def get_settings(self) -> Any:
        """
        Load and cache all configuration settings as an object.

        Returns:
            Any: Settings object with each section as an attribute
        """
        if self.settings:
            return self.settings

        class Settings:
            pass

        self.settings = Settings()
        # Add each section as a lowercase attribute (e.g., SERVICE -> service)
        for section in self.config.keys():
            setattr(self.settings, section.lower(), self._parse_section(section))

        return self.settings
