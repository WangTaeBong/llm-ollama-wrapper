import json
import logging


class JsonConfigLoader:
    """
    A class to load and manage configuration settings from a JSON file.

    Attributes:
        config_path (str): The path to the JSON configuration file.
        config_data (dict): The loaded configuration data stored in memory.
        _logger (logging.Logger): Logger instance for logging errors and debug messages.
    """

    def __init__(self, config_path):
        """
        Initializes the JsonConfigLoader with a specified configuration file path.

        Args:
            config_path (str): Path to the JSON configuration file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file contains invalid JSON or non-dict data.
        """
        self.config_path = config_path  # Store the path to the configuration file
        self.config_data = {}  # Initialize an empty dictionary for configuration data
        self._logger = logging.getLogger(__name__)  # Create a logger for the class
        # Load configuration at initialization
        self.load_config()

    def load_config(self):
        """
        Loads the JSON configuration file into memory.

        This method reads the JSON file specified in `self.config_path`,
        validates the data structure, and stores it in `self.config_data`.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            ValueError: If the configuration file contains invalid JSON or non-dict data.
            Exception: For any unexpected errors during file loading.
        """
        try:
            # Attempt to open and read the configuration file
            with open(self.config_path, 'r', encoding='utf-8') as config_file:
                self.config_data = json.load(config_file)  # Parse JSON data into a dictionary

            # Validate that the loaded data is a dictionary
            if not isinstance(self.config_data, dict):
                raise ValueError("Configuration data must be a JSON object.")

        except FileNotFoundError:
            # Log and raise an error if the file is not found
            self._logger.error(f"Configuration file not found: {self.config_path}")
            raise

        except json.JSONDecodeError as e:
            # Log and raise an error if the JSON is invalid
            self._logger.error(f"Error decoding JSON configuration: {e}")
            raise ValueError(f"Error decoding JSON configuration: {e}")

        except Exception as e:
            # Log and raise unexpected errors
            self._logger.error(f"Unexpected error while loading configuration: {e}")
            raise

    def get_settings(self):
        """
        Retrieves the currently loaded configuration settings.

        Returns:
            dict: The loaded configuration data.

        Example:
            >>> loader = JsonConfigLoader('config.json')
            >>> settings = loader.get_settings()
            >>> print(settings)
        """
        return self.config_data

    def reload(self):
        """
        Reloads the configuration file to refresh the settings.

        This method is useful when the configuration file has been updated
        and the changes need to be reflected in the application.

        Example:
            >>> loader = JsonConfigLoader('config.json')
            >>> loader.reload()
        """
        self.load_config()
