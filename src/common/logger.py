import atexit
import logging
import os
import sys
import time as pytime
from datetime import datetime
from threading import Timer
from typing import Union

from colorama import Fore, Style, init  # Used for colored output in the console
from loguru import logger

from src.common.config_loader import ConfigLoader

# Initialize colorama for cross-platform compatibility with colored output
init(autoreset=True)


# ================================
#         Configuration
# ================================

class Settings:
    """
    Load configuration values from a settings file.
    """

    class Log:
        # Load settings using the ConfigLoader
        config_loader = ConfigLoader()
        config = config_loader.get_settings()

        # Logging configuration values
        log_path = config.log.log_path
        log_filename = config.log.log_filename
        uvicorn_log_level = config.log.uvicorn_log_level
        uvicorn_log_filename = config.log.uvicorn_log_filename
        uvicorn_access_filename = config.log.uvicorn_access_log_filename
        uvicorn_error_filename = config.log.uvicorn_error_log_filename
        log_level = config.log.log_level
        log_backtrace = config.log.log_backtrace
        log_diagnose = config.log.log_diagnose
        log_retention_days = config.log.log_retention_days

    log = Log()


# Instantiate the settings and timers list for scheduled tasks
settings = Settings()
timers = []

# Ensure that the logging directory exists
os.makedirs(settings.log.log_path, exist_ok=True)


# ================================
#       Logging Classes
# ================================

class InterceptHandler(logging.Handler):
    """
    Custom logging handler that intercepts logs from the standard logging module
    and forwards them to Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Try to get the corresponding Loguru log level name
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            # Fallback to numeric log level if level name is not found
            level = record.levelno

        # Determine the depth of the log call stack to record correct context
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Forward the log record to Loguru
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# ================================
#       Logger Configuration
# ================================

def setup_logger_output(sink):
    """
    Set up Loguru to output logs in a consistent format with colors
    to the specified sink (stdout or a file).

    Args:
        sink: The destination for the log output (e.g., sys.stdout or a file path).
    """
    logger.add(
        sink,
        format="<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> [<level>{level}</level>] {message}",
        level=settings.log.log_level,
        backtrace=settings.log.log_backtrace,
        diagnose=settings.log.log_diagnose,
        colorize=True,
    )


def configure_uvicorn_access_log():
    """
    Configure Uvicorn access logs to output to both a dedicated file and the console,
    using colored formatting for improved readability.
    """
    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.setLevel(settings.log.uvicorn_log_level)
    uvicorn_access_logger.handlers = []  # Remove existing handlers

    # Create file handler for Uvicorn access logs
    access_log_file = os.path.join(settings.log.log_path, settings.log.uvicorn_access_filename)
    file_handler = logging.FileHandler(str(access_log_file))

    # Create console handler for Uvicorn access logs
    console_handler = logging.StreamHandler(sys.stdout)

    # Define formatter for file logs
    file_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Define colorized formatter for console logs
    colorized_format = (
        f"{Fore.GREEN}[%(asctime)s]{Style.RESET_ALL} "
        f"{Fore.YELLOW}[%(levelname)s]{Style.RESET_ALL} %(message)s"
    )
    colorized_formatter = logging.Formatter(colorized_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Set the formatters for each handler
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(colorized_formatter)

    # Add handlers to the Uvicorn access logger
    uvicorn_access_logger.addHandler(file_handler)
    uvicorn_access_logger.addHandler(console_handler)


def configure_uvicorn_error_log():
    """
    Configure Uvicorn error logs to output to both a dedicated file and the console,
    using colored formatting for enhanced readability.
    """
    uvicorn_error_logger = logging.getLogger("uvicorn.error")
    uvicorn_error_logger.setLevel(logging.WARNING)
    uvicorn_error_logger.handlers = []  # Remove existing handlers

    # Create file handler for Uvicorn error logs
    error_log_file = os.path.join(settings.log.log_path, settings.log.uvicorn_error_filename)
    file_handler = logging.FileHandler(str(error_log_file))

    # Create console handler for Uvicorn error logs
    console_handler = logging.StreamHandler(sys.stdout)

    # Define formatter for file logs
    file_format = "[%(asctime)s] [%(levelname)s] %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Define colorized formatter for console logs
    colorized_format = (
        f"{Fore.CYAN}[%(asctime)s]{Style.RESET_ALL} "
        f"{Fore.MAGENTA}[%(levelname)s]{Style.RESET_ALL} %(message)s"
    )
    colorized_formatter = logging.Formatter(colorized_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Set the formatters for each handler
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(colorized_formatter)

    # Add handlers to the Uvicorn error logger
    uvicorn_error_logger.addHandler(file_handler)
    uvicorn_error_logger.addHandler(console_handler)


def copy_and_truncate(src_file_path: str, prefix: str):
    """
    Copy the current log file's contents to a new file with the current date in its name,
    then truncate (clear) the original log file.

    Args:
        src_file_path: The source file path of the current log file.
        prefix: A prefix used to generate the new log file's name.
    """
    if not os.path.exists(src_file_path):
        return

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    dst_file_path = os.path.join(settings.log.log_path, f"{prefix}_{date_str}.log")

    try:
        # Read the content of the source log file and write it to the destination file
        with open(src_file_path, 'rb') as src, open(dst_file_path, 'wb') as dst:
            dst.write(src.read())

        # Truncate the source file (clear its contents)
        with open(src_file_path, 'w', encoding='utf-8'):
            pass
    except Exception as e:
        # Handle errors during log processing to prevent impact on the main program
        logger.warning(f"Error during log rotation: {e}")


def schedule_midnight_rotation():
    """
    Schedule the log file rotation to occur at the next midnight.
    """
    current_time = pytime.localtime()
    next_midnight = pytime.mktime((
        current_time.tm_year,
        current_time.tm_mon,
        current_time.tm_mday + 1,
        0, 0, 0, 0, 0, 0
    ))
    delay = next_midnight - pytime.mktime(current_time)

    def rotate_logs():
        try:
            # Rotate main controller log file
            controller_log_path = os.path.join(settings.log.log_path, settings.log.log_filename)
            copy_and_truncate(str(controller_log_path), "MAI_Controller")

            # Rotate Uvicorn log file
            uvicorn_log_path = os.path.join(settings.log.log_path, settings.log.uvicorn_log_filename)
            copy_and_truncate(str(uvicorn_log_path), "uvicorn")

            # Rotate Uvicorn access log file
            uvicorn_access_log_path = os.path.join(settings.log.log_path, settings.log.uvicorn_access_filename)
            copy_and_truncate(str(uvicorn_access_log_path), "uvicorn_access")

            # Rotate Uvicorn error log file
            uvicorn_error_log_path = os.path.join(settings.log.log_path, settings.log.uvicorn_error_filename)
            copy_and_truncate(str(uvicorn_error_log_path), "uvicorn_error")

            # Reschedule the rotation for the next midnight
            schedule_midnight_rotation()
        except Exception as e:
            logger.warning(f"Error during log rotation: {e}")
            # Schedule next rotation even if an error occurs
            schedule_midnight_rotation()

    # Set up a timer to trigger log rotation at midnight
    timer = Timer(delay, rotate_logs)
    timer.daemon = True  # Daemonize the timer so it won't block program exit
    timers.append(timer)
    timer.start()


def configure_logging():
    """
    Initialize logging for the application.
    This configures both Loguru and the standard logging module (including Uvicorn logs)
    to output in a consistent, colorized format.
    """
    # Configure standard logging to pass logs to Loguru via InterceptHandler
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET)

    # Remove all existing Loguru handlers
    logger.remove()

    # Set up Loguru outputs for stdout and log files
    setup_logger_output(sys.stdout)
    setup_logger_output(os.path.join(settings.log.log_path, settings.log.log_filename))
    setup_logger_output(os.path.join(settings.log.log_path, settings.log.uvicorn_log_filename))

    # Configure Uvicorn access and error logs (both file and console output)
    configure_uvicorn_access_log()
    configure_uvicorn_error_log()

    # Schedule log rotation at midnight
    schedule_midnight_rotation()


@atexit.register
def cleanup_timers():
    """
    Cancel all scheduled timers upon program exit to ensure a clean shutdown.
    """
    for timer in timers:
        timer.cancel()


# ================================
#       Testing the Logger
# ================================

if __name__ == "__main__":
    # Initialize the logging configuration
    configure_logging()

    # Create a test logger bound with a custom name
    test_logger = logger.bind(name="TEST")

    # Log messages with various severity levels for testing
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")
    test_logger.critical("Critical error")
