"""This script provides utility functions for configuring and retrieving loggers.

configure_logging:
------------------
    Configures the base logging settings.
    This function sets up the basic configuration for the logging module. It configures the log format, log level,
    and a stream handler to print log messages to the standard output (stdout).

get_logger:
-----------
    Retrieves a logger instance with the specified name.
    This function returns a logger instance with the specified name. It allows customizing the log level by reading
    the level from the environment variable 'LOG'. If the environment variable is not set, it defaults to the ERROR level.
    The function takes the following parameter:
    - name: The name of the logger.
    

Returns:
    logging.Logger: A logger instance with the specified name.

Note: For detailed documentation of the functions and their parameters, please refer to the docstrings within the code.
"""


import logging
import os
import sys

__all__ = ["configure_logging", "get_logger"]


def configure_logging() -> None:
    """Configures the base logging settings.

    This function sets up the basic configuration for the logging module. It configures the log format, log level,
    and a stream handler to print log messages to the standard output (stdout).
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
        level=logging.WARNING,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str) -> logging.Logger:
    """Retrieves a logger instance with the specified name.

    Parameters:
        name (str): The name of the logger.

    Returns:
        logging.Logger: A logger instance with the specified name.
    """
    # Default level
    level = 40  # ERROR LEVEL

    # Get level from environment if present
    if os.getenv("LOG") is not None:
        level = int(os.getenv("LOG"))  # type: ignore

    logger = logging.getLogger(name=name)

    logger.setLevel(level=level)  # type: ignore

    return logger
