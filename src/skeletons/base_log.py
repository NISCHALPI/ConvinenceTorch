import logging
import typing as tp
import sys
import os

__all__ = ['configure_logging', 'get_logger']

def configure_logging():
    """
    Configures the base logging settings.
    
    This function sets up the basic configuration for the logging module. It configures the log format, log level,
    and a stream handler to print log messages to the standard output (stdout).
    """
    # Default level
    level = 40 # ERROR LEVEL
    
    # Get level from environment if present
    if os.getenv('LOG') is not None:
        level = int(os.getenv('LOG'))
    
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s',
                        level=level,
                        handlers=[logging.StreamHandler(sys.stdout)])

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the specified name.
    
    Parameters:
        name (str): The name of the logger.
        
    Returns:
        logging.Logger: A logger instance with the specified name.
    """
    return logging.getLogger(name=name)
