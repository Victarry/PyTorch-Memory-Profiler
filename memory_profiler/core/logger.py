import logging
import os
from typing import Optional, List

from rich.console import Console
from rich.logging import RichHandler

# Configure default logging levels
DEFAULT_LOG_LEVEL = logging.INFO
CONSOLE = Console(highlight=True, width=200)

def configure_logging(
    level: Optional[int] = None,
    log_to_file: bool = False,
    log_file: Optional[str] = None,
    rich_format: bool = True
) -> None:
    """Configure the logging system for memory profiler.
    
    Args:
        level: Logging level (use logging.DEBUG, logging.INFO, etc.)
        log_to_file: Whether to log to a file
        log_file: Path to log file if log_to_file is True
        rich_format: Whether to use rich formatting for console output
    """
    # Determine log level from env var or parameter
    log_level = level or int(os.environ.get("MEMORY_PROFILER_LOG_LEVEL", DEFAULT_LOG_LEVEL))
    
    # Create handlers
    handlers = []
    
    # Console handler (with or without rich formatting)
    if rich_format:
        console_handler = RichHandler(console=CONSOLE, rich_tracebacks=True)
        log_format = "%(message)s"
    else:
        console_handler = logging.StreamHandler()
        log_format = "[%(levelname)s | %(name)s | %(asctime)s] %(message)s"
        
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if requested
    if log_to_file:
        file_path = log_file or "memory_profiler.log"
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(logging.Formatter(
            "[%(levelname)s | %(name)s | %(asctime)s] %(message)s"
        ))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        force=True
    )

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger with the given name.
    
    Args:
        name: Name for the logger, typically __name__
        
    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)

def log_memory_table(logger: logging.Logger, 
                    title: str, 
                    headers: List[str], 
                    rows: List[List[str]], 
                    level: int = logging.INFO) -> None:
    """Log a formatted table of memory information.
    
    Args:
        logger: Logger instance
        title: Table title
        headers: List of column headers
        rows: List of rows, each a list of values
        level: Logging level for this message
    """
    if logger.isEnabledFor(level):
        # Calculate maximum width for each column
        col_widths = [len(h) for h in headers]
        
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        
        # Function to format a row
        def format_row(items, widths):
            return " | ".join(f"{str(item):<{widths[i]}}" for i, item in enumerate(items))

        # Create the table header
        header_row = format_row(headers, col_widths)
        separator = "-" * len(header_row)
        
        # Log the table
        logger.log(level, f"\n{title}")
        logger.log(level, separator)
        logger.log(level, header_row)
        logger.log(level, separator)
        
        for row in rows:
            formatted_row = format_row(row, col_widths)
            logger.log(level, formatted_row)
        
        logger.log(level, separator)

# Initialize default logging configuration
configure_logging() 