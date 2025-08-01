"""
Logging configuration for embedding analysis.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Optional directory for log files
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('embedding_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file or log_dir:
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(exist_ok=True)
            
            if not log_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = f"analysis_{timestamp}.log"
                
            log_path = log_dir / log_file
        else:
            log_path = Path(log_file)
            
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_path}")
        
    # Set up child loggers
    for module in ['core', 'models', 'visualization', 'utils']:
        child_logger = logging.getLogger(f'embedding_analysis.{module}')
        child_logger.setLevel(logger.level)
        
    return logger