import logging
import logging.handlers
import os

def setup_logger(name, level=logging.INFO, log_file='crawler.log', max_bytes=10*1024*1024, backup_count=5):
    """
    Set up and return a logger with the specified name.

    Parameters:
        name (str): Name of the logger.
        level (int): Logging level.
        log_file (str): The filename of the log file.
        max_bytes (int): Maximum file size in bytes before rotation occurs.
        backup_count (int): Number of backup files to keep.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)

    if not logger.handlers:
        # Ensure the directory for log files exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Create console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)

        # Create rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        # Add handlers to logger
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

    return logger
