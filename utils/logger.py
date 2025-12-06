import logging
from pathlib import Path


def setup_logger(name: str, file_name: str = None, level=logging.DEBUG) -> logging.Logger:
    """Sets up a logger with the specified name and file output.

    Args:
        name (str): The name of the logger.
        file_name (str, optional): The file to log messages to. If None, logs to console. Defaults to None.
        level (int, optional): The logging level. Defaults to logging.DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    datefmt = '%Y-%m-%d %H:%M:%S'
    log_format = '[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s'

    formatter = logging.Formatter(fmt=log_format, datefmt=datefmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if file_name is not None:
        file_path = Path(file_name)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger
