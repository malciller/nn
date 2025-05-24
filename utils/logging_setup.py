import logging
from config.config import LOG_FILE_PATH

def setup_logging():
    """Setup centralized logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        filename=LOG_FILE_PATH,
        filemode='w'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)
    
    return logging.getLogger(__name__)

def get_logger(name):
    """Get a logger instance."""
    return logging.getLogger(name) 