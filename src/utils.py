import logging
import os
import sys
from datetime import datetime

# ==============================BASIC SETUP ===========================================================

# Setup logging directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Log file with UTF-8 encoding
log_file = os.path.join(LOG_DIR, f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log")
file_handler = logging.FileHandler(log_file, encoding='utf-8')
formatter = logging.Formatter("[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Optional: also log to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ====================================================================================================

# Custom exception class
class CustomException(Exception):
    def __init__(self, msg, details: sys):
        _, _, tb = details.exc_info()
        self.msg = f"Error in [{tb.tb_frame.f_code.co_filename}] at line [{tb.tb_lineno}]: {msg}"
    def __str__(self): return self.msg