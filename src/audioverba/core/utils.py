import torch
import logging
from typing import Literal

# Set up logging
logger = logging.getLogger(__name__)

def get_device() -> torch.device:
    """
    Detects and returns the appropriate torch device ('cuda' if available, else 'cpu').

    Returns:
        A torch.device object representing the best available device.
    """
    if torch.cuda.is_available():
        device_str = 'cuda'
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device_str = 'cpu'
        logger.info("CUDA not available. Using CPU.")
    return torch.device(device_str)
