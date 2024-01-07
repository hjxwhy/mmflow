# Copyright (c) OpenMMLab. All rights reserved.
import logging
from typing import Optional

from mmengine.logging import MMLogger


def get_root_logger(log_file: Optional[str] = None,
                    log_level: int = logging.INFO) -> logging.Logger:
    """Get the logger when training or testing task.

    Args:
        log_file (str, optional): The name of the log file. Defaults to None.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
    Returns:
        logging.Logger: The expected logger.
    """
    return MMLogger('mmflow', log_file=log_file, log_level=log_level)
