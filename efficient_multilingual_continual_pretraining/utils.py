import random
import time
from contextlib import ContextDecorator
from functools import wraps
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from efficient_multilingual_continual_pretraining import logger


class log_with_message(ContextDecorator):
    def __init__(
        self,
        message: str,
        log_time: bool = True,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ):
        self.message = message
        self.log_time = log_time
        self.log_level = log_level
        self.start_time = None

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            logger.exception(f"An exception occurred: {exc_type}")
        else:
            elapsed_time = time.time() - self.start_time
            if self.log_time:
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)
                time_str = (
                    f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
                    if hours
                    else f"{int(minutes)}m {seconds:.2f}s" if minutes else f"{seconds:.2f}s"
                )
                logger.log(self.log_level, f"Finished {self.message}. Time taken: {time_str}.")
            else:
                logger.log(self.log_level, f"Finished {self.message}.")

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            # Correct string concatenation
            complete_message = f"Started {self.message}"
            if "set_name" in kwargs:
                complete_message += f", set name {kwargs['set_name']}."
            else:
                complete_message += "."

            logger.log(self.log_level, complete_message)
            self.start_time = time.time()
            with self:
                return func(*args, **kwargs)

        return wrapped


def verbose_iterator(
    iterator,
    verbose,
    desc: str | None = None,
    leave: bool = False,
    **kwargs,
):
    if verbose:
        return tqdm(
            iterator,
            leave=leave,
            desc=desc,
            **kwargs,
        )
    return iterator


def seed_everything(
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def generate_device(use_cuda: bool = True) -> torch.device:
    """Provides a devices based on either you want to use `cuda` or not.

    Parameters
    ----------
    use_cuda : bool
        If using a `cuda` device if possible is required.

    Returns
    -------
    device : torch.device
        The available device for further usage.

    """
    if use_cuda:
        if not torch.cuda.is_available():
            message = "CUDA is not available while being asked for it. Falling back to CPU."
            logger.warning(message)
            return torch.device("cpu")

        return torch.device("cuda:0")

    return torch.device("cpu")
