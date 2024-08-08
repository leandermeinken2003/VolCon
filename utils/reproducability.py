"""Define function to ensure reproducability of training."""

import os
import random

import numpy as np
import torch


DEFAULT_RANDOM_SEED = 1111


def ensure_reproducability(seed=DEFAULT_RANDOM_SEED):
    """Set all relevant seeds and ensure deterministic execution of code."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
