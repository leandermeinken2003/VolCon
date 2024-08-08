"""Define functions to help save gpu."""

import gc
import torch


def remove_gpu_copies(*args):
    gpu_copies = [*args]
    del gpu_copies
    clear_memory()


def clear_memory() -> None:
    torch.cuda.empty_cache()
    gc.collect()
