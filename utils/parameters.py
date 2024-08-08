"""Define utility functions for dealing with parameters."""

import math

import torch
from torch import nn

def create_parameters(shape: tuple[int]) -> torch.Tensor:
    """Create an empty parameter tensor of a given shape."""
    empty_parameters = torch.empty(shape)
    return nn.Parameter(empty_parameters)

def initialize_parameters(parameter: torch.Tensor) -> None:
    """Initialize the weights of a given parameter tensor."""
    torch.nn.init.kaiming_normal_(parameter, a=math.sqrt(5))
