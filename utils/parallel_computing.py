"""Create function parallelize cpu computation."""

from typing import (
    Callable,
    Any,
)
from joblib import (
    Parallel,
    delayed,
)


def parallel_execution_of_function(
        num_executions: int, func: Callable, *args, **kwargs,
) -> list[Any]:
    """Execute a function with fixed arguments a given number of times."""
    parallel_executer = Parallel(n_jobs=-1, verbose=False)
    parallel_jobs = (delayed(func)(*args, **kwargs) for _ in range(num_executions))
    return parallel_executer(parallel_jobs)
