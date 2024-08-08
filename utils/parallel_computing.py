"""Create function parallelize cpu computation."""

from typing import (
    Callable,
    Any,
)
from joblib import (
    Parallel,
    delayed,
)


def parallel_execution_of_function_with_same_params(
        num_executions: int, num_workers: int, func: Callable, *args, **kwargs,
) -> list[Any]:
    """Execute a function with fixed arguments a given number of times."""
    parallel_executer = Parallel(n_jobs=num_workers, verbose=False)
    parallel_jobs = (delayed(func)(*args, **kwargs) for _ in range(num_executions))
    return parallel_executer(parallel_jobs)


def parallel_execution_of_function_with_param_list(
        num_workers: int, func: Callable, param_list: list[Any], *args, **kwargs,
) -> list[Any]:
    """Execute a function over a list of parameters in paralell."""
    parallel_executer = Parallel(n_jobs=num_workers, verbose=False)
    parallel_jobs = (delayed(func)(params, *args, **kwargs) for params in param_list)
    return parallel_executer(parallel_jobs)
