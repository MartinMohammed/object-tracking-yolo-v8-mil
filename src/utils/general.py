from typing import Callable, Any
import time


def measure_time(name: str, cb: Callable[..., Any], *args) -> Any:
    """
    Measures the execution time of a callback function.

    Parameters:
        name (str): A name for the operation being measured.
        cb (Callable[..., Any]): The callback function to be executed.
        *args: Variable length argument list to pass to the callback function.

    Returns:
        Any: The output of the callback function.
    """
    start_time = time.time()
    output = cb(*args)
    end_time = time.time()
    print(f"Time to execute '{name}' took {round(end_time - start_time, 3)} seconds")
    return output
