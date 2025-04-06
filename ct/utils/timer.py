import time
def timer_decorator( func ):
    """
    A decorator that prints the execution time of the function it decorates.
    """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} Execution time: {end_time - start_time:.4f} seconds")
        return result
    return wrapper