"""Utility functions for the BANKING77 classification system."""
import time
import functools
import os


def timer(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"  [{func.__name__}] completed in {elapsed:.2f}s")
        return result
    return wrapper


def get_dir_size_mb(path: str) -> float:
    """Get total size of a directory in MB."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def print_separator(title: str = "", width: int = 60):
    """Print a formatted separator."""
    if title:
        print(f"\n{'='*width}")
        print(f"  {title}")
        print(f"{'='*width}")
    else:
        print(f"{'='*width}")


def format_percentage(value: float) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.2f}%"
