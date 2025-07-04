"""
This type stub file was generated by pyright.
"""

"""
Helpers for logging.

This module needs much love to become useful.
"""
def format_time(t): # -> str:
    ...

def short_format_time(t): # -> str:
    ...

def pformat(obj, indent=..., depth=...): # -> str:
    ...

class Logger:
    """Base class for logging messages."""
    def __init__(self, depth=..., name=...) -> None:
        """
        Parameters
        ----------
        depth: int, optional
            The depth of objects printed.
        name: str, optional
            The namespace to log to. If None, defaults to joblib.
        """
        ...
    
    def warn(self, msg): # -> None:
        ...
    
    def info(self, msg): # -> None:
        ...
    
    def debug(self, msg): # -> None:
        ...
    
    def format(self, obj, indent=...): # -> str:
        """Return the formatted representation of the object."""
        ...
    


class PrintTime:
    """Print and log messages while keeping track of time."""
    def __init__(self, logfile=..., logdir=...) -> None:
        ...
    
    def __call__(self, msg=..., total=...): # -> None:
        """Print the time elapsed between the last call and the current
        call, with an optional message.
        """
        ...
    


