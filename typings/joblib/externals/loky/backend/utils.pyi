"""
This type stub file was generated by pyright.
"""

def kill_process_tree(process, use_psutil=...): # -> None:
    """Terminate process and its descendants with SIGKILL"""
    ...

def recursive_terminate(process, use_psutil=...): # -> None:
    ...

def get_exitcodes_terminated_worker(processes): # -> str:
    """Return a formatted string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    ...

