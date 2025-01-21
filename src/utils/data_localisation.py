import os

def is_kaggle() -> bool:
    """
    Check if the code is running in a Kaggle environment.

    Returns:
    -------
    bool
        True if in Kaggle environment, False otherwise.
    """
    return os.path.exists('/kaggle/')

def is_notebook() -> bool:
    """
    Check if the code is running in a Jupyter notebook environment.

    Returns:
    -------
    bool
        True if in a Jupyter notebook, False otherwise.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  
        elif shell == 'TerminalInteractiveShell':
            return False  
        else:
            return False  
    except NameError:
        return False  
    
def find_capstone_dir(base_dir: str) -> str:
    """
    Traverse up the directory tree to find the 'capstone_group5' folder.

    Parameters:
    ----------
    base_dir : str
        Starting directory for the search.

    Returns:
    -------
    str
        The path to the 'capstone_group5' directory.

    Raises:
    -------
    FileNotFoundError
        If the 'capstone_group5' directory is not found.
    """
    while not base_dir.endswith('capstone_group5'):
        parent_dir = os.path.dirname(base_dir)
        if parent_dir == base_dir:  
            raise FileNotFoundError("Could not find 'capstone_group5' in the directory tree.")
        base_dir = parent_dir
    return base_dir