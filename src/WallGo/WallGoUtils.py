"""
Collection of non-physics related functions.
Common physics/math functions should go into helpers.py.
"""
import importlib.resources
import numpy as np

def getSafePathToResource(relativePathToResource: str) -> str:
    """ 
    Gives a safe path to a packaged resource. The input is a relative path
    from the WallGo package directory (ie. where __init__.py is located).
    Use this function to convert the relative path to a path that is safe
    to use in packaged context.
    Example relative path: /Data/Something/example.txt.
    
    Parameters
    ----------
    relativePathToResource : str
        relative path.

    Returns
    -------
    Path to the resource file : str.
    """

    # fallback to "WallGo" if the package call fails for some reason
    packageName = __package__ or "WallGo"

    return importlib.resources.files(packageName) / relativePathToResource


def clamp(x: float, minx: float, maxx: float) -> float:
    """
    Clamp x in range [minx, maxx], inclusive.
    
    Parameters
    ----------
    x : float
        parameter to be clamped.
    minx : float
        minimum return value.
    maxx : float
        maximum return value.

    Returns
    -------
    clamped x : float
        returns minx when x < minx, returns maxx when x > maxx,
        otherwise returns x.


    """
    return np.clip(x, minx, maxx)
