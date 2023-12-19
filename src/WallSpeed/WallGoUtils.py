
import importlib.resources

## Put common non-physics related functions here. Common physics/math functions should go into helpers.py


def getSafePathToResource(relativePathToResource: str) -> str:
    """ Gives a safe path to a packaged resource. The input is a relative path
    from WallGo package directory (ie. where __init__.py is located).
    Use this function to convert the relative path to a path that is safe to use in packaged context.
    Example relative path: /Data/Something/example.txt.
    
    Returns
    -------
    Path to the resource file: str.
    """

    ## fallback to "WallSpeed" if the package call fails for some reason
    packageName = __package__ or "WallSpeed"

    return importlib.resources.files(packageName) / relativePathToResource
