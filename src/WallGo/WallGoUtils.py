import configparser
import importlib.resources

## Put common non-physics related functions here. Common physics/math functions should go into helpers.py

def loadConfig(filePath: str) -> dict:

    parser = configparser.ConfigParser()

    parser.read(filePath) # silently fails if file not found

    # Read the .ini contents in dict
    configDict = {}
    
    for section in parser.sections():
        configDict[section] = {}
        
        for option in parser.options(section):
            configDict[section][option] = parser.get(section, option)


    return configDict


def getPackagedDataPath(relativeModulePath: str, fileName: str) -> str:
    """ Common routine for accessing packaged data files within WallGo, using modern importlib practices.
        Usage: if the file is WallGo/Data/Something/example.txt, 
        call this as getPackagedDataPath("WallGo.Data.Something", "example.txt").

        Returns
        -------
        Path to the resource file: str.
    """
    return str( importlib.resources.files(relativeModulePath).joinpath(fileName) )


def getSafePathToResource(relativePathToResource: str) -> str:
    """ Gives a safe path to a packaged resource. The input is a relative path
    from WallGo package directory (ie. where __init__.py is located).
    Use this function to convert the relative path to a path that is safe to use in packaged context.
    Example relative path: /Data/Something/example.txt.
    
    Returns
    -------
    Path to the resource file: str.
    """

    ## fallback to "WallGo" if the package call fails for some reason
    packageName = __package__ or "WallGo"

    return importlib.resources.files(packageName) / relativePathToResource
