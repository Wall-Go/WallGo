import configparser
from pathlib import Path

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


def getProjectRoot() -> Path:
    """ Returns Path object that points to WallGo root directory.
    """
    # This assumes dir structure <ProjectRoot>/src/<PackageName>/<ThisFile.py>, so need parent calls
    return Path(__file__).absolute().parent.parent.parent
