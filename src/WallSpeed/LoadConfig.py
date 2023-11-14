import configparser


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

