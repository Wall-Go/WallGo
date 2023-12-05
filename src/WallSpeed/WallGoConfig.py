import configparser

class Config:
    """class Config -- Manages configuration variables for WallGo. This is essentially a wrapper around ConfigParser.
    Config variables are stored in a dict - access these using the __call__ method (ie. config = Config(); config["someVariable"]). 
    """

    config: configparser.ConfigParser

    def __init__(self, configFile: str = None):

        self.config = configparser.ConfigParser()
        self.configFile = configFile


    ## Specify defaults here. These will be used unless overriden by a user-specified config file.
    def _loadDefaultConfig(self) -> None:

        ## TODO add file paths to interpolation tables etc
        self.config['DEFAULT'] = {
            
        }
