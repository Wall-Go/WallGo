import configparser
import io

class Config:
    """class Config -- Manages configuration variables for WallGo. This is essentially a wrapper around ConfigParser.
    Accessing variables works as with ConfigParser: 
    config.get("Section", "someKey")
    """

    configParser: configparser.ConfigParser

    def __init__(self):

        self.config = configparser.ConfigParser()


    def readINI(self, filePath: str):
        self.config.read(filePath)


    def get(self, section: str, key: str) -> any:
        return self.config.get(section, key)
    
    def __str__(self) -> str:
        """Print the content in similar format as an .ini file would be.
        """

        ## Do magic by writing the contents into a file-like string buffer
        buffer = io.StringIO()
        self.config.write(buffer)
        
        return buffer.getvalue()
