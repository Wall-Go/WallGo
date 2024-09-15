"""
Class that manages configuration variables for WallGo.
"""

import configparser
import io
from typing import Any


class Config:
    """
    This class is essentially a wrapper around ConfigParser. Accessing variables
    works as with ConfigParser: config.get("Section", "someKey")
    """

    configParser: configparser.ConfigParser

    def __init__(self) -> None:

        self.config = configparser.ConfigParser()
        self.config.optionxform = str  # preserve case

    def readINI(self, filePath: str) -> None:
        """
        Reads a configuration file.

        Parameters
        ----------
        filePath : str
            Path to the configuration file.

        Returns
        -------
        """
        self.config.read(filePath)

    def get(self, section: str, key: str) -> Any:
        """
        Returns a variable from the configuration.

        Parameters
        ----------
        section : str
            The section from which to obtain the variable.
        key : str
            The key of the variable.

        Returns
        -------
        variable: Any
            The configuration variable.
        """
        return self.config.get(section, key)

    def getint(self, section: str, key: str) -> int:
        """
        Returns an integer variable from the configuration.

        Parameters
        ----------
        section : str
            The section from which to obtain the variable.
        key : str
            The key of the variable.

        Returns
        -------
        variable: int
            The configuration variable.
        """
        return self.config.getint(section, key)

    def getfloat(self, section: str, key: str) -> float:
        """
        Returns a float variable from the configuration.

        Parameters
        ----------
        section : str
            The section from which to obtain the variable.
        key : str
            The key of the variable.

        Returns
        -------
        variable: float
            The configuration variable.
        """
        return self.config.getfloat(section, key)

    def set(self, section: str, key: str, value: str) -> None:
        """
        Modifies a variable in the configuration

        Parameters
        ----------
        section : str
            The section under which the variable is listed.
        key : str
            The key of the variable.
        value : str
            The new value of the variable.

        Returns
        -------
        """
        self.config.set(section, key, value)

    def getboolean(self, section: str, key: str) -> bool:
        """
        Returns a boolean variable from the configuration.

        Parameters
        ----------
        section : str
            The section from which to obtain the variable.
        key : str
            The key of the variable.

        Returns
        -------
        variable: bool
            The configuration variable.
        """
        return self.config.getboolean(section, key)

    def __str__(self) -> str:
        """
        Prints the content in similar format as an .ini file would be.
        """

        ## Do magic by writing the contents into a file-like string buffer
        buffer = io.StringIO()
        self.config.write(buffer)

        return buffer.getvalue()
