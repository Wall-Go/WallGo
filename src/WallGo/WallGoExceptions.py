from .WallGoTypes import PhaseInfo

class WallGoError(Exception):

    def __init__(self, message: str, data: dict[str, any] = None):
        ## Use the data dict for reporting arbitrary data with the error message
        self.message = message
        self.data = data

    def __str__(self):
        msg = str(self.message)
        if self.data:
            msg += "\nAdditional info:\n" + str(self.data)

        return msg

class WallGoPhaseValidationError(WallGoError):
    """Exception raised when WallGo fails to operate with the user specified phase input.
    """

    def __init__(self, message: str, phaseInfo: 'PhaseInfo', data: dict[str, any] = None):
        ## Use the data dict for reporting arbitrary data with the error message
        self.message = message
        self.phaseInfo = phaseInfo
        self.data = data

    def __str__(self):
        msg = str(self.message) + "\nPhase was: \n" + str(self.phaseInfo)
        if self.data:
            msg += "\nAdditional info:\n" + str(self.data)

        return msg