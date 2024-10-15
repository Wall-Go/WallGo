"""
Specific Exception types used in WallGo
"""

import typing
from .containers import PhaseInfo


class WallGoError(Exception):
    """
    Generic Exception type for WallGo errors.
    """

    def __init__(
        self, message: str, data: typing.Optional[dict[str, typing.Any]] = None
    ) -> None:
        # Use the data dict for reporting arbitrary data with the error message
        self.message = message
        self.data = data

    def __str__(self) -> str:
        """
        Conversion of WallGoError to str, includes any data.
        """
        msg = str(self.message)
        if self.data:
            msg += "\nAdditional info:\n" + str(self.data)

        return msg


class WallGoPhaseValidationError(WallGoError):
    """
    Exception for failures related to phase input.
    """

    def __init__(
        self,
        message: str,
        phaseInfo: "PhaseInfo",
        data: typing.Optional[dict[str, typing.Any]] = None,
    ) -> None:
        # Additional phaseInfo
        super().__init__(message, data)
        self.phaseInfo = phaseInfo

    def __str__(self) -> str:
        msg = str(self.message) + "\nPhase was: \n" + str(self.phaseInfo)
        if self.data:
            msg += "\nAdditional info:\n" + str(self.data)

        return msg


class CollisionLoadError(Exception):
    """Raised when collision integrals fail to load"""
    pass



