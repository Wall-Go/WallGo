"""
Data classes for compiling and returning results
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from .fields import Fields
from .containers import BoltzmannBackground, BoltzmannDeltas, WallParams


@dataclass
class BoltzmannResults:
    """
    Holds results to be returned by BoltzmannSolver
    """

    deltaF: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(z, p_z, p_\parallel)`."""

    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`."""

    truncationError: float
    r"""Estimated relative error in :math:`\delta f` due to truncation
    of spectral expansion."""

    # These two criteria are to evaluate the validity of the linearization of the
    # Boltzmann equation. The arrays contain one element for each out-of-equilibrium
    # particle. To be valid, at least one criterion must be small for each particle.
    linearizationCriterion1: np.ndarray
    r"""Ratio of out-of-equilibrium and equilibrium pressures,
    :math:`|P[\delta f]| / |P[f_\text{eq}]|`. One element for each
    out-of-equilibrium particle."""

    linearizationCriterion2: np.ndarray
    r"""Ratio of collision and Liouville operators in Boltzmann equation,
    :math:`|\mathcal{C}[\delta f]|/ |\mathcal{L}[\delta f]|`. One element for each
    out-of-equilibrium particle."""

    def __mul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __rmul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __add__(self, other: "BoltzmannResults") -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=other.deltaF + self.deltaF,
            Deltas=other.Deltas + self.Deltas,
            truncationError=other.truncationError + self.truncationError,
            linearizationCriterion1=other.linearizationCriterion1
            + self.linearizationCriterion1,
            linearizationCriterion2=other.linearizationCriterion2
            + self.linearizationCriterion2,
        )

    def __sub__(self, other: "BoltzmannResults") -> "BoltzmannResults":
        return self.__add__((-1) * other)


@dataclass
class HydroResults:
    """
    Holds results to be returned by Hydro
    """

    temperaturePlus: float
    r"""Temperature in front of the bubble, :math:`T_+`,
    from hydrodynamic matching conditions."""

    temperatureMinus: float
    r"""Temperature behind the bubble, :math:`T_-`,
    from hydrodynamic matching conditions."""

    velocityJouguet: float
    r"""Jouguet velocity, :math:`v_J`, the smallest velocity for a detonation."""

    def __init__(
        self,
        temperaturePlus: float,
        temperatureMinus: float,
        velocityJouguet: float,
    ):
        self.temperaturePlus = temperaturePlus
        self.temperatureMinus = temperatureMinus
        self.velocityJouguet = velocityJouguet


class ESolutionType(Enum):
    """
    Enum class used to label the different types of solution WallGo can find
    """

    DEFLAGRATION = 1
    """
    Indicates that a solution was found while looking for a deflagration or hybrid. Can
    also be used when the pressure was always positive while looking for a detonation.
    """

    DETONATION = 2
    """ Indicates that a solution was found while looking for a detonation. """

    RUNAWAY = 3
    """
    Indicates that no solution was found and that the pressure was always negative.
    """

    DEFLAGRATION_OR_RUNAWAY = 4
    """
    Used when no stable solution was found while looking for a detonation with a
    positive pressure at vw=vJ and negative at vw=1.
    """

    ERROR = 5
    """ Indicates that the calculation was not successful. """


@dataclass
class WallGoResults:
    """
    Compiles output results for users of WallGo
    """

    wallVelocity: float | None
    """Bubble wall velocity. None if no solution was found."""

    wallVelocityError: float | None
    """Estimated error in bubble wall velocity. None if no solution was found."""

    wallVelocityLTE: float | None
    """Bubble wall velocity in local thermal equilibrium. None when looking for a 
    detonation solution, since no detonation exists in LTE."""

    temperaturePlus: float
    r"""Temperature in front of the bubble, :math:`T_+`,
    from hydrodynamic matching conditions."""

    temperatureMinus: float
    r"""Temperature behind the bubble, :math:`T_-`,
    from hydrodynamic matching conditions."""

    velocityJouguet: float
    r"""Jouguet velocity, :math:`v_J`, the smallest velocity for a detonation."""

    wallWidths: np.ndarray  # 1D array
    """Bubble wall widths in each field direction."""

    wallOffsets: np.ndarray  # 1D array
    """Bubble wall offsets in each field direction."""

    velocityProfile: np.ndarray
    """Fluid velocity as a function of position."""

    fieldProfiles: Fields
    """Field profile as a function of position."""

    temperatureProfile: np.ndarray
    """Temperarture profile as a function of position."""

    linearizationCriterion1: np.ndarray
    r"""Ratio of out-of-equilibrium and equilibrium pressures,
    :math:`|P[\delta f]| / |P[f_\text{eq}]|`. One element for each
    out-of-equilibrium particle."""

    linearizationCriterion2: np.ndarray
    r"""Ratio of collision and Liouville operators in Boltzmann equation,
    :math:`|\mathcal{C}[\delta f]|/ |\mathcal{L}[\delta f]|`. One element for each
    out-of-equilibrium particle."""

    deltaF: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(z, p_z, p_\parallel)`."""

    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`."""

    truncationError: float
    r"""Estimated relative error in :math:`\delta f` due to truncation
    of spectral expansion."""

    deltaFFiniteDifference: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(z, p_z, p_\parallel)`, using finite differences instead
    of spectral expansion."""

    DeltasFiniteDifference: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`,
    using finite differences instead of spectral expansion."""

    solutionType: ESolutionType
    """
    Describes the type of solution obtained. Must be a ESolutionType object. The
    function WallGoManager.solveWall() will return DEFLAGRATION if a solution was found
    and RUNAWAY otherwise. The function WallGoManager.solveWallDetonation() will return
    DETONATION if a solution was found. Otherwise, it returns RUNAWAY if the pressure is
    negative everywhere between vJ and 1, DEFLAGRATION if the pressure is always
    positive, and DEFLAGRATION_OR_RUNAWAY if the pressure is positive at vJ and negative
    at 1 and no stable solution was found. In both cases, returns ERROR if
    success=False.
    """

    success: bool
    """Whether or not the calculation was successful. Will still be True if no solution
    was found, as long as no error happened along the way."""

    message: str
    """Description of the cause of the termination."""

    def __init__(self) -> None:
        pass

    def setWallVelocities(
        self,
        wallVelocity: float | None,
        wallVelocityError: float | None,
        wallVelocityLTE: float | None,
    ) -> None:
        """
        Set wall velocity results
        """
        self.wallVelocity = wallVelocity
        self.wallVelocityError = wallVelocityError
        self.wallVelocityLTE = wallVelocityLTE

    def setHydroResults(self, hydroResults: HydroResults) -> None:
        """
        Set hydrodynamics results
        """
        self.temperaturePlus = hydroResults.temperaturePlus
        self.temperatureMinus = hydroResults.temperatureMinus
        self.velocityJouguet = hydroResults.velocityJouguet

    def setWallParams(self, wallParams: WallParams) -> None:
        """
        Set wall parameters results
        """
        self.wallWidths = wallParams.widths
        self.wallOffsets = wallParams.offsets

    def setBoltzmannBackground(self, boltzmannBackground: BoltzmannBackground) -> None:
        """
        Set Boltzmann background results
        """
        self.velocityProfile = boltzmannBackground.velocityProfile
        self.fieldProfiles = boltzmannBackground.fieldProfiles
        self.temperatureProfile = boltzmannBackground.temperatureProfile

    def setBoltzmannResults(self, boltzmannResults: BoltzmannResults) -> None:
        """
        Set Boltzmann results
        """
        self.deltaF = boltzmannResults.deltaF
        self.Deltas = boltzmannResults.Deltas
        self.truncationError = boltzmannResults.truncationError
        self.linearizationCriterion1 = boltzmannResults.linearizationCriterion1
        self.linearizationCriterion2 = boltzmannResults.linearizationCriterion2

    def setFiniteDifferenceBoltzmannResults(
        self, boltzmannResults: BoltzmannResults
    ) -> None:
        """
        Set finite difference Boltzmann results
        """
        self.deltaFFiniteDifference = boltzmannResults.deltaF
        self.DeltasFiniteDifference = boltzmannResults.Deltas

    def setSuccessState(
        self,
        success: bool,
        solutionType: ESolutionType,
        message: str,
    ) -> None:
        """
        Set the termination message, the success flag and the solution type.
        """
        assert isinstance(success, bool), "WallGoResults Error: success must be a bool."
        assert isinstance(message, str), "WallGoResults Error: message must be a str."
        assert isinstance(solutionType, ESolutionType), (
            "WallGoResults Error: " "solutionType must be a ESolutionType object."
        )
        self.success = success
        self.message = message
        self.solutionType = solutionType
