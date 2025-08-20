"""
Data classes for compiling and returning results
"""

from dataclasses import dataclass
from enum import Enum
import numpy as np
from typing import Tuple
from .fields import Fields
from .containers import BoltzmannBackground, BoltzmannDeltas, WallParams


@dataclass
class BoltzmannResults:
    """
    Holds results to be returned by BoltzmannSolver
    """

    deltaF: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(\xi, p_z, p_\parallel)`."""

    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`."""

    truncationError: float
    r"""Estimated relative error in :math:`\delta f` due to truncation
    of spectral expansion."""

    truncatedTail: tuple[bool, bool, bool]
    """True if tail 1/3 of spectral expansion was truncated in each direction, False otherwise."""

    spectralPeaks: tuple[int, int, int]
    r"""Indices of the peaks in the spectral expansion in each direction."""

    # These two criteria are to evaluate the validity of the linearization of the
    # Boltzmann equation. The arrays contain one element for each out-of-equilibrium
    # particle. To be valid, at least one criterion must be small for each particle.
    linearizationCriterion1: float = 0
    r"""Ratio of out-of-equilibrium and equilibrium pressures,
    :math:`|P[\delta f]| / |P[f_\text{eq}]|`. Default is 0."""

    linearizationCriterion2: float = 0
    r"""Ratio of the first-order correction due to nonlinearities and total pressure
    computed by WallGo, :math:`|P[\delta f_2]| / |P[f_\text{eq}+\delta f]|`.
    Default is 0."""

    def __mul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            truncatedTail=self.truncatedTail,
            spectralPeaks=self.spectralPeaks,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __rmul__(self, number: float) -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=number * self.deltaF,
            Deltas=number * self.Deltas,
            truncationError=abs(number) * self.truncationError,
            truncatedTail=self.truncatedTail,
            spectralPeaks=self.spectralPeaks,
            linearizationCriterion1=abs(number) * self.linearizationCriterion1,
            linearizationCriterion2=self.linearizationCriterion2,
        )

    def __add__(self, other: "BoltzmannResults") -> "BoltzmannResults":
        return BoltzmannResults(
            deltaF=other.deltaF + self.deltaF,
            Deltas=other.Deltas + self.Deltas,
            truncationError=other.truncationError + self.truncationError,
            truncatedTail=self.truncatedTail,
            spectralPeaks=self.spectralPeaks,
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
    r"""
    Used when no stable solution was found while looking for a detonation with a
    positive pressure at :math:`v_w=v_\text{J}` and negative at :math:`v_w=1`.
    """

    ERROR = 5
    """ Indicates that the calculation was not successful. """


@dataclass
class WallGoResults:
    """
    Compiles output results for users of WallGo
    """

    wallVelocity: float | None
    """Bubble wall velocity :math:`v_w`. None if no solution was found."""

    wallVelocityError: float | None
    r"""Estimated error in bubble wall velocity :math:`\delta v_w`. None if no solution was found."""

    wallVelocityLTE: float | None
    r"""Bubble wall velocity in local thermal equilibrium :math:`v_w^\text{LTE}`. None when looking for a 
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
    r"""Bubble wall widths in each field direction, :math:`L_i`."""

    wallOffsets: np.ndarray  # 1D array
    r"""Bubble wall offsets in each field direction, :math:`\delta_i`."""

    velocityProfile: np.ndarray
    r"""Fluid velocity as a function of position, :math:`v_\text{pl}(\xi)`."""

    fieldProfiles: Fields
    r"""Field profile as a function of position, :math:`\phi_i(\xi)`."""

    temperatureProfile: np.ndarray
    r"""Temperarture profile as a function of position, :math:`T(\xi)`."""

    linearizationCriterion1: float
    r"""Ratio of out-of-equilibrium and equilibrium pressures,
    :math:`|P[\delta f]| / |P[f_\text{eq}]|`."""
    
    eomResidual: np.ndarray
    r"""
    Residual of the EOM due to the tanh ansatz. There is one element for each scalar
    field. It is estimated by the integral

    .. math:: \sqrt{\Delta[\mathrm{EOM}^2]/|\mathrm{EOM}^2|}
        
    with 

    .. math:: \Delta[\mathrm{EOM}^2]=\int\! dz\, (-\partial_z^2 \phi+ \partial V_{\mathrm{eq}}/ \partial \phi+ \partial V_{\mathrm{out}}/ \partial \phi )^2

    and

    .. math:: |\mathrm{EOM}^2|=\int\! dz\, [(\partial_z^2 \phi)^2+ (\partial V_{\mathrm{eq}}/ \partial \phi)^2+ (\partial V_{\mathrm{out}}/ \partial \phi)^2].
    """

    linearizationCriterion2: float
    r"""Ratio of the first-order correction due to nonlinearities and total pressure
    computed by WallGo, :math:`|P[\delta f_2]| / |P[f_\text{eq}+\delta f]|`."""

    deltaF: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(\xi, p_z, p_\parallel)`."""

    Deltas: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`."""

    truncationError: float
    r"""Estimated relative error in :math:`\delta f` due to truncation
    of spectral expansion."""

    deltaFFiniteDifference: np.ndarray
    r"""Deviation of probability density function from equilibrium,
    :math:`\delta f(\xi, p_z, p_\parallel)`, using finite differences instead
    of spectral expansion."""

    DeltasFiniteDifference: BoltzmannDeltas  # pylint: disable=invalid-name
    r"""Relativistically invariant integrals over
    :math:`\mathcal{E}_\text{pl}^{n_\mathcal{E}}\mathcal{P}_\text{pl}^{n_\mathcal{P}}\delta f`,
    using finite differences instead of spectral expansion."""

    violationOfEMConservation: Tuple[float,float]
    """
    RMS along the grid of the violation of the conservation of the components T30 and T33 of
    the energy-momentum tensor.
    """

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

    def setViolationOfEMConservation(
        self, violationOfEMConservation: Tuple[float, float]
    ) -> None:
        """
        Set the violation of energy-momentum conservation results
        """
        assert (
            len(violationOfEMConservation) == 2
        ), "WallGoResults Error: violationOfEMConservation must be a tuple of two floats."
        self.violationOfEMConservation = violationOfEMConservation

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