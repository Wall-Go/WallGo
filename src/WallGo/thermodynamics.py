"""
Classes that contain thermodynamics quantities like pressure, enthalpy, energy density 
for both phases
"""

import numpy as np
import numpy.typing as npt
import scipy.optimize
from typing import Tuple

from .EffectivePotential import EffectivePotential
from .exceptions import WallGoError
from .Fields import Fields
from .FreeEnergy import FreeEnergy

# TODO should make most functions here non-public

class Thermodynamics:
    """
    Thermodynamic functions corresponding to the effectivePotential
    """

    def __init__(
        self,
        effectivePotential: EffectivePotential,
        nucleationTemperature: float,
        phaseLowT: Fields,
        phaseHighT: Fields
    ):
        """Initialisation

        Parameters
        ----------
        effectivePotential : EffectivePotential
            An object of the EffectivePotential class.
        nucleationTemperature : float
            The nucleation temperature.
        phaseLowT : Fields
            The location of the low temperature phase at the nucleation
            temperature. Does not need to be exact, as resolved internally
            with input as starting point.
        phaseHighT: Fields
            The location of the high temperature phase at the nucleation
            temperature. Does not need to be exact, as resolved internally
            with input as starting point.

        Returns
        -------
        cls: Thermodynamics
            An object of the Thermodynamics class.
        """
        self.effectivePotential = effectivePotential
        self.Tnucl = nucleationTemperature
        self.phaseLowT = phaseLowT
        self.phaseHighT = phaseHighT

        self.freeEnergyHigh = FreeEnergy(
            self.effectivePotential,
            self.Tnucl,
            self.phaseHighT,
        )
        self.freeEnergyLow = FreeEnergy(
            self.effectivePotential,
            self.Tnucl,
            self.phaseLowT,
        )

    def getCoexistenceRange(self) -> Tuple[float, float]:
        """
        Finds the temperature range where the two phases coexist

        Parameters
        ----------

        Returns
        -------
        TMin, Tmax:
            The minimum and maximum temperature of phase coexistence
        """
        TMin = max(
            self.freeEnergyHigh.minPossibleTemperature,
            self.freeEnergyLow.minPossibleTemperature,
        )
        TMax = min(
            self.freeEnergyHigh.maxPossibleTemperature,
            self.freeEnergyLow.maxPossibleTemperature,
        )
        return (TMin, TMax)

    def findCriticalTemperature(
        self, dT: float, rTol: float = 1e-6, paranoid: bool = True
    ) -> float:
        """
        Computes the critical temperature by finding the temperature for which the 
        free energy of both phases is equal.

        Parameters
        ----------
        dT: float
            Temperature step size for the determination of Tc
        rTol: float, optional 
            Error tolerance for the phase tracing
        paranoid: bool, optional
            Setting for phase tracing. When True, recomputes minimum at every step

        paranoid: bool
        
        Returns
        -------
        Tc:
            The value of the critical temperature
        """
        # getting range over which both phases naively exist
        # (if we haven't traced the phases yet)
        TMin, TMax = self.getCoexistenceRange()
        if TMin > TMax:
            raise WallGoError(
                "findCriticalTemperature needs TMin < TMax",
                {"TMax": TMax, "TMin": TMin},
            )

        # tracing phases and ensuring they are stable
        if not self.freeEnergyHigh.hasInterpolation():
            print(f"Tracing high-T phase: {TMin=}, {TMax=}, {dT=}, {rTol=}")
            self.freeEnergyHigh.tracePhase(
                TMin, TMax, dT, rTol, spinodal=True, paranoid=paranoid
            )
        if not self.freeEnergyLow.hasInterpolation():
            print("Tracing low-T phase")
            self.freeEnergyLow.tracePhase(
                TMin, TMax, dT, rTol, spinodal=True, paranoid=paranoid
            )

        # getting range over which both phases are stable
        TMin, TMax = self.getCoexistenceRange()

        # Wrapper that computes free-energy difference between our phases.
        # This goes into scipy so scalar in, scalar out
        def freeEnergyDifference(inputT: np.double) -> np.double:
            f1 = self.freeEnergyHigh(inputT).veffValue
            f2 = self.freeEnergyLow(inputT).veffValue
            diff = f2 - f1
            # Force into scalar type. This errors out if the size is not 1;
            # no failsafes to avoid overhead
            return diff.item()

        # start from TMax and decrease temperature in small steps until
        # the free energy difference changes sign
        T = TMax
        TStep = (TMax - TMin) / 10
        signAtStart = np.sign(freeEnergyDifference(T))
        bConverged = False

        while T > TMin:
            T -= TStep
            if np.sign(freeEnergyDifference(T)) != signAtStart:
                bConverged = True
                break

        if not bConverged:
            raise WallGoError("Could not find critical temperature")

        # Improve Tc estimate by solving DeltaF = 0 in narrow range near T
        # NB: bracket will break if the function has same sign on both ends.
        # The rough loop above should prevent this.
        rootResults = scipy.optimize.root_scalar(
            freeEnergyDifference,
            bracket=(T, T + TStep),
            method="brentq",
            rtol=rTol,
            xtol=min(rTol * T, 0.5 * dT),
        )

        if not rootResults.converged:
            raise WallGoError(
                "Error finding critical temperature",
                rootResults,
            )

        return rootResults.root

    def pHighT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Pressure in the high-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        pHighT : array-like (float)
            Pressure in the high-temperature phase.

        """
        veffValue = self.freeEnergyHigh(temperature).veffValue
        return -veffValue

    def dpHighT(self, temperature: np.ndarray) -> np.ndarray:
        """
        Temperature derivative of the pressure in the high-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        dpHighT : array-like (float)
            Temperature derivative of the pressure in the high-temperature phase.
        """
        return -self.freeEnergyHigh.derivative(temperature, order=1).veffValue

    ## LN: could just have something like dpdT(n) that calculates nth order derivative
    def ddpHighT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Second temperature derivative of the pressure in the high-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        ddpHighT : array-like (float)
            Second temperature derivative of the pressure in the high-temperature phase.
        """
        return -self.freeEnergyHigh.derivative(temperature, order=2).veffValue

    def eHighT(self, temperature: np.ndarray) -> np.ndarray:
        r"""
        Energy density in the high-temperature phase, obtained via 
        :math:`e(T) = T \frac{dp}{dT}-p`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        eHighT : array-like (float)
            Energy density in the high-temperature phase.
        """
        return temperature * self.dpHighT(temperature) - self.pHighT(temperature)

    def deHighT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Temperature derivative of the energy density in the high-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        deHighT : array-like (float)
            Temperature derivative of the energy density in the high-temperature phase.
        """
        return temperature * self.ddpHighT(temperature)

    def wHighT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Enthalpy density in the high-temperature phase, obtained via
        :math:`w(T) = p(T)+e(T)`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        wHighT : array-like (float)
            Enthalpy density in the high-temperature phase.
        """
        return temperature * self.dpHighT(temperature)

    def csqHighT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Sound speed squared in the high-temperature phase,
        obtained via :math:`c_s^2 = \frac{dp/dT}{de/dT}`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        csqHighT : array-like (float)
            Sound speed squared in the high-temperature phase.
        """
        return self.dpHighT(temperature) / self.deHighT(temperature)

    def pLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Pressure in the low-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        pLowT : array-like (float)
            Pressure in the low-temperature phase.
        """

        VeffValue = self.freeEnergyLow(temperature).veffValue
        return -VeffValue

    def dpLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Temperature derivative of the pressure in the low-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        dpLowT : array-like (float)
            Temperature derivative of the pressure in the low-temperature phase.
        """
        return -self.freeEnergyLow.derivative(temperature, order=1).veffValue

    def ddpLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Second temperature derivative of the pressure in the low-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        ddpLowT : array-like (float)
            Second temperature derivative of the pressure in the low-temperature phase.
        """
        return -self.freeEnergyLow.derivative(temperature, order=2).veffValue

    def eLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Energy density in the low-temperature phase,
        obtained via :math:`e(T) = T \frac{dp}{dT}-p`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        eLowT : array-like (float)
            Energy density in the low-temperature phase.
        """
        return temperature * self.dpLowT(temperature) - self.pLowT(temperature)

    def deLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """
        Temperature derivative of the energy density in the low-temperature phase.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        deLowT : array-like (float)
            Temperature derivative of the energy density in the low-temperature phase.
        """
        return temperature * self.ddpLowT(temperature)

    def wLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Enthalpy density in the low-temperature phase,
        obtained via :math:`w(T) = p(T)+e(T)`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        wLowT : array-like (float)
            Enthalpy density in the low-temperature phase.
        """
        return temperature * self.dpLowT(temperature)

    def csqLowT(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        Sound speed squared in the low-temperature phase, 
        obtained via :math:`c_s^2 = \frac{dp/dT}{de/dT}`.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        csqLowT : array-like (float)
            Sound speed squared in the low-temperature phase.
        """
        return self.dpLowT(temperature) / self.deLowT(temperature)

    def alpha(self, T: npt.ArrayLike) -> npt.ArrayLike:
        r"""
        The phase transition strength at the temperature :math:`T`, computed via
        :math:`\alpha = \frac{eHighT(T)-eLowT(T) -(pHighT(T)-pLowT(T)) 
        /csqLowT(T)}{3wHighT(T)}`

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        alpha : array-like (float)
            Phase transition strength.
        """
        # LN: keeping T instead of 'temperature' here since the expression is long
        # LN: Please add reference to a paper and eq number
        return (
            (
                self.eHighT(T)
                - self.eLowT(T)
                - (self.pHighT(T) - self.pLowT(T)) / self.csqLowT(T)
            )
            / 3
            / self.wHighT(T)
        )


class ThermodynamicsExtrapolate:
    """
    Thermodynamics functions corresponding to the potential, extrapolated with the
    template model [LM15]_ outside of the allowed temperture range of the potential
    """

    def __init__(
        self,
        thermodynamics: Thermodynamics,
    ):
        """Initialisation

        Parameters
        ----------
        thermodynamics : Thermodynamics
            A thermodynamics object that is not extrapolated outside of the allowed
            temperature ranges of freeEnergyHigh and freeEnergyLow

        Returns
        -------
        cls: ThermodynamicsExtrapolate
            An object of the ThermodynamicsExtrapolate class.

        References
        ----------
        .. [LM15] L. Leitao and A. Megevand, Hydrodynamics of phase transition fronts and 
            the speed of sound in the plasma, Nucl.Phys.B 891 (2015) 159-199
            doi:10.1016/j.nuclphysb.2014.12.008
        """
        self.thermodynamics = thermodynamics
        self.TMaxHighT = thermodynamics.freeEnergyHigh.maxPossibleTemperature
        self.TMinHighT = thermodynamics.freeEnergyHigh.minPossibleTemperature
        self.TMaxLowT = thermodynamics.freeEnergyLow.maxPossibleTemperature
        self.TMinLowT = thermodynamics.freeEnergyLow.minPossibleTemperature

        # The following parameters are defined such that the thermodynamic quantities
        # can be extrapolated beyond the minimum and maximum temperatures
        # by mapping onto the template model
        self.muMinHighT = 1 + 1 / self.thermodynamics.csqHighT(self.TMinHighT)
        self.aMinHighT = (
            3
            * self.thermodynamics.wHighT(self.TMinHighT)
            / (self.muMinHighT * pow(self.TMinHighT, self.muMinHighT))
        )
        self.epsilonMinHighT = 1 / 3.0 * self.aMinHighT * pow(
            self.TMinHighT, self.muMinHighT
        ) - thermodynamics.pHighT(self.TMinHighT)
        self.muMaxHighT = 1 + 1 / self.thermodynamics.csqHighT(self.TMaxHighT)
        self.aMaxHighT = (
            3
            * self.thermodynamics.wHighT(self.TMaxHighT)
            / (self.muMaxHighT * pow(self.TMaxHighT, self.muMaxHighT))
        )
        self.epsilonMaxHighT = 1 / 3.0 * self.aMaxHighT * pow(
            self.TMaxHighT, self.muMaxHighT
        ) - thermodynamics.pHighT(self.TMaxHighT)

        self.muMinLowT = 1 + 1 / self.thermodynamics.csqLowT(self.TMinLowT)
        self.aMinLowT = (
            3
            * self.thermodynamics.wLowT(self.TMinLowT)
            / (self.muMinLowT * pow(self.TMinLowT, self.muMinLowT))
        )
        self.epsilonMinLowT = 1 / 3.0 * self.aMinLowT * pow(
            self.TMinLowT, self.muMinLowT
        ) - thermodynamics.pLowT(self.TMinLowT)
        self.muMaxLowT = 1 + 1 / self.thermodynamics.csqLowT(self.TMaxLowT)
        self.aMaxLowT = (
            3
            * self.thermodynamics.wLowT(self.TMaxLowT)
            / (self.muMaxLowT * pow(self.TMaxLowT, self.muMaxLowT))
        )
        self.epsilonMaxLowT = 1 / 3.0 * self.aMaxLowT * pow(
            self.TMaxLowT, self.muMaxLowT
        ) - thermodynamics.pLowT(self.TMaxLowT)

    def pHighT(self, temperature: float) -> float:
        r"""
        Pressure in the high-temperature phase, obtained from thermodynamics.pHighT 
        for the allowed temperature range and extrapolated to the template model 
        outside of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        pHighT : array-like (float)
            Pressure in the high-temperature phase.
        """
        if temperature < self.TMinHighT:
            return (
                1 / 3.0 * self.aMinHighT * pow(temperature, self.muMinHighT)
                - self.epsilonMinHighT
            )
        elif temperature > self.TMaxHighT:
            return (
                1 / 3.0 * self.aMaxHighT * pow(temperature, self.muMaxHighT)
                - self.epsilonMaxHighT
            )
        else:
            return self.thermodynamics.pHighT(temperature)

    def dpHighT(self, temperature: float) -> float:
        r"""
        Temperature-derivative of the pressure in the high-temperature phase, 
        obtained from thermodynamics.dpHighT for the allowed temperature range
        and extrapolated to the template model outside of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        dpHighT : array-like (float)
            Temperature-derivative of the pressure in the high-temperature phase.
        """
        if temperature < self.TMinHighT:
            return (
                1
                / 3.0
                * self.muMinHighT
                * self.aMinHighT
                * pow(temperature, self.muMinHighT - 1)
            )
        elif temperature > self.TMaxHighT:
            return (
                1
                / 3.0
                * self.muMaxHighT
                * self.aMaxHighT
                * pow(temperature, self.muMaxHighT - 1)
            )
        else:
            return self.thermodynamics.dpHighT(temperature)

    def ddpHighT(self, temperature: float) -> float:
        r"""
        Second temperature-derivative of the pressure in the high-temperature phase, 
        obtained from thermodynamics.ddpHighT for the allowed temperature range
        and extrapolated to the template model outside of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        ddpHighT : array-like (float)
            Second temperature-derivative of the pressure in the high-temperature phase.
        """
        if temperature < self.TMinHighT:
            return (
                1
                / 3.0
                * self.muMinHighT
                * (self.muMinHighT - 1)
                * self.aMinHighT
                * pow(temperature, self.muMinHighT - 2)
            )
        elif temperature > self.TMaxHighT:
            return (
                1
                / 3.0
                * self.muMaxHighT
                * (self.muMaxHighT - 1)
                * self.aMaxHighT
                * pow(temperature, self.muMaxHighT - 2)
            )
        else:
            return self.thermodynamics.ddpHighT(temperature)

    def eHighT(self, temperature: float) -> float:
        r"""
        Energy density in the high-temperature phase, 
        obtained via :math:`e(T) = T \frac{dp}{dT}-p`,
        valid outside of the allowed temeperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        eHighT : array-like (float)
            Energy density in the high-temperature phase.
        """
        return temperature * self.dpHighT(temperature) - self.pHighT(temperature)

    def deHighT(self, temperature: float) -> float:
        r"""
        Temperature derivative of the energy density in the high-temperature phase,
        obtained via :math:`e(T) = T \frac{d^2p}{dT^2}`,
        valid outside of the allowed temperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        deHighT : array-like (float)
            Temperature derivative of the energy density in the high-temperature phase.
        """
        return temperature * self.ddpHighT(temperature)

    def wHighT(self, temperature: float) -> float:
        r"""
        Enthalpy density in the high-temperature phase, obtained via
        :math:`w(T) = p(T)+e(T)`, outside of the allowed temperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        wHighT : array-like (float)
            Enthalpy density in the high-temperature phase.
        """
        return temperature * self.dpHighT(temperature)

    def csqHighT(self, temperature: float) -> float:
        r"""
        Sound speed squared in the high-temperature phase, obtained via
        :math:`c_s^2 = \frac{dp/dT}{de/dT}` inside of the allowed temperature range and
        by its value at TMinHighT and TMaxHighT outside of the allowed temperature
        range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        csqHighT : array-like (float)
            Sound speed squared in the high-temperature phase.
        """
        if temperature < self.TMinHighT:
            return self.thermodynamics.csqHighT(self.TMinHighT)
        elif temperature > self.TMaxHighT:
            return self.thermodynamics.csqHighT(self.TMaxHighT)
        else:
            return self.thermodynamics.csqHighT(temperature)

    def pLowT(self, temperature: float) -> float:
        r"""
        Pressure in the low-temperature phase, obtained from thermodynamics.pLowT for 
        the allowed temperature range and extrapolated to the template model outside 
        of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        pLowT : array-like (float)
            Pressure in the low-temperature phase.
        """
        if temperature < self.TMinLowT:
            return (
                1 / 3.0 * self.aMinLowT * pow(temperature, self.muMinLowT)
                - self.epsilonMinLowT
            )
        elif temperature > self.TMaxLowT:
            return (
                1 / 3.0 * self.aMaxLowT * pow(temperature, self.muMaxLowT)
                - self.epsilonMaxLowT
            )
        else:
            return self.thermodynamics.pLowT(temperature)

    def dpLowT(self, temperature: float) -> float:
        r"""
        Temperature-derivative of the pressure in the low-temperature phase, 
        obtained from thermodynamics.dpLowT for the allowed temperature range
        and extrapolated to the template model outside of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        dpLowT : array-like (float)
            Temperature-derivative of the pressure in the low-temperature phase.
        """
        if temperature < self.TMinLowT:
            return (
                1
                / 3.0
                * self.muMinLowT
                * self.aMinLowT
                * pow(temperature, self.muMinLowT - 1)
            )
        elif temperature > self.TMaxLowT:
            return (
                1
                / 3.0
                * self.muMaxLowT
                * self.aMaxLowT
                * pow(temperature, self.muMaxLowT - 1)
            )
        else:
            return self.thermodynamics.dpLowT(temperature)

    def ddpLowT(self, temperature: float) -> float:
        r"""
        Second temperature-derivative of the pressure in the low-temperature phase, 
        obtained from thermodynamics.ddpLowT for the allowed temperature range
        and extrapolated to the template model outside of the allowed temperature range

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        ddpLowT : array-like (float)
            Second temperature-derivative of the pressure in the low-temperature phase.
        """
        if temperature < self.TMinLowT:
            return (
                1
                / 3.0
                * self.muMinLowT
                * (self.muMinLowT - 1)
                * self.aMinLowT
                * pow(temperature, self.muMinLowT - 2)
            )
        elif temperature > self.TMaxLowT:
            return (
                1
                / 3.0
                * self.muMaxLowT
                * (self.muMaxLowT - 1)
                * self.aMaxLowT
                * pow(temperature, self.muMaxLowT - 2)
            )
        else:
            return self.thermodynamics.ddpLowT(temperature)

    def eLowT(self, temperature: float) -> float:
        r"""
        Energy density in the low-temperature phase, obtained via 
        :math:`e(T) = T \frac{dp}{dT}-p`, valid outside the allowed temeperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        eLowT : array-like (float)
            Energy density in the low-temperature phase.
        """
        return temperature * self.dpLowT(temperature) - self.pLowT(temperature)

    def deLowT(self, temperature: float) -> float:
        r"""
        Temperature derivative of the energy density in the low-temperature phase, 
        obtained via :math:`e(T) = T \frac{d^2p}{dT^2}`,
        valid outside of the allowed temperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        deLowT : array-like (float)
            Temperature derivative of the energy density in the low-temperature phase.
        """
        return temperature * self.ddpLowT(temperature)

    def wLowT(self, temperature: float) -> float:
        r"""
        Enthalpy density in the low-temperature phase, obtained via 
        :math:`w(T) = p(T)+e(T)`, outside of the allowed temperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        wLowT : array-like (float)
            Enthalpy density in the low-temperature phase.
        """
        return temperature * self.dpLowT(temperature)

    def csqLowT(self, temperature: float) -> float:
        r"""
        Sound speed squared in the low-temperature phase, obtained via
        :math:`c_s^2 = \frac{dp/dT}{de/dT}` inside of the allowed temperature range and
        by its value at TMinLowT and TMaxLowT outside of the allowed temperature range.

        Parameters
        ----------
        temperature : array-like
            Temperature(s)

        Returns
        -------
        csqLowT : array-like (float)
            Sound speed squared in the low-temperature phase.
        """
        if temperature < self.TMinLowT:
            return self.thermodynamics.csqLowT(self.TMinLowT)
        elif temperature > self.TMaxLowT:
            return self.thermodynamics.csqLowT(self.TMaxLowT)
        else:
            return self.thermodynamics.csqLowT(temperature)
