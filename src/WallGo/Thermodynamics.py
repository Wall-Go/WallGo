import numpy as np
import scipy.optimize

from .EffectivePotential import EffectivePotential
from .Fields import Fields
from .FreeEnergy import FreeEnergy
from .WallGoExceptions import WallGoError
import WallGo.helpers


""" LN: As far as I understand, this class is intended to work as an intermediator between the Veff and other parts of the code 
that require T-dependent quantities (like Hydro).  
I've modified this so that instead of a FreeEnergy object, this operates on an EffectivePotential object; this gets rid of one layer of complexity. 
The potential itself does not contain info about Tc or Tn, so those are now given as inputs to the Thermodynamics constructor.
Finally, we need some indicator of how to find the low- and high-T phases from Veff. 
For now, just take in the field values in both phases at Tn: these can be given as initial guesses to Veff.findLocalMinimum().
As a future improvement, I propose we pre-calculate the lowT/highT phase locations over some sensible T-range.

Questions: 
1. Is it necessary to have separate functions like pHighT, pLowT etc?
2. Should we get rid of labels highT, lowT altogether and use something like phase1/phase2 or phaseStart/phaseEnd? 
"""

# TODO should make most functions here non-public

## LN: I don't think this needs Tc tbh. It seems to be only accessed in Hydro, and used to set some upper bounds?
## In which case I'd propose Hydro to use separate variables for its T-range for more flexibility, 
## and so that we don't have to store Tc in many places. 
  
class Thermodynamics:
    """
    Thermodynamic functions corresponding to the potential
    """

    effectivePotential: EffectivePotential

    def __init__(
        self,
        effectivePotential: EffectivePotential,
        nucleationTemperature: float,
        phaseLowT: Fields,
        phaseHighT: Fields,
        criticalTemperature: float = None,
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
        criticalTemperature: float = None
            Optional input critical temperature. If not given, will be
            solved for numerically.

        Returns
        -------
        cls: Thermodynamics
            An object of the Thermodynamics class.
        """
        self.effectivePotential = effectivePotential
        self.Tnucl = nucleationTemperature
        self.Tc = criticalTemperature
        self.phaseLowT = phaseLowT
        self.phaseHighT = phaseHighT

        # Small temperature difference to use for derivatives.
        # TODO this needs to be read from a config file or solved
        # for, based on some desired tolerance.
        self.dT = 1e-2

        self.freeEnergyHigh = FreeEnergy(
            self.effectivePotential, self.Tnucl, self.phaseHighT,
        )
        self.freeEnergyLow = FreeEnergy(
            self.effectivePotential, self.Tnucl, self.phaseLowT,
        )

    def getCoexistenceRange(self):
        """
        Ensures that there is phase coexistence, by comparing the temperature ranges
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
        Computes the critical temperature
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
            print(f"Hi: tracing high-T phase: {TMin=}, {TMax=}, {dT=}, {rTol=}")
            self.freeEnergyHigh.tracePhase(
                TMin, TMax, dT, rTol, spinodal=True, paranoid=paranoid
            )
        if not self.freeEnergyLow.hasInterpolation():
            print("Hi: tracing low-T phase")
            self.freeEnergyLow.tracePhase(
                TMin, TMax, dT, rTol, spinodal=True, paranoid=paranoid
            )

        # getting range over which both phases are stable
        TMin, TMax = self.getCoexistenceRange()

        # Wrapper that computes free-energy difference between our phases.
        # This goes into scipy so scalar in, scalar out
        def freeEnergyDifference(inputT: np.double) -> np.double:
            f1 = self.freeEnergyHigh(inputT).getVeffValue()
            f2 = self.freeEnergyLow(inputT).getVeffValue()
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

        while (T > TMin):
            T -= TStep
            if (np.sign(freeEnergyDifference(T)) != signAtStart):
                bConverged = True
                break

        if (not bConverged):
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

    def pHighT(self, T: float):
        """
        Pressure in the high-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        pHighT : double
            Pressure in the high-temperature phase.

        """
        VeffValue = self.freeEnergyHigh(T).getVeffValue()
        return -VeffValue

    def dpHighT(self, T):
        """
        Temperature derivative of the pressure in the high-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        dpHighT : double
            Temperature derivative of the pressure in the high-temperature phase.
        """
        return WallGo.helpers.derivative(
            self.pHighT,
            T,
            dx=self.dT,
            n=1,
            order=4,
        )

    ## LN: could just have something like dpdT(n) that calculates nth order derivative
    def ddpHighT(self, T):
        """
        Second temperature derivative of the pressure in the high-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        ddpHighT : double
            Second temperature derivative of the pressure in the high-temperature phase.
        """
        return WallGo.helpers.derivative(
            self.pHighT,
            T,
            dx=self.dT,
            n=2,
            order=4,
        )

    def eHighT(self, T):
        r"""
        Energy density in the high-temperature phase, obtained via :math:`e(T) = T \frac{dp}{dT}-p`.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        eHighT : double
            Energy density in the high-temperature phase.
        """
        return T*self.dpHighT(T) - self.pHighT(T)

    def deHighT(self, T):
        """
        Temperature derivative of the energy density in the high-temperature phase.
        
        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        deHighT : double
            Temperature derivative of the energy density in the high-temperature phase.
        """
        return T*self.ddpHighT(T)

    def wHighT(self,T):
        r"""
        Enthalpy density in the high-temperature phase, obtained via :math:`w(T) = p(T)+e(T)`.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        wHighT : double
            Enthalpy density in the high-temperature phase.
        """
        return self.pHighT(T)+self.eHighT(T)

    def csqHighT(self,T):
        r"""
        Sound speed squared in the high-temperature phase, obtained via :math:`c_s^2 = \frac{dp/dT}{de/dT}`.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        csqHighT : double
            Sound speed squared in the high-temperature phase.
        """
        return self.dpHighT(T)/self.deHighT(T)

    def pLowT(self, T):
        """
        Pressure in the low-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        pLowT : double
            Pressure in the low-temperature phase.

        """

        VeffValue = self.freeEnergyLow(T).getVeffValue()
        return -VeffValue

    def dpLowT(self, T):
        """
        Temperature derivative of the pressure in the low-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        dpLowT : double
            Temperature derivative of the pressure in the low-temperature phase.
        """
        return WallGo.helpers.derivative(
            self.pLowT,
            T,
            dx=self.dT,
            n=1,
            order=4,
        )

    def ddpLowT(self, T):
        """
        Second temperature derivative of the pressure in the low-temperature phase.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        ddpLowT : double
            Second temperature derivative of the pressure in the low-temperature phase.
        """
        return WallGo.helpers.derivative(
            self.pLowT,
            T,
            dx=self.dT,
            n=2,
            order=4,
        )

    def eLowT(self, T):
        r"""
        Energy density in the low-temperature phase, obtained via :math:`e(T) = T \frac{dp}{dT}-p`.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        eLowT : double
            Energy density in the low-temperature phase.
        """
        return T*self.dpLowT(T) - self.pLowT(T)

    def deLowT(self, T):
        """
        Temperature derivative of the energy density in the low-temperature phase.
        
        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        deLowT : double
            Temperature derivative of the energy density in the low-temperature phase.
        """
        return T*self.ddpLowT(T)

    def wLowT(self,T):
        r"""
        Enthalpy density in the low-temperature phase, obtained via :math:`w(T) = p(T)+e(T)`.

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        wLowT : double
            Enthalpy density in the low-temperature phase.
        """
        return self.pLowT(T)+self.eLowT(T)

    def csqLowT(self,T):
        r"""
        Sound speed squared in the low-temperature phase, obtained via :math:`c_s^2 = \frac{dp/dT}{de/dT}`.

        Parameters
        ----------
        T : double
            Temperature
e
        Returns
        -------
        csqLowT : double
            Sound speed squared in the low-temperature phase.
        """
        return self.dpLowT(T)/self.deLowT(T)

    def alpha(self,T):
        r"""
        The phase transition strength at the temperature :math:`T`, computed via 
        :math:`\alpha = \frac{(eHighT(T)-pHighT(T)/csqHighT(T))-(eLowT(T)-pLowT(T)/csqLowT(T))}{3wHighT(T)}`

        Parameters
        ----------
        T : double
            Temperature

        Returns
        -------
        alpha : double
            Phase transition strength.
        """
        # LN: Please add reference to a paper and eq number
        return (self.eHighT(T)-self.pHighT(T)/self.csqHighT(T)-self.eLowT(T)+self.pLowT(T)/self.csqLowT(T))/3/self.wHighT(T)

