import numpy as np

from .EffectivePotential import EffectivePotential
from .FreeEnergy import FreeEnergy

import WallGo.helpers

from .Fields import Fields

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
        effectivePotential,
        criticalTemperature: float,
        nucleationTemperature: float, 
        phaseLowT: Fields,
        phaseHighT: Fields
    ):
        """Initialisation

        Parameters
        ----------
        effectivePotential : class
        nucleationTemperature : float
        criticalTemperature : float

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

        self.freeEnergyHigh = FreeEnergy(self.effectivePotential, self.phaseHighT, temperatureScale=self.Tc-self.Tnucl)
        self.freeEnergyLow = FreeEnergy(self.effectivePotential, self.phaseLowT, temperatureScale=self.Tc-self.Tnucl)


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
        return -self.freeEnergyHigh(T, derivOrder=1).getVeffValue()

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
        return -self.freeEnergyHigh(T, derivOrder=2).getVeffValue()

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
        return T*self.dpHighT(T)

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
        return -self.freeEnergyLow(T, derivOrder=1).getVeffValue()

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
        return -self.freeEnergyLow(T, derivOrder=2).getVeffValue()

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
        return T*self.dpLowT(T)

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
        return (self.eHighT(T)-self.eLowT(T)-(self.pHighT(T)-self.pLowT(T))/self.csqLowT(T))/3/self.wHighT(T)

