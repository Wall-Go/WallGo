from .model import FreeEnergy
from .helpers import derivative


class Thermodynamics:
    """
    Thermodynamic functions corresponding to the potential specified
    in the FreeEnergy class
    """

    def __init__(
        self,
        freeEnergy,
    ):
        """Initialisation

        Parameters
        ----------
        freeEnergy : class

        Returns
        -------
        cls: Thermodynamics
            An object of the Thermodynamics class.
        """
        self.freeEnergy = freeEnergy
        self.Tnucl = freeEnergy.Tnucl
        self.Tc = freeEnergy.Tc

    def pHighT(self, T):
        """
        Pressure in the high-temperature phase.
        """
        return self.freeEnergy.pressureHighT(T)

    def dpHighT(self, T):
        """
        Temperature derivative of the pressure in the high-temperature phase.
        """
        return derivative(
            self.freeEnergy.pressureHighT,
            T,
            dx=self.freeEnergy.dT,
            n=1,
            order=4,
        )

    def ddpHighT(self, T):
        """
        Second temperature derivative of the pressure in the high-temperature phase.
        """
        return derivative(
            self.freeEnergy.pressureHighT,
            T,
            dx=self.freeEnergy.dT,
            n=2,
            order=4,
        )

    def eHighT(self, T):
        """
        Energy density in the high-temperature phase.
        """
        return T*self.dpHighT(T) - self.pHighT(T)

    def deHighT(self, T):
        """
        Temperature derivative of the energy density in the high-temperature phase.
        """
        return T*self.ddpHighT(T)

    def wHighT(self,T):
        """
        Enthalpy density in the high-temperature phase.
        """
        return self.pHighT(T)+self.eHighT(T)

    def csqHighT(self,T):
        """
        Sound speed squared in the high-temperature phase.
        """
        return self.dpHighT(T)/self.deHighT(T)

    def pLowT(self, T):
        """
        Pressure in the low-temperature phase
        """
        return self.freeEnergy.pressureLowT(T)

    def dpLowT(self, T):
        """
        Temperature derivative of the pressure in the low-temperature phase.
        """
        return derivative(
            self.freeEnergy.pressureLowT,
            T,
            dx=self.freeEnergy.dT,
            n=1,
            order=4,
        )

    def ddpLowT(self, T):
        """
        Second temperature derivative of the pressure in the low-temperature phase.
        """
        return derivative(
            self.freeEnergy.pressureLowT,
            T,
            dx=self.freeEnergy.dT,
            n=2,
            order=4,
        )

    def eLowT(self, T):
        """
        Energy density in the low-temperature phase.
        """
        return T*self.dpLowT(T) - self.pLowT(T)

    def deLowT(self, T):
        """
        Temperature derivative of the energy density in the low-temperature phase.
        """
        return T*self.ddpLowT(T)

    def wLowT(self,T):
        """
        Enthalpy density in the low-temperature phase.
        """
        return self.pLowT(T)+self.eLowT(T)

    def csqLowT(self,T):
        """
        Sound speed squared in the low-temperature phase.
        """
        return self.dpLowT(T)/self.deLowT(T)
