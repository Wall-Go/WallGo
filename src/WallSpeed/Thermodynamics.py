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

        Note that the derivatives of the pressures are currently hard-coded.
        These all need to be modified.
        """
        self.freeEnergy = freeEnergy
        self.Tnucl = freeEnergy.Tnucl
        self.Tc = freeEnergy.Tc



    #Pressure in high T phase
    def pHighT(self, T):
        return self.freeEnergy.pressureHighT(T)

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        #p = self.freeEnergy.params # hard coded!
        #return (T**3*(p["ts"]**2 + 4*p["b"]*p["lams"])-p["ts"]*T*p["mussq"])/p["lams"]
        return derivative(
            self.freeEnergy.pressureHighT,
            T,
            dx=self.freeEnergy.dT,
            n=1,
            order=4,
        )

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        #p = self.freeEnergy.params # hard coded!
        #return (3*T**2*(p["ts"]**2+4*p["b"]*p["lams"])-p["ts"]*p["mussq"])/p["lams"]
        return derivative(
            self.freeEnergy.pressureHighT,
            T,
            dx=self.freeEnergy.dT,
            n=2,
            order=4,
        )

    #Energy density in the high T phase
    def eHighT(self, T):
        return T*self.dpHighT(T) - self.pHighT(T)

    #T-derivative of the energy density in the high T phase
    def deHighT(self, T):
        return T*self.ddpHighT(T)

    #Enthalpy in the high T phase
    def wHighT(self,T):
        return self.pHighT(T)+self.eHighT(T)

    #Sound speed squared in the high T phase
    def csqHighT(self,T):
        return self.dpHighT(T)/self.deHighT(T)

        #Pressure in the low T phase
    def pLowT(self, T):
        return self.freeEnergy.pressureLowT(T)

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        #p = self.freeEnergy.params # hard coded!
        #return (T**3*(p["th"]**2 + 4*p["b"]*p["lamh"])-p["th"]*T*p["muhsq"])/p["lamh"]
        return derivative(
            self.freeEnergy.pressureLowT,
            T,
            dx=self.freeEnergy.dT,
            n=1,
            order=4,
        )

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        #p = self.freeEnergy.params # hard coded!
        #return (3*T**2*(p["th"]**2+4*p["b"]*p["lamh"])-p["th"]*p["muhsq"])/p["lamh"]
        return derivative(
            self.freeEnergy.pressureLowT,
            T,
            dx=self.freeEnergy.dT,
            n=2,
            order=4,
        )

    #Energy density in the low T phase
    def eLowT(self, T):
        return T*self.dpLowT(T) - self.pLowT(T)

    #T-derivative of the energy density in the low T phase
    def deLowT(self, T):
        return T*self.ddpLowT(T)

    #Enthalpy in the high T phase
    def wLowT(self,T):
        return self.pLowT(T)+self.eLowT(T)

    #Sound speed squared in the low T phase
    def csqLowT(self,T):
        return self.dpLowT(T)/self.deLowT(T)
