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


    #JvdV: We should replace broken/symm labels with lowT/highT. Haven't done it yet, because hydro uses the same
    #therminology

    #Pressure in symmetric phase
    def pHighT(self, T):
        return self.freeEnergy.pressureHighT(T)

    #T-derivative of the pressure in the symmetric phase
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

    #Second T-derivative of the pressure in the symmetric phase
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

    #Energy density in the symmetric phase
    def eHighT(self, T):
        return T*self.dpHighT(T) - self.pHighT(T)

    #T-derivative of the energy density in the symmetric phase
    def deHighT(self, T):
        return T*self.ddpHighT(T)

    #Enthalpy in the symmetric phase
    def wHighT(self,T):
        return self.pHighT(T)+self.eHighT(T)

    #Sound speed squared in the symmetric phase
    def csqHighT(self,T):
        return self.dpHighT(T)/self.deHighT(T)

        #Pressure in the broken phase
    def pLowT(self, T):
        return self.freeEnergy.pressureLowT(T)

    #T-derivative of the pressure in the broken phase
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

    #Second T-derivative of the pressure in the broken phase
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

    #Energy density in the broken phase
    def eLowT(self, T):
        return T*self.dpLowT(T) - self.pLowT(T)

    #T-derivative of the energy density in the broken phase
    def deLowT(self, T):
        return T*self.ddpLowT(T)

    #Enthalpy in the symmetric phase
    def wLowT(self,T):
        return self.pLowT(T)+self.eLowT(T)

    #Sound speed squared in the broken phase
    def csqLowT(self,T):
        return self.dpLowT(T)/self.deLowT(T)
