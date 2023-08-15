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


    #JvdV: We should replace broken/symm labels with lowT/highT. Haven't done it yet, because hydro uses the same
    #therminology

    #Pressure in symmetric phase
    def pSym(self, T):
        return self.freeEnergy.pressureHighT(T)

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
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
    def ddpSym(self, T):
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
    def eSym(self, T):
        return T*self.dpSym(T) - self.pSym(T)

    #T-derivative of the energy density in the symmetric phase
    def deSym(self, T):
        return T*self.ddpSym(T)

    #Enthalpy in the symmetric phase
    def wSym(self,T):
        return self.pSym(T)+self.eSym(T)

    #Sound speed squared in the symmetric phase
    def csqSym(self,T):
        return self.dpSym(T)/self.deSym(T)

        #Pressure in the broken phase
    def pBrok(self, T):
        return self.freeEnergy.pressureLowT(T)

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
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
    def ddpBrok(self, T):
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
    def eBrok(self, T):
        return T*self.dpBrok(T) - self.pBrok(T)

    #T-derivative of the energy density in the broken phase
    def deBrok(self, T):
        return T*self.ddpBrok(T)

    #Enthalpy in the symmetric phase
    def wBrok(self,T):
        return self.pBrok(T)+self.eBrok(T)

    #Sound speed squared in the broken phase
    def csqBrok(self,T):
        return self.dpBrok(T)/self.deBrok(T)
