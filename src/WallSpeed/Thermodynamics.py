from .model import FreeEnergy

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

        self.muhsq = freeEnergy.muhsq # this function shouldn't need to know these parameters
        self.lamh = freeEnergy.lamh
        self.mussq = freeEnergy.mussq
        self.lams = freeEnergy.lams

        self.th = freeEnergy.th
        self.ts = freeEnergy.ts

        self.b = np.pi**2 *107.75/90


    #JvdV: We should replace broken/symm labels with lowT/highT. Haven't done it yet, because hydro uses the same
    #therminology

    #Pressure in symmetric phase
    def pSym(self, T):
        return self.freeEnergy.pressureHighT(T)

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return (T**3*(self.ts**2 + 4*self.b*self.lams)-self.ts*T*self.mussq)/self.lams

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return (3*T**2*(self.ts**2+4*self.b*self.lams)-self.ts*self.mussq)/self.lams

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
        return (T**3*(self.th**2 + 4*self.b*self.lamh)-self.th*T*self.muhsq)/self.lamh

    #Second T-derivative of the pressure in the broken phase
    def ddpBrok(self, T):
        return (3*T**2*(self.th**2+4*self.b*self.lamh)-self.th*self.muhsq)/self.lamh

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
