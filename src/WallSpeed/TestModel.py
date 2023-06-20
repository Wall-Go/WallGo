# defines the toy xSM model, used in 2004.06995 and 2010.09744 
# critical temperature is at T=1

class TestModel2Step():
    abrok = 0.2
    asym = 0.1
    musqq = 0.4

    def __init__(self, abrok, asym, musqq):
        self.aBrok = abrok
        self.aSym = asym
        self.musq = musqq

    #Pressure in symmetric phase
    def pSym(self, T):
        return T**4. + (self.aBrok - self.aSym + self.aSym*T**2 - self.musq)**2-self.musq**2

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return 4*T**3. + 4 * self.aSym * T *(self.aBrok - self.aSym + self.aSym * T**2-self.musq)

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return 12*T**2. +8 * self.aSym**2 * T**2 + 4*self.aSym*(self.aBrok-self.aSym +self.aSym*T**2-self.musq)

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
        return T**4. + (self.aBrok*T**2 - self.musq)**2 - self.musq**2 

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
        return 4*T**3. + 4 * self.aBrok * T *(self.aBrok*T**2 - self.musq)

    #Second T-derivative of the pressure in the broken phase 
    def ddpBrok(self, T):
        return 12*T**2. +8 * self.aBrok**2 * T**2 + 4*self.aBrok*(self.aBrok*T**2-self.musq)

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

    def Tc(self):
        return 1


class TestModelBag():

    
    def __init__(self, psi):
        self.psi = psi #number of degrees of freedom of the broken phase
        self.eps = 1. - psi #this is not the bag constant times 3 and divided by the number of degrees of freedom of the symmetric phase

    #Pressure in symmetric phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pSym(self, T):
        return T**4. - self.eps

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return 4*T**3. 

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return 12*T**2. 

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
        return 1/3.

    
    #Pressure in the broken phase
    def pBrok(self, T):
        return self.psi*T**4. 

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
        return 4*self.psi*T**3.

    #Second T-derivative of the pressure in the broken phase 
    def ddpBrok(self, T):
        return 12*self.psi*T**2.
    
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
        return 1/3.



    def Tc(self):
        return 1.

