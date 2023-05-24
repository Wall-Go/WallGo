import numpy

# defines the model 
# critical temperature is at T=1

class TestModel():
    abrok = 0.2
    asym = 0.1
    musqq = 0.4

    def __init__(self, abrok, asym, musqq):
        self.aBrok = abrok
        self.aSym = asym
        self.musq = musqq


    def pSym(self, T):
        return T**4. + (self.aBrok - self.aSym + self.aSym*T**2 - self.musq)**2-self.musq**2

    def dpSym(self, T):
        return 4*T**3. + 4 * self.aSym * T *(self.aBrok - self.aSym + self.aSym * T**2-self.musq)

    def ddpSym(self, T):
        return 12*T**2. +8 * self.aSym**2 * T**2 + 4*self.aSym*(self.aBrok-self.aSym +self.aSym*T**2-self.musq)

    def eSym(self, T):
        return T*self.dpSym(T) - self.pSym(T) 

    def deSym(self, T):
        return T*self.ddpSym(T)

    def csqSym(self,T):
        return self.dpSym(T)/self.deSym(T)

    def pBrok(self, T):
        return T**4. + (self.aBrok*T**2 - self.musq)**2 - self.musq**2 

    def dpBrok(self, T):
        return 4*T**3. + 4 * self.aBrok * T *(self.aBrok*T**2 - self.musq)

    def ddpBrok(self, T):
        return 12*T**2. +8 * self.aBrok**2 * T**2 + 4*self.aBrok*(self.aBrok*T**2-self.musq)

    def eBrok(self, T):
        return T*self.dpBrok(T) - self.pBrok(T)
    
    def deBrok(self, T):
        return T*self.ddpBrok(T)

    def csqBrok(self,T):
        return self.dpBrok(T)/self.deBrok(T)

