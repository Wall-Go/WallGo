# defines the toy xSM model, used in 2004.06995 and 2010.09744
# critical temperature is at T=1
class TestModel2Step():
    __test__ = False
    abrok = 0.2
    asym = 0.1
    musqq = 0.4

    def __init__(self, abrok, asym, musqq, Tn):
        self.aBrok = abrok
        self.aSym = asym
        self.musq = musqq
        self.Tnucl = Tn
        self.Tc = 1

    #Pressure in symmetric phase
    def pSym(self, T):
        return T**4. + (self.aBrok - self.aSym + self.aSym*T**2 - self.musq)**2-self.musq**2

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return 4*T**3. + 4. * self.aSym * T *(self.aBrok - self.aSym + self.aSym * T**2-self.musq)

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return 12.*T**2. +8 * self.aSym**2. * T**2. + 4.*self.aSym*(self.aBrok-self.aSym +self.aSym*T**2-self.musq)

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
        return T**4. + (self.aBrok*T**2. - self.musq)**2. - self.musq**2.

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
        return 4.*T**3. + 4. * self.aBrok * T *(self.aBrok*T**2 - self.musq)

    #Second T-derivative of the pressure in the broken phase
    def ddpBrok(self, T):
        return 12.*T**2. +8. * self.aBrok**2. * T**2. + 4.*self.aBrok*(self.aBrok*T**2.-self.musq)

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

#Defines the bag equation of state
#Note that a factor 1/3 a_+ Tc**4 has been scaled out
#The critical temperature is at Tc=1, which relates psi and the (rescaled) bag constant epsilon: eps = 1-psi
#The phase transition strength at temperature t is given by: \alpha(t) = 1/3.*(1-psi)(1/t)**4

class TestModelBag():
    __test__ = False

    def __init__(self, psi, Tn):
        self.psi = psi #number of degrees of freedom of the broken phase divided by the number of degrees of freedom in the symmetric phase
        self.eps = 1. - psi #this is the bag constant times 3 and divided by (the number of degrees of freedom of the symmetric phase times Tc^4)
        self.Tnucl = Tn
        self.Tc = 1

    #Pressure in symmetric phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pSym(self, T):
        return T**4. - self.eps

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return 4.*T**3.

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return 12.*T**2.

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


    #Pressure in the broken phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pBrok(self, T):
        return self.psi*T**4.

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
        return 4.*self.psi*T**3.

    #Second T-derivative of the pressure in the broken phase
    def ddpBrok(self, T):
        return 12.*self.psi*T**2.

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



class TestModelTemplate():
    __test__ = False

    def __init__(self, alN, psiN, cb2, cs2, Tn, Tc, wn=1):
        self.alN = alN # Strength parameter alpha_n of the phase transition at the nucleation temperature
        self.psiN = psiN # Enthalpy in the broken phase divided by the enthalpy in the symmetric phase (both evaluated at the nucleation temperature)
        self.cb2 = cb2
        self.cs2 = cs2
        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2

        self.Tnucl = Tn # Nucleation temperature
        self.Tc = Tc
        self.wn = wn # Enthalpy in the symmetric phase at the nucleation temperature
        self.ap = 3*wn/(self.mu*Tn**self.mu)
        self.am = 3*wn*psiN/(self.nu*Tn**self.nu)
        self.eps = 0
        self.eps = (self.pSym(Tn)-self.pBrok(Tn)-cb2*(self.eSym(Tn)-self.eBrok(Tn)-3*wn*alN))/(1+cb2)

    #Pressure in symmetric phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pSym(self, T):
        return self.ap*T**self.mu/3 - self.eps

    #T-derivative of the pressure in the symmetric phase
    def dpSym(self, T):
        return self.mu*self.ap*T**(self.mu-1)/3

    #Second T-derivative of the pressure in the symmetric phase
    def ddpSym(self, T):
        return self.mu*(self.mu-1)*self.ap*T**(self.mu-2)/3

    #Energy density in the symmetric phase
    def eSym(self, T):
        return T*self.dpSym(T) - self.pSym(T)

    #T-derivative of the energy density in the symmetric phase
    def deSym(self, T):
        return T*self.ddpSym(T)

    #Enthalpy in the symmetric phase
    def wSym(self,T):
        return T*self.dpSym(T)

    #Sound speed squared in the symmetric phase
    def csqSym(self,T):
        return self.cs2

    #Pressure in the broken phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pBrok(self, T):
        return self.am*T**self.nu/3

    #T-derivative of the pressure in the broken phase
    def dpBrok(self, T):
        return self.nu*self.am*T**(self.nu-1)/3

    #Second T-derivative of the pressure in the broken phase
    def ddpBrok(self, T):
        return self.nu*(self.nu-1)*self.am*T**(self.nu-2)/3

    #Energy density in the broken phase
    def eBrok(self, T):
        return T*self.dpBrok(T) - self.pBrok(T)

    #T-derivative of the energy density in the broken phase
    def deBrok(self, T):
        return T*self.ddpBrok(T)

    #Enthalpy in the symmetric phase
    def wBrok(self,T):
        return T*self.dpBrok(T)

    #Sound speed squared in the broken phase
    def csqBrok(self,T):
        return self.cb2
