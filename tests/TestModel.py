# defines the toy xSM model, used in 2004.06995 and 2010.09744
# critical temperature is at T=1
class TestModel2Step():
    __test__ = False
    abrok = 0.2
    asym = 0.1
    musqq = 0.4

    def __init__(self, abrok, asym, musqq, Tn):
        self.aLowT = abrok
        self.aHighT = asym
        self.musq = musqq
        self.Tnucl = Tn
        self.Tc = 1

    #Pressure in high T phase
    def pHighT(self, T):
        return T**4. + (self.aLowT - self.aHighT + self.aHighT*T**2 - self.musq)**2-self.musq**2

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return 4*T**3. + 4. * self.aHighT * T *(self.aLowT - self.aHighT + self.aHighT * T**2-self.musq)

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return 12.*T**2. +8 * self.aHighT**2. * T**2. + 4.*self.aHighT*(self.aLowT-self.aHighT +self.aHighT*T**2-self.musq)

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
        return T**4. + (self.aLowT*T**2. - self.musq)**2. - self.musq**2.

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return 4.*T**3. + 4. * self.aLowT * T *(self.aLowT*T**2 - self.musq)

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return 12.*T**2. +8. * self.aLowT**2. * T**2. + 4.*self.aLowT*(self.aLowT*T**2.-self.musq)

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

#Defines the bag equation of state
#Note that a factor 1/3 a_+ Tc**4 has been scaled out
#The critical temperature is at Tc=1, which relates psi and the (rescaled) bag constant epsilon: eps = 1-psi
#The phase transition strength at temperature t is given by: \alpha(t) = 1/3.*(1-psi)(1/t)**4

class TestModelBag():
    __test__ = False

    def __init__(self, psi, Tn):
        self.psi = psi #number of degrees of freedom of the low T phase divided by the number of degrees of freedom in the high T phase
        self.eps = 1. - psi #this is the bag constant times 3 and divided by (the number of degrees of freedom of the high T phase times Tc^4)
        self.Tnucl = Tn
        self.Tc = 1

    #Pressure in high T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pHighT(self, T):
        return T**4. - self.eps

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return 4.*T**3.

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return 12.*T**2.

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
        return 1/3.


    #Pressure in the low T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pLowT(self, T):
        return self.psi*T**4.

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return 4.*self.psi*T**3.

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return 12.*self.psi*T**2.

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
        return 1/3.



class TestModelTemplate():
    __test__ = False

    def __init__(self, alN, psiN, cb2, cs2, Tn, Tc, wn=1):
        self.alN = alN # Strength parameter alpha_n of the phase transition at the nucleation temperature
        self.psiN = psiN # Enthalpy in the low T phase divided by the enthalpy in the high T phase (both evaluated at the nucleation temperature)
        self.cb2 = cb2
        self.cs2 = cs2
        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2

        self.Tnucl = Tn # Nucleation temperature
        self.Tc = Tc
        self.wn = wn # Enthalpy in the high T phase at the nucleation temperature
        self.ap = 3*wn/(self.mu*Tn**self.mu)
        self.am = 3*wn*psiN/(self.nu*Tn**self.nu)
        self.eps = 0
        self.eps = (self.pHighT(Tn)-self.pLowT(Tn)-cb2*(self.eHighT(Tn)-self.eLowT(Tn)-3*wn*alN))/(1+cb2)

    #Pressure in high T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pHighT(self, T):
        return self.ap*T**self.mu/3 - self.eps

    #T-derivative of the pressure in the high T phase
    def dpHighT(self, T):
        return self.mu*self.ap*T**(self.mu-1)/3

    #Second T-derivative of the pressure in the high T phase
    def ddpHighT(self, T):
        return self.mu*(self.mu-1)*self.ap*T**(self.mu-2)/3

    #Energy density in the high T phase
    def eHighT(self, T):
        return T*self.dpHighT(T) - self.pHighT(T)

    #T-derivative of the energy density in the high T phase
    def deHighT(self, T):
        return T*self.ddpHighT(T)

    #Enthalpy in the high T phase
    def wHighT(self,T):
        return T*self.dpHighT(T)

    #Sound speed squared in the high T phase
    def csqHighT(self,T):
        return self.cs2

    #Pressure in the low T phase -- but note that a factor 1/3 a+ Tc**4 has been scaled out
    def pLowT(self, T):
        return self.am*T**self.nu/3

    #T-derivative of the pressure in the low T phase
    def dpLowT(self, T):
        return self.nu*self.am*T**(self.nu-1)/3

    #Second T-derivative of the pressure in the low T phase
    def ddpLowT(self, T):
        return self.nu*(self.nu-1)*self.am*T**(self.nu-2)/3

    #Energy density in the low T phase
    def eLowT(self, T):
        return T*self.dpLowT(T) - self.pLowT(T)

    #T-derivative of the energy density in the low T phase
    def deLowT(self, T):
        return T*self.ddpLowT(T)

    #Enthalpy in the high T phase
    def wLowT(self,T):
        return T*self.dpLowT(T)

    #Sound speed squared in the low T phase
    def csqLowT(self,T):
        return self.cb2
