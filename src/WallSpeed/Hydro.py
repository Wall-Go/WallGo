import Model
import numpy as np
from scipy.optimize import fsolve

def findJouguetVelocity(Model,Tnucl):
        """
        Finds the Jouguet velocity for a thermal effective potential, defined by Model,
        using that the derivative of v+ wrt T- is zero at the Jouguet velocity
        """
        def vpDerivNum(tm): #the numerator of the derivative of v+^2
            num1 = Model.pSym(Tnucl) - Model.pBrok(tm) #first factor in the numerator of v+^2
            num2 = Model.pSym(Tnucl) + Model.eBrok(tm) 
            den1 = Model.eSym(Tnucl) - Model.eBrok(tm) #first factor in the denominator of v+^2
            den2 = Model.eSym(Tnucl) + Model.pBrok(tm) 
            dnum1 = - Model.dpBrok(tm) #T-derivative of first factor wrt tm
            dnum2 = Model.deBrok(tm)
            dden1 = - Model.deBrok(tm) #T-derivative of second factor wrt tm
            dden2 = Model.dpBrok(tm)
            return(dnum1*num2*den1*den2 + num1*dnum2*den1*den2 - num1*num2*dden1*den2 - num1*num2*den1*dden2)

        tmSol = fsolve(vpDerivNum,Tnucl*1.1)[0]
        vp = np.sqrt((Model.pSym(Tnucl) - Model.pBrok(tmSol))*(Model.pSym(Tnucl) + Model.eBrok(tmSol))/(Model.eSym(Tnucl) - Model.eBrok(tmSol))/(Model.eSym(Tnucl) + Model.pBrok(tmSol)))
        return(vp)

def findMatching(Model,vwTry,Tnucl):
    """
    """
    vJouguet = findJouguetVelocity(Model,Tnucl)
    if vwTry > vJouguet: #Detonation
        vp = vwTry
        Tp = Tnucl
        def tmFromvpsq(tm): #determine Tm from the expression for vp^2
            lhs = vp**2*(Model.eSym(Tp)+Model.pBrok(tm))*(Model.eSym(Tp)-Model.eBrok(tm))      
            rhs = (Model.eBrok(tm) + Model.pSym(Tp))*(Model.pSym(Tp)-Model.pBrok(tm))
            return lhs - rhs
        Tm = fsolve(tmFromvpsq,Tp*1.1)[0]
        vm = np.sqrt((Model.pSym(Tp)-Model.pBrok(Tm))*(Model.eSym(Tp)+Model.pBrok(Tm))/(Model.eSym(Tp)-Model.eBrok(Tm))/(Model.eBrok(Tm) + Model.pSym(Tp)) )
            
    else: #Hybrid or deflagration
        #loop over v+ until the temperature in front of the shock matches the nucleation temperature
        dv = vwTry/50 #initial velocity step
        vptry = dv #initial value of vp
        while dv < 

        
    return (vp,vm,Tp,Tm)
