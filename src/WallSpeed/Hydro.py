import Model
import numpy as np
from scipy.optimize import fsolve

def FindJouguetVelocity(Model,Tnucl):
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


def vpovm(Model, Tp, Tm):
     return (Model.eBrok(Tm) + Model.pSym(Tp))/(Model.eSym(Tp)+Model.pBrok(Tm))

def vpvm(Model, Tp, Tm):
    return (Model.pSym(Tp)-Model.pBrok(Tm))/(Model.eSym(Tp)-Model.eBrok(Tm))

def FindMatching(Model,vwTry,Tnucl):
    """
    """
    vJouguet = FindJouguetVelocity(Model,Tnucl)
    if vwTry > vJouguet: #Detonation
        vp = vwTry
        Tp = Tnucl
        def tmFromvpsq(tm): #determine Tm from the expression for vp^2
            lhs = vp**2*(Model.eSym(Tp)+Model.pBrok(tm))*(Model.eSym(Tp)-Model.eBrok(tm))      
            rhs = (Model.eBrok(tm) + Model.pSym(Tp))*(Model.pSym(Tp)-Model.pBrok(tm))
            return lhs - rhs
        Tm = fsolve(tmFromvpsq,Tp*1.1)[0]
        vm = np.sqrt(vpvm(Model,Tp,Tm)/vpovm(Model,Tp,Tm))
            
    else: #Hybrid or deflagration
        #loop over v+ until the temperature in front of the shock matches the nucleation temperature
        dv = vwTry/50 #initial velocity step
        vptry = dv #initial value of vp
        vm = 1
 #       while dv < 10**-5: #adjust precision

        def MatchDeflag(Tpm):
            return (vpvm(Model,Tpm[0],Tpm[1])*vpovm(Model,Tpm[0],Tpm[1])-vptry**2,vpvm(Model,Tpm[0],Tpm[1])/vpovm(Model,Tpm[0],Tpm[1])-vwTry**2)

        def MatchHybrid(Tpm):
            return (vpvm(Model,Tpm[0],Tpm[1])*vpovm(Model,Tpm[0],Tpm[1])-vptry**2,vpvm(Model,Tpm[0],Tpm[1])/vpovm(Model,Tpm[0],Tpm[1])-Model.cb(Tpm[1])**2)
        
        try:
            Tp,Tm = fsolve(MatchDeflag,[0.5,0.5])
        except:
            deflagFail = True
            print(deflagFail)
        else:
            deflagFail = False
            if vwTry < Model.cb(Tm):
                vm = vwTry

        if deflagFail == True or vwTry > Model.cb(Tm):
            try:
                Tp,Tm = fsolve(MatchHybrid,[0.5,0.5])
            except:
                print('Cant find a hybrid or deflagration solution')
            else:
                vm = Model.cb(Tm)
            
    return (vptry,vm,Tp,Tm)
