import Model
import numpy as np
from scipy.optimize import fsolve

def findJouguetVelocity(model,Tnucl):
    r"""
    Finds the Jouguet velocity for a thermal effective potential, defined by Model,
    using that the derivative of :math:`v_+` with respect to :math:`T_-` is zero at the Jouguet velocity
    """
    def vpDerivNum(tm): #the numerator of the derivative of v+^2
        num1 = model.pSym(Tnucl) - model.pBrok(tm) #first factor in the numerator of v+^2
        num2 = model.pSym(Tnucl) + model.eBrok(tm) 
        den1 = model.eSym(Tnucl) - model.eBrok(tm) #first factor in the denominator of v+^2
        den2 = model.eSym(Tnucl) + model.pBrok(tm) 
        dnum1 = - model.dpBrok(tm) #T-derivative of first factor wrt tm
        dnum2 = model.deBrok(tm)
        dden1 = - model.deBrok(tm) #T-derivative of second factor wrt tm
        dden2 = model.dpBrok(tm)
        return(dnum1*num2*den1*den2 + num1*dnum2*den1*den2 - num1*num2*dden1*den2 - num1*num2*den1*dden2)

    tmSol = fsolve(vpDerivNum,Tnucl*1.1)[0]
    vp = np.sqrt((model.pSym(Tnucl) - model.pBrok(tmSol))*(model.pSym(Tnucl) + model.eBrok(tmSol))/(model.eSym(Tnucl) - model.eBrok(tmSol))/(model.eSym(Tnucl) + model.pBrok(tmSol)))
    return(vp)


def vpovm(model, Tp, Tm):
    r"""
    Returns the ratio :math:`v_+/v_-` as a function of :math:`T_+, T_-`
    """
    return (model.eBrok(Tm) + model.pSym(Tp))/(model.eSym(Tp)+model.pBrok(Tm))

def vpvm(model, Tp, Tm):
    r"""
    Returns the product :math:`v_+v_-` as a function of :math:`T_+, T_-`
    """
    return (model.pSym(Tp)-model.pBrok(Tm))/(model.eSym(Tp)-model.eBrok(Tm))


def matchDeton(model,vw,Tnucl):
    r"""
    Returns :math:`v_+, v_-, T_+, T_-` for a detonation as a function of the wall velocity and `T_n`.
    """
    vp = vw
    Tp = Tnucl
    def tmFromvpsq(tm): #determine Tm from the expression for vp^2
        lhs = vp**2*(model.eSym(Tp)+model.pBrok(tm))*(model.eSym(Tp)-model.eBrok(tm))      
        rhs = (model.eBrok(tm) + model.pSym(Tp))*(model.pSym(Tp)-model.pBrok(tm))
        return lhs - rhs
    Tm = fsolve(tmFromvpsq,Tp*1.1)[0]
    vm = np.sqrt(vpvm(model,Tp,Tm)/vpovm(model,Tp,Tm))
    return (vp, vm, Tp, Tm)

def matchDeflagOrHyb(model,vw,vp):
    def matchDeflag(Tpm):
        return (vpvm(model,Tpm[0],Tpm[1])*vpovm(model,Tpm[0],Tpm[1])-vp**2,vpvm(model,Tpm[0],Tpm[1])/vpovm(model,Tpm[0],Tpm[1])-vw**2)

    def matchHybrid(Tpm):
        return (vpvm(model,Tpm[0],Tpm[1])*vpovm(model,Tpm[0],Tpm[1])-vp**2,vpvm(model,Tpm[0],Tpm[1])/vpovm(model,Tpm[0],Tpm[1])-model.cb(Tpm[1])**2)

    try:
        Tp,Tm = fsolve(matchDeflag,[0.5,0.5])
    except:
        deflagFail = True
    else:
        deflagFail = False
        if vw < model.cb(Tm):
            vm = vw

    if deflagFail == True or vw > model.cb(Tm):
        try:
            Tp,Tm = fsolve(matchHybrid,[0.5,0.5])
        except:
            print('Cant find a hybrid or deflagration solution')
        else:
            vm = model.cb(Tm)
    return (vp, vm, Tp, Tm)


def solveHydroShock(vp,Tp):
    

def findMatching(model,vwTry,Tnucl):
    """
    """
    vJouguet = findJouguetVelocity(model,Tnucl)
    if vwTry > vJouguet: #Detonation
        vp,vm,Tp,Tm = matchDeton(model,vwTry,Tnucl)
            
    else: #Hybrid or deflagration
        #loop over v+ until the temperature in front of the shock matches the nucleation temperature
        vpmax = model.cs(model.Tc())
        vpmin = 0.01 #minimum value of vpmin!
        vptry = (vpmax + vpmin)/2
        TnTry = 0
        error = 10**-2 #adjust error here
        while(np.abs(TnTry - Tnucl)/Tnucl > error):
            vp,vm,Tp,Tm = matchDeflagOrHyb(model,vw,vptry)

            Tntry = solveHydroShock(vp,Tp)            

            if Tntry < Tnucl:
                vpmax = vptry
                vptry = (vpmax + vpmin)/2
            else:
                vpmin = vptry
                vptry = (vpmax + vpmin)/2
                    
    return (vp,vm,Tp,Tm)


