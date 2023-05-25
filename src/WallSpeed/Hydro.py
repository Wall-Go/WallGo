import TestModel
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

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
    if tmSol < Tnucl:
        tmSol = fsolve(vpDerivNum,Tnucl*2.)[0] #replace by solution with constraint?
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
    r"""
    Returns :math:`v_+, v_-, T_+, T_-` for a deflagrtion or hybrid when the wall velocity and :math:`v_+` are given
    """
    def matchDeflag(Tpm):
        return (vpvm(model,Tpm[0],Tpm[1])*vpovm(model,Tpm[0],Tpm[1])-vp**2,vpvm(model,Tpm[0],Tpm[1])/vpovm(model,Tpm[0],Tpm[1])-vw**2)

    def matchHybrid(Tpm):
        return (vpvm(model,Tpm[0],Tpm[1])*vpovm(model,Tpm[0],Tpm[1])-vp**2,vpvm(model,Tpm[0],Tpm[1])/vpovm(model,Tpm[0],Tpm[1])-model.csqBrok(Tpm[1]))

    try:
        Tp,Tm = fsolve(matchDeflag,[0.5,0.5])
    except:
        deflagFail = True
    else:
        deflagFail = False
        if vw < np.sqrt(model.csqBrok(Tm)):
            vm = vw

    if deflagFail == True or vw > np.sqrt(model.csqBrok(Tm)):
        try:
            Tp,Tm = fsolve(matchHybrid,[0.5,0.5])
        except:
            print('Cant find a hybrid or deflagration solution')
        else:
            vm = np.sqrt(model.csqBrok(Tm))
    return vp, vm, Tp, Tm

def gammasq(v):
    """
    Lorentz factor :math:`\gamma^2` corresponding to velocity :math:`v`
    """
    return 1./(1. - v*v)

def mu(xi,v):
    """
    Lorentz-transformed velocity
    """
    return (xi - v)/(1. - xi*v)

def shockDE(xiAndT,v,model):
    """
    Hydrodynamic equations for the self-similar coordinate :math:`\xi` and the fluid temperature :math:`T` in terms of the fluid velocity :math:`v`
    """
    xi, T = xiAndT
    dxiAndTdv = [gammasq(v) * (1. - v*xi)*(mu(xi,v)*mu(xi,v)/model.csqSym(T)-1.)*xi/2./v,model.wSym(T)/model.dpSym(T)*gammasq(v)*mu(xi,v)]
    return dxiAndTdv
    
def solveHydroShock(model,vw,vp,Tp):
    """
    Solves the hydrodynamic equations in the shock for a given wall velocity and `v_+, T_+` and determines the position of the shock. Returns the nucleation temperature.
    """
    xi0T0 = [vw,Tp]
    vpcent = mu(vw,vp)
    maxindex = 1024
    vs = np.linspace(vpcent,0,maxindex)
    solshock = odeint(shockDE,xi0T0,vs,args=(model,)) #solve differential equation all the way from v = v+ to v = 0
    xisol = solshock[:,0]
    Tsol = solshock[:,1]
    #now need to determine the position of the shock, which is set by mu(xi,v)^2 xi = cs^2
    index = 0
    while mu(xisol[index],vs[index])*xisol[index] < model.csqSym(Tsol[index]) and index<maxindex-1:
        index +=1
    def TiiShock(tn): #continuity of Tii
        return model.wSym(tn)*xisol[index]/(1-xisol[index]**2) - model.wSym(Tsol[index])*mu(xisol[index],vs[index])*gammasq(mu(xisol[index],vs[index]))
    Tn = fsolve(TiiShock,Tp*0.9)[0]
    return Tn
    

def findMatching(model,vwTry,Tnucl):
    """
    Returns :math:`v_+, v_-, T_+, T_-` as a function of the wall velocity and the nucleation temperature. For detonations, these follow directly from the function
    matchDeton, for deflagrations and hybrids, the code varies `v_+' until the temperature in front of the shock equals the nucleation temperature
    """
    vJouguet = findJouguetVelocity(model,Tnucl)
    if vwTry > vJouguet: #Detonation
        vp,vm,Tp,Tm = matchDeton(model,vwTry,Tnucl)
            
    else: #Hybrid or deflagration
        #loop over v+ until the temperature in front of the shock matches the nucleation temperature
        vpmax = np.sqrt(model.csqSym(model.Tc()))
        vpmin = 0.01 #minimum value of vpmin
        vptry = (vpmax + vpmin)/2.
        TnTry = 0
        error = 10**-2 #adjust error here
        count = 0
        while np.abs(TnTry - Tnucl)/Tnucl > error and count <100:
            vp,vm,Tp,Tm = matchDeflagOrHyb(model,vwTry,vptry)
            Tntry = solveHydroShock(model,vwTry,vptry,Tp)

            if Tntry > Tnucl:
                vpmax = vptry
                vptry = (vpmax + vpmin)/2.
            else:
                vpmin = vptry
                vptry = (vpmax + vpmin)/2.
            count += 1
                    
    return (vp,vm,Tp,Tm)

def c1c2(model, vwTry, Tnucl):
    """
    Returns :math:`c_1, c_2` for a given wall velocity and nucleation temperature
    """
    vp,vm,Tp,Tm = findMatching(model, vwTry, Tnucl)
    c1 = model.wSym(Tp)*gammasq(vp)*vp
    c2 = model.pSym(Tp)+model.wSym(Tp)*gammasq(vp)*vp**2
    return (c1, c2)


