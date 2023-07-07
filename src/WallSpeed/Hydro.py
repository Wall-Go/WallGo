import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar,root, minimize_scalar
from scipy.integrate import solve_ivp
from .HydroTemplateModel import HydroTemplateModel
import matplotlib.pyplot as plt


class Hydro:
    def __init__(self, model, Tnucl, rtol=1e-6, atol=1e-6):
        self.model = model
        self.Tnucl = Tnucl
        self.rtol,self.atol = rtol,atol
        self.vJ = self.findJouguetVelocity()
        self.template = HydroTemplateModel(model, Tnucl, rtol=1e-6, atol=1e-6)
        
    def findJouguetVelocity(self):
        r"""
        Finds the Jouguet velocity for a thermal effective potential, defined by Model,
        using that the derivative of :math:`v_+` with respect to :math:`T_-` is zero at the Jouguet velocity.
        """
        pSym = self.model.pSym(self.Tnucl)
        eSym = self.model.eSym(self.Tnucl)
        def vpDerivNum(tm): # The numerator of the derivative of v+^2
            pBrok = self.model.pBrok(tm)
            eBrok = self.model.eBrok(tm)
            num1 = pSym - pBrok # First factor in the numerator of v+^2
            num2 = pSym + eBrok
            den1 = eSym - eBrok # First factor in the denominator of v+^2
            den2 = eSym + pBrok 
            dnum1 = - self.model.dpBrok(tm) # T-derivative of first factor wrt tm
            dnum2 = self.model.deBrok(tm)
            dden1 = - dnum2 # T-derivative of second factor wrt tm
            dden2 = - dnum1
            return(dnum1*num2*den1*den2 + num1*dnum2*den1*den2 - num1*num2*dden1*den2 - num1*num2*den1*dden2)
        
        # For detonations, Tm has a lower bound of Tn, but no upper bound.
        # We increase Tmax until we find a value that brackets our root.
        Tmin,Tmax = self.Tnucl,1.5*self.Tnucl
        bracket1,bracket2 = vpDerivNum(Tmin),vpDerivNum(Tmax)
        while bracket1*bracket2 > 0 and Tmax < 10*self.Tnucl:
            Tmin = Tmax
            bracket1 = bracket2
            Tmax *= 1.5
            bracket2 = vpDerivNum(Tmax)
        
        tmSol = None
        if bracket1*bracket2 <= 0: # If Tmin and Tmax bracket our root, use the 'brentq' method.
            tmSol = root_scalar(vpDerivNum,bracket =[Tmin, Tmax], method='brentq', xtol=self.atol, rtol=self.rtol).root 
        else: # If we cannot bracket the root, use the 'secant' method instead.
            tmSol = root_scalar(vpDerivNum, method='secant', x0=self.Tnucl, x1=1.5*Tmax, xtol=self.atol, rtol=self.rtol).root 
        vp = np.sqrt((pSym - self.model.pBrok(tmSol))*(pSym + self.model.eBrok(tmSol))/(eSym - self.model.eBrok(tmSol))/(eSym + self.model.pBrok(tmSol)))
        return(vp)
    
    def vpvmAndvpovm(self, Tp, Tm):
        r"""
        Returns :math:`v_+v_-` and :math:`v_+/v_-` as a function of :math:`T_+, T_-`.
        """
        
        pSym,pBrok = self.model.pSym(Tp),self.model.pBrok(Tm)
        eSym,eBrok = self.model.eSym(Tp),self.model.eBrok(Tm)
        return (pSym-pBrok)/(eSym-eBrok), (eBrok+pSym)/(eSym+pBrok)
    
    
    def matchDeton(self, vw, branch=1):
        r"""
        Returns :math:`v_+, v_-, T_+, T_-` for a detonation as a function of the wall velocity and `T_n`.
        """
        vp = vw
        Tp = self.Tnucl
        pSym,wSym = self.model.pSym(Tp),self.model.wSym(Tp)
        eSym = wSym - pSym
        
        def tmFromvpsq(tm):
            pBrok,wBrok = self.model.pBrok(tm),self.model.wBrok(tm)
            eBrok = wBrok - pBrok
            return vp**2*(eSym-eBrok) - (pSym-pBrok)*(eBrok+pSym)/(eSym+pBrok)
        
        Tmax = minimize_scalar(tmFromvpsq,bounds=[self.Tnucl,10*self.Tnucl],method='Bounded').x
        Tm = root_scalar(tmFromvpsq,bracket =[self.Tnucl, Tmax], method='brentq', xtol=self.atol, rtol=self.rtol).root
        vpvm,vpovm = self.vpvmAndvpovm(Tp, Tm)
        vm = np.sqrt(vpvm/vpovm)
        return (vp, vm, Tp, Tm)
    
    def matchDeflagOrHyb(self, vw, vp=None):
        r"""
        Returns :math:`v_+, v_-, T_+, T_-` for a deflagration or hybrid when the wall velocity is given.
        
        Parameters
        ----------
        vw : double
            Wall velocity.
        vp : double or None, optional
            Plasma velocity in front of the wall :math:`v_-`. If None, vp is determined from conservation of 
            entropy. Default is None.
        """
        
        vwMapping = None
        if vp is None:
            vwMapping = vw
        
        # Finds an initial guess for Tp and Tm using the template model and make sure it satisfies all 
        # the relevant bounds.
        try:
            Tpm0 = self.template.matchDeflagOrHybInitial(min(vw,self.template.vJ), vp)
        except:
            Tpm0 = [1.1*self.Tnucl,self.Tnucl]
        if (vwMapping is None) and (Tpm0[0] <= Tpm0[1]):
            Tpm0[0] = 1.01*Tpm0[1]
        if (vwMapping is not None) and (Tpm0[0] <= Tpm0[1] or Tpm0[0] > Tpm0[1]/np.sqrt(1-min(vw**2,self.model.csqBrok(Tpm0[1])))):
            Tpm0[0] = Tpm0[1]*(1+1/np.sqrt(1-min(vw**2,self.model.csqBrok(Tpm0[1]))))/2
        
        def match(XpXm):
            Tpm = self.__inverseMappingT(XpXm,vwMapping)
            vmsq = min(vw**2,self.model.csqBrok(Tpm[1]))
            if vp is None:
                vpsq = (Tpm[1]**2-Tpm[0]**2*(1-vmsq))/Tpm[1]**2
            else:
                vpsq = vp**2
            vpvm,vpovm = self.vpvmAndvpovm(Tpm[0],Tpm[1])
            eq1 = vpvm*vpovm-vpsq
            eq2 = vpvm/vpovm-vmsq
            
            # We multiply the equations by c to make sure the solver
            # do not explore arbitrarly small or large values of Tm and Tp.
            c = (2**2+(Tpm[0]/Tpm0[0])**2+(Tpm[1]/Tpm0[1])**2)*(2**2+(Tpm0[0]/Tpm[0])**2+(Tpm0[1]/Tpm[1])**2)
            return (eq1*c,eq2*c)
    
        # We map Tm and Tp, which satisfy 0<Tm<Tp (and Tp < Tm/sqrt(1-vm**2) if entropy is conserved), 
        # to the interval (-inf,inf) which is used by the solver.
        sol = root(match,self.__mappingT(Tpm0,vwMapping),method='hybr',options={'xtol':self.atol})
        self.success = sol.success
        [Tp,Tm] = self.__inverseMappingT(sol.x,vwMapping)
        vm = min(vw, np.sqrt(self.model.csqBrok(Tm)))
        if vp is None:
            vp = np.sqrt((Tm**2-Tp**2*(1-vm**2)))/Tm
        return vp, vm, Tp, Tm
    
    def gammasq(self, v):
        r"""
        Lorentz factor :math:`\gamma^2` corresponding to velocity :math:`v`
        """
        return 1./(1. - v*v)
    
    def mu(self, xi, v):
        """
        Lorentz-transformed velocity
        """
        return (xi - v)/(1. - xi*v)
    
    def shockDE(self, v, xiAndT):
        r"""
        Hydrodynamic equations for the self-similar coordinate :math:`\xi` and the fluid temperature :math:`T` in terms of the fluid velocity :math:`v`
        """
        xi, T = xiAndT
        eq1 = self.gammasq(v) * (1. - v*xi)*(self.mu(xi,v)**2/self.model.csqSym(T)-1.)*xi/2./v
        eq2 = self.model.wSym(T)/self.model.dpSym(T)*self.gammasq(v)*self.mu(xi,v)
        return [eq1,eq2]
        
    def solveHydroShock(self, vw, vp, Tp):
        r"""
        Solves the hydrodynamic equations in the shock for a given wall velocity and `v_+, T_+` and determines the position of the shock. Returns the nucleation temperature.
        """
        
        def shock(v, xiAndT):
            xi, T = xiAndT
            return self.mu(xi,v)*xi - self.model.csqSym(T)
        shock.terminal = True
        xi0T0 = [vw,Tp]
        vpcent = self.mu(vw,vp)
        if shock(vpcent,xi0T0) > 0:
            vm_sh = vp
            xi_sh = vw
            Tm_sh = Tp
        else:
            solshock = solve_ivp(self.shockDE, [vpcent,1e-8], xi0T0, events=shock, rtol=self.rtol, atol=0) #solve differential equation all the way from v = v+ to v = 0
            vm_sh = solshock.t[-1]
            xi_sh,Tm_sh = solshock.y[:,-1]
        
        def TiiShock(tn): #continuity of Tii
            return self.model.wSym(tn)*xi_sh/(1-xi_sh**2) - self.model.wSym(Tm_sh)*self.mu(xi_sh,vm_sh)*self.gammasq(self.mu(xi_sh,vm_sh))
        Tmin,Tmax = 0.9*self.Tnucl,Tm_sh
        bracket1,bracket2 = TiiShock(Tmin),TiiShock(Tmax)
        while bracket1*bracket2 > 0 and Tmin > self.Tnucl/10:
            Tmax = Tmin
            bracket2 = bracket1
            Tmin /= 1.5
            bracket1 = TiiShock(Tmin)
        
        if bracket1*bracket2 <= 0: #If Tmin and Tmax bracket our root, use the 'brentq' method.
            Tn = root_scalar(TiiShock, bracket=[Tmin, Tmax], method='brentq', xtol=self.atol, rtol=self.rtol)  
        else: #If we cannot bracket the root, use the 'secant' method instead.
            Tn = root_scalar(TiiShock, method='secant', x0=self.Tnucl, x1=Tm_sh, xtol=self.atol, rtol=self.rtol)
        
        return Tn.root
    
    def strongestShock(self, vw):
        r"""
        Returns the minimum temperature for which a shock can exist.
        For the strongest shock, :math:`v_+=0`, which yields `T_+,T_-`.
        The fluid equations in the shock are then solved to determine the strongest shock.
        """
        def vpnum(Tpm):
            return (self.model.eBrok(Tpm[1])+self.model.pSym(Tpm[0]),self.model.pSym(Tpm[0])-self.model.pBrok(Tpm[1]))
    
        Tp,Tm = np.abs(fsolve(vpnum,[0.2,0.2]))
        return self.solveHydroShock(vw,0,Tp)
    
    def findMatching(self, vwTry):
        r"""
        Returns :math:`v_+, v_-, T_+, T_-` as a function of the wall velocity and the nucleation temperature. For detonations, these follow directly from the function
        matchDeton, for deflagrations and hybrids, the code varies `v_+' until the temperature in front of the shock equals the nucleation temperature
        """
        if vwTry > self.vJ: # Detonation
            vp,vm,Tp,Tm = self.matchDeton(vwTry)
                
        else: # Hybrid or deflagration
            # Loop over v+ until the temperature in front of the shock matches the nucleation temperature
            vpmax = min(vwTry,self.model.csqSym(self.model.Tc())/vwTry)
            vpmin = 1e-5 # Minimum value of vpmin
            
            def func(vpTry):
                _,_,Tp,_ = self.matchDeflagOrHyb(vwTry,vpTry)
                return self.solveHydroShock(vwTry,vpTry,Tp)-self.Tnucl
            
            fmin,fmax = func(vpmin),func(vpmax)
            if fmin*fmax <= 0:
                sol = root_scalar(func, bracket=[vpmin,vpmax], xtol=self.atol, rtol=self.rtol)
            else:
                extremum = minimize_scalar(lambda x: np.sign(fmax)*func(x), bounds=[vpmin,vpmax], method='Bounded',tol=1e-3)
                if extremum.fun > 0:
                    return (None,None,None,None) # If no deflagration solution exists, returns None.
                sol = root_scalar(func, bracket=[vpmin,extremum.x], xtol=self.atol, rtol=self.rtol)
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vwTry,sol.root)
                        
        return (vp,vm,Tp,Tm)
    
    def findHydroBoundaries(self, vwTry):
        r"""
        Returns :math:`c_1, c_2, T_+, T_-` for a given wall velocity and nucleation temperature.
        """
        vp,vm,Tp,Tm = self.findMatching(vwTry)
        if vp is None:
            return (vp,vm,Tp,Tm)
        wSym = self.model.wSym(Tp)
        c1 = wSym*self.gammasq(vp)*vp
        c2 = self.model.pSym(Tp)+wSym*self.gammasq(vp)*vp**2
        return (c1, c2, Tp, Tm)
    
    def findvwLTE(self):
        r"""
        Returns the wall velocity in local thermal equilibrium for a given nucleation temperature.
        The wall velocity is determined by solving the matching condition :math:`T_+ \gamma_+= T_-\gamma_-`. 
        For small wall velocity :math:`T_+ \gamma_+> T_-\gamma_-`, and -- if a solution exists -- :math:`T_+ \gamma_+< T_-\gamma_-` for large wall velocity.
        If the phase transition is too weak for a solution to exist, returns 0. If it is too strong, returns 1.
        The solution is always a deflagration or hybrid.
        """
        def func(vw): # Function given to the root finder
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vw)
            Tntry = self.solveHydroShock(vw,vp,Tp)
            return Tntry - self.Tnucl
        def shock(vw): # Equation to find the position of the shock front. If shock(vw) < 0, the front is ahead of vw.
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vw)
            return vp*vw-self.model.csqSym(Tp)
        
        self.success = True
        vmin = 0.01
        vmax = self.vJ
        if shock(vmax) > 0: # Finds the maximum vw such that the shock front is ahead of the wall.
            vmax = root_scalar(shock,bracket=[self.model.csqSym(self.Tnucl)**0.5,self.vJ], xtol=self.atol, rtol=self.rtol).root-1e-6
        fmax = func(vmax)
        if fmax > 0 or not self.success: # There is no deflagration or hybrid solution, we return 1.
            return 1
        
        fmin = func(vmin)
        if fmin < 0: # vw is smaller than vmin, we return 0.
            return 0
        else:
            sol = root_scalar(func, bracket=(vmin,vmax), xtol=self.atol, rtol=self.rtol)
            return sol.root
        
    def __mappingT(self, TpTm, vw=None):
        """
        Maps the variables Tp and Tm, which are constrained by 0<Tm<Tp for deflagration/hybrid walls, and additionally 
        Tp < Tm/sqrt(1-vm**2) if entropy is conserved. 
        They are mapped to the interval (-inf,inf) to allow root finding algorithms to explore different values of (Tp,Tm),
        without going outside of the bounds above.
    
        Parameters
        ----------
        TpTm : array_like, shape (2,)
            List containing Tp and Tm.
        vw : double, optional
            Wall velocity. Must be provided only if entropy is conserved, in which case 
            0 < Tm < Tp < Tm/sqrt(1-vm**2). If None, only 0 < Tm < Tp is enforced.
            Default is None (entropy not conserved).
        """
        
        Tp,Tm = TpTm
        if vw is None: # Entropy is not conserved, so we only impose 0 < Tm < Tp.
            Xm = 0.5*(2*Tm-Tp)/np.sqrt(Tm*(Tp-Tm))
            Xp = Tp/self.Tnucl-1 if Tp > self.Tnucl else 1-self.Tnucl/Tp
            return [Xp,Xm]
        else: # Entropy is conserved, so we also impose Tp < Tm/sqrt(1-vm**2).
            vmsq = min(vw**2,self.model.csqBrok(Tm))
            Xm = Tm/self.Tnucl-1 if Tm > self.Tnucl else 1-self.Tnucl/Tm
            r = Tm*(1/np.sqrt(1-vmsq)-1)
            Xp = (0.5*r+Tm-Tp)/np.sqrt((Tp-Tm)*(r+Tm-Tp))
            return [Xp,Xm]
    
    def __inverseMappingT(self, XpXm, vw=None):
        """
        Inverse of __mappingT.
        """
        
        Xp,Xm = XpXm
        if vw is None:
            Tp = self.Tnucl*(1+Xp) if Xp > 0 else self.Tnucl/(1-Xp)
            Tm = 0.5*Tp*(1+Xm/np.sqrt(1+Xm**2))
            return [Tp,Tm]
        else:
            Tm = self.Tnucl*(Xm+1) if Xm > 0 else self.Tnucl/(1-Xm)
            vmsq = min(vw**2,self.model.csqBrok(Tm))
            r = Tm*(1/np.sqrt(1-vmsq)-1)
            Tp = Tm + 0.5*r*(1+Xp/np.sqrt(1+Xp**2))
            return [Tp,Tm]















