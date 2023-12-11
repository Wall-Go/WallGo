import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import root_scalar,root, minimize_scalar
from scipy.integrate import solve_ivp
from .Thermodynamics import Thermodynamics
from .HydroTemplateModel import HydroTemplateModel
from .helpers import gammaSq, boostVelocity


class Hydro:
    """
    Class for solving the hydrodynamic equations of the plasma,
    at distances far enough from the wall such that the wall can be treated as infinitesimally thin.

    NOTE: We use the conventions that the velocities are always positive, even in the wall frame (vp and vm).
    These conventions are consistent with the literature, e.g. with arxiv:1004.4187.
    These conventions differ from the conventions used in the EOM and Boltzmann part of the code.
    The conversion is made in findHydroBoundaries.
    """

    def __init__(self, thermodynamics, TminGuess=0, TmaxGuess=10, rtol=1e-6, atol=1e-6):
        """Initialisation

        Parameters
        ----------
        thermodynamics : class
        rtol :
        atol:

        Returns
        -------
        cls: Hydro
            An object of the Hydro class.

        """
        self.thermodynamics = thermodynamics
        self.Tnucl = thermodynamics.Tnucl
        self.Tc = thermodynamics.Tc
        self.TminGuess = TminGuess
        self.TmaxGuess = 5.*self.Tnucl
        self.rtol,self.atol = rtol,atol
        self.vJ = self.findJouguetVelocity()
        self.template = HydroTemplateModel(thermodynamics, rtol=1e-6, atol=1e-6)

    def findJouguetVelocity(self):
        r"""
        Finds the Jouguet velocity for a thermal effective potential, defined by thermodynamics, and at the model's nucleation temperature,
        using that the derivative of :math:`v_+` with respect to :math:`T_-` is zero at the Jouguet velocity.

        Returns
        -------
        vJ: double
            The value of the Jouguet velocity for this model.

        """
        pHighT = self.thermodynamics.pHighT(self.Tnucl)
        eHighT = self.thermodynamics.eHighT(self.Tnucl)
        def vpDerivNum(tm): # The numerator of the derivative of v+^2
            pLowT = self.thermodynamics.pLowT(tm)
            eLowT = self.thermodynamics.eLowT(tm)
            num1 = pHighT - pLowT # First factor in the numerator of v+^2
            num2 = pHighT + eLowT
            den1 = eHighT - eLowT # First factor in the denominator of v+^2
            den2 = eHighT + pLowT
            dnum1 = - self.thermodynamics.dpLowT(tm) # T-derivative of first factor wrt tm
            dnum2 = self.thermodynamics.deLowT(tm)
            dden1 = - dnum2 # T-derivative of second factor wrt tm
            dden2 = - dnum1
            return(dnum1*num2*den1*den2 + num1*dnum2*den1*den2 - num1*num2*dden1*den2 - num1*num2*den1*dden2)

        # For detonations, Tm has a lower bound of Tn, but no upper bound.
        # We increase Tmax until we find a value that brackets our root.
        Tmin,Tmax = self.Tnucl,self.TmaxGuess
        bracket1,bracket2 = vpDerivNum(Tmin),vpDerivNum(Tmax)
        while bracket1*bracket2 > 0 and Tmax < 10*self.Tnucl:
            Tmin = Tmax
            bracket1 = bracket2
            Tmax *= 1.5
            bracket2 = vpDerivNum(Tmax)

        tmSol = None
        if bracket1*bracket2 <= 0: # If Tmin and Tmax bracket our root, use the 'brentq' method.
            tmSol = root_scalar(vpDerivNum,bracket =[Tmin, Tmax], method='brentq', xtol=self.atol, rtol=self.rtol).root
        else: # If we cannot bracket the root, use the 'secant' method instead. This will call thermodynamics outside of its interpolation range?
            tmSol = root_scalar(vpDerivNum, method='secant', x0=self.Tnucl, x1=1.5*Tmax, xtol=self.atol, rtol=self.rtol).root

        vp = np.sqrt((pHighT - self.thermodynamics.pLowT(tmSol))*(pHighT + self.thermodynamics.eLowT(tmSol))/(eHighT - self.thermodynamics.eLowT(tmSol))/(eHighT + self.thermodynamics.pLowT(tmSol)))
        return(vp)

    def vpvmAndvpovm(self, Tp, Tm):
        r"""
        Finds :math:`v_+v_-` and :math:`v_+/v_-` as a function of :math:`T_+, T_-`, from the matching conditions.

        Parameters
        ----------
        Tp : double
            Plasma temperature right in front of the bubble wall
        Tm : double
            Plasma temperature right behind the bubble wall

        Returns
        -------
        vpvm, vpovm: double
            `v_+v_-` and :math:`v_+/v_-`
        """

        pHighT,pLowT = self.thermodynamics.pHighT(Tp),self.thermodynamics.pLowT(Tm)
        eHighT,eLowT = self.thermodynamics.eHighT(Tp),self.thermodynamics.eLowT(Tm)
        vpvm = (pHighT-pLowT)/(eHighT-eLowT) if eHighT != eLowT else (pHighT-pLowT)*1e50
        vpovm = (eLowT+pHighT)/(eHighT+pLowT)
        return vpvm,vpovm


    def matchDeton(self, vw, branch=1):
        r"""
        Solves the matching conditions for a detonation. In this case, :math:`v_w = v_+` and :math:`T_+ = T_n` and :math:`v_-,T_-` are found from the matching equations.

        Parameters
        ---------
        vw : double
            Wall velocity
        branch : int
            Don't think this is used, can we remove it?

        Returns
        -------
        vp,vm,Tp,Tm : double
            The value of the fluid velocities in the wall frame and the temperature right in front of the wall and right behind the wall.

        """
        vp = vw
        Tp = self.Tnucl
        pHighT,wHighT = self.thermodynamics.pHighT(Tp),self.thermodynamics.wHighT(Tp)
        eHighT = wHighT - pHighT

        def tmFromvpsq(tm):
            pLowT,wLowT = self.thermodynamics.pLowT(tm),self.thermodynamics.wLowT(tm)
            eLowT = wLowT - pLowT
            return vp**2*(eHighT-eLowT) - (pHighT-pLowT)*(eLowT+pHighT)/(eHighT+pLowT)

        Tmax = minimize_scalar(tmFromvpsq,bounds=[self.Tnucl,self.TmaxGuess],method='Bounded').x
        Tm = root_scalar(tmFromvpsq,bracket =[self.Tnucl, Tmax], method='brentq', xtol=self.atol, rtol=self.rtol).root
        vpvm,vpovm = self.vpvmAndvpovm(Tp, Tm)
        vm = np.sqrt(vpvm/vpovm)
        return (vp, vm, Tp, Tm)

    def matchDeflagOrHyb(self, vw, vp=None):
        r"""
        Obtains the matching parameters :math:`v_+, v_-, T_+, T_-` for a deflagration or hybrid by solving the matching relations.

        Parameters
        ----------
        vw : double
            Wall velocity.
        vp : double or None, optional
            Plasma velocity in front of the wall :math:`v_-`. If None, vp is determined from conservation of
            entropy. Default is None.

        Returns
        -------
        vp,vm,Tp,Tm : double
            The value of the fluid velocities in the wall frame and the temperature right in front of the wall and right behind the wall.

        """

        vwMapping = None
        if vp is None:
            vwMapping = vw

        # Finds an initial guess for Tp and Tm using the template model and make sure it satisfies all
        # the relevant bounds.
        try:
            Tpm0 = self.template.matchDeflagOrHybInitial(min(vw,self.template.vJ), vp)
        except:
            Tpm0 = [np.min([self.TmaxGuess,1.1*self.Tnucl]),self.Tnucl]
        if (vwMapping is None) and (Tpm0[0] <= Tpm0[1]):
            Tpm0[0] = 1.01*Tpm0[1]
        if (vwMapping is not None) and (Tpm0[0] <= Tpm0[1] or Tpm0[0] > Tpm0[1]/np.sqrt(1-min(vw**2,self.thermodynamics.csqLowT(Tpm0[1])))):
            Tpm0[0] = Tpm0[1]*(1+1/np.sqrt(1-min(vw**2,self.thermodynamics.csqLowT(Tpm0[1]))))/2

        def match(XpXm):
            Tpm = self.__inverseMappingT(XpXm,vwMapping)
            vmsq = min(vw**2,self.thermodynamics.csqLowT(Tpm[1]))
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
        self.success = sol.success or np.sum(sol.fun**2) < 1e-10 #If the error is small enough, we consider that root has converged even if it returns False.
        [Tp,Tm] = self.__inverseMappingT(sol.x,vwMapping)

        vmsq = min(vw**2, self.thermodynamics.csqLowT(Tm))
        vm = np.sqrt(max(vmsq, 0))
        if vp is None:
            vp = np.sqrt((Tm**2-Tp**2*(1-vm**2)))/Tm
        return vp, vm, Tp, Tm


    def shockDE(self, v, xiAndT):
        r"""
        Hydrodynamic equations for the self-similar coordinate :math:`\xi = r/t` and the fluid temperature :math:`T` in terms of the fluid velocity :math:`v`

        Parameters
        ----------
        v : array
            Fluid velocities.
        xiAndT : array
            Values of the self-similar coordinate :math:`\xi = r/t` and the temperature :math:`T`

        Returns
        -------
        eq1, eq2 : array
            The expressions for :math:`\frac{\partial v}{\partial \xi}` and :math:`\frac{\partial v}{\partial T}`
        """
        xi, T = xiAndT
        eq1 = gammaSq(v) * (1. - v*xi)*(boostVelocity(xi,v)**2/self.thermodynamics.csqHighT(T)-1.)*xi/2./v
        eq2 = self.thermodynamics.wHighT(T)/self.thermodynamics.dpHighT(T)*gammaSq(v)*boostVelocity(xi,v)
        return [eq1,eq2]

    def solveHydroShock(self, vw, vp, Tp):
        r"""
        Solves the hydrodynamic equations in the shock for a given wall velocity :math:`v_w` and matching parameters :math:`v_+,T_+`
        and returns the corresponding nucleation temperature :math:`T_n`, which is the temperature of the plasma in front of the shock.

        Parameters
        ----------
        vw : double
            Wall velocity
        vp : double
            Value of the fluid velocity in the wall frame, right in front of the bubble
        Tp : double
            Value of the fluid temperature right in front of the bubble

        Returns
        -------
        Tn : double
            Nucleation temperature

        """

        def shock(v, xiAndT):
            xi, T = xiAndT
            return boostVelocity(xi,v)*xi - self.thermodynamics.csqHighT(T)
        shock.terminal = True
        xi0T0 = [vw,Tp]
        vpcent = boostVelocity(vw,vp)
        if shock(vpcent,xi0T0) > 0:
            vm_sh = vpcent
            xi_sh = vw
            Tm_sh = Tp
        elif vw == vp:
            vm_sh = 0
            xi_sh = self.thermodynamics.csqHighT(Tp)**0.5
            Tm_sh = Tp
        else:
            solshock = solve_ivp(self.shockDE, [vpcent,1e-8], xi0T0, events=shock, rtol=self.rtol, atol=0) #solve differential equation all the way from v = v+ to v = 0
            vm_sh = solshock.t[-1]
            xi_sh,Tm_sh = solshock.y[:,-1]

        def TiiShock(tn): #continuity of Tii
            return self.thermodynamics.wHighT(tn)*xi_sh/(1-xi_sh**2) - self.thermodynamics.wHighT(Tm_sh)*boostVelocity(xi_sh,vm_sh)*gammaSq(boostVelocity(xi_sh,vm_sh))
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
        Finds the smallest nucleation temperature for which a shock can exist.
        For the strongest shock, the fluid is at rest in front of the bubble (in the wall frame), :math:`v_+=0`, which yields :math:`T_+,T_-`.
        The nucleation temperature corresponding to these matching conditions is obtained by solving the hydrodynamic equations in the shock.

        Parameters
        ----------
        vw : double
            The wall velocity

        Returns
        -------
        Tn : double
            The nucleation temperature with the strongest possible shock

        """
        def vpnum(Tpm):
            return (self.thermodynamics.eLowT(Tpm[1])+self.thermodynamics.pHighT(Tpm[0]),self.thermodynamics.pHighT(Tpm[0])-self.thermodynamics.pLowT(Tpm[1]))

        Tp,Tm = np.abs(fsolve(vpnum,[0.2,0.2]))
        return self.solveHydroShock(vw,0,Tp)

    def findMatching(self, vwTry):
        r"""
        Finds the matching parameters :math:`v_+, v_-, T_+, T_-` as a function of the wall velocity and for the nucleation temperature of the model.
        For detonations, these follow directly from :func:`matchDeton`,
        for deflagrations and hybrids, the code varies :math:`v_+` until the temperature in front of the shock equals the nucleation temperature

        Parameters
        ----------
        vwTry : double
            The value of the wall velocity

        Returns
        -------
        vp,vm,Tp,Tm : double
            The value of the fluid velocities in the wall frame and the temperature right in front of the wall and right behind the wall.

        """
        if vwTry > self.vJ: # Detonation
            vp,vm,Tp,Tm = self.matchDeton(vwTry)

        else: # Hybrid or deflagration
            # Loop over v+ until the temperature in front of the shock matches the nucleation temperature
            vpmax = min(vwTry,self.thermodynamics.csqHighT(self.Tc)/vwTry)
            vpmin = 1e-5 # Minimum value of vpmin

            def func(vpTry):
                _,_,Tp,_ = self.matchDeflagOrHyb(vwTry,vpTry)
                return self.solveHydroShock(vwTry,vpTry,Tp)-self.Tnucl

            fmin,fmax = func(vpmin),func(vpmax)
            if fmin*fmax <= 0:
                sol = root_scalar(func, bracket=[vpmin,vpmax], xtol=self.atol, rtol=self.rtol)
            else:
                extremum = minimize_scalar(lambda x: np.sign(fmax)*func(x), bounds=[vpmin,vpmax], method='Bounded')
                if extremum.fun > 0:
                    return self.template.findMatching(vwTry)
                sol = root_scalar(func, bracket=[vpmin,extremum.x], xtol=self.atol, rtol=self.rtol)
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vwTry,sol.root)

        return (vp,vm,Tp,Tm)


    def findHydroBoundaries(self, vwTry):
        r"""
        Finds the relevant boundary conditions (:math:`c_1,c_2,T_+,T_-` and the fluid velocity in right in front of the wall) for the scalar and plasma equations of motion for a given wall velocity and the model's nucletion temperature.

        NOTE: the sign of :math:`c_1` is chosen to match the convention for the fluid velocity used in EOM and
        Hydro. In those conventions, math:`v_+` would be negative, and therefore :math:`c_1` has to be negative as well.

        Parameters
        ----------
        vwTry : double
            The value of the wall velocity

        Returns
        -------
        c1,c2,Tp,Tm,vMid : double
            The boundary conditions for the scalar field and plasma equation of motion

        """
        vp,vm,Tp,Tm = self.findMatching(vwTry)
        if vp is None:
            # not sure what is going on here
            #return (vp,vm,Tp,Tm,boostVelocity(vwTry,vp))
            return (vp,vm,Tp,Tm,None)
        wHighT = self.thermodynamics.wHighT(Tp)
        c1 = -wHighT*gammaSq(vp)*vp
        c2 = self.thermodynamics.pHighT(Tp)+wHighT*gammaSq(vp)*vp**2
        vMid = -boostVelocity(0.5*(vm+vp), vwTry)  # minus sign for convention change
        return (c1, c2, Tp, Tm, vMid)


    def findvwLTE(self):
        r"""
        Returns the wall velocity in local thermal equilibrium for the model's nucleation temperature.
        The wall velocity is determined by solving the matching condition :math:`T_+ \gamma_+= T_-\gamma_-`.
        For small wall velocity :math:`T_+ \gamma_+> T_-\gamma_-`, and -- if a solution exists -- :math:`T_+ \gamma_+< T_-\gamma_-` for large wall velocity.
        If the phase transition is too weak for a solution to exist, returns 0. If it is too strong, returns 1.
        The solution is always a deflagration or hybrid.

        Parameters
        ----------

        Returns
        -------
        vwLTE
            The value of the wall velocity for this model in local thermal equilibrium.
        """
        def func(vw): # Function given to the root finder
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vw)
            Tntry = self.solveHydroShock(vw,vp,Tp)
            return Tntry - self.Tnucl
        def shock(vw): # Equation to find the position of the shock front. If shock(vw) < 0, the front is ahead of vw.
            vp,vm,Tp,Tm = self.matchDeflagOrHyb(vw)
            return vp*vw-self.thermodynamics.csqHighT(Tp)

        self.success = True
        vmin = 0.01
        vmax = self.vJ
        if shock(vmax) > 0: # Finds the maximum vw such that the shock front is ahead of the wall.
            vmax = root_scalar(shock,bracket=[self.thermodynamics.csqHighT(self.Tnucl)**0.5,self.vJ], xtol=self.atol, rtol=self.rtol).root-1e-6
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
#            Xp = Tp/self.Tnucl-1 if Tp > self.Tnucl else 1-self.Tnucl/(Tp-self.TminGuess)
            Xp = np.tan(np.pi/(self.TmaxGuess-self.TminGuess)*(Tp-(self.TmaxGuess+self.TminGuess)/2))
            return [Xp,Xm]
        else: # Entropy is conserved, so we also impose Tp < Tm/sqrt(1-vm**2).
            vmsq = min(vw**2,self.thermodynamics.csqLowT(Tm))
            Xm = Tm/self.Tnucl-1 if Tm > self.Tnucl else 1-self.Tnucl/Tm
            r = Tm*(1/np.sqrt(1-vmsq)-1)
            Xp = -(0.5*r+Tm-Tp)/np.sqrt((Tp-Tm)*(r+Tm-Tp))
            return [Xp,Xm]

    def __inverseMappingT(self, XpXm, vw=None):
        """
        Inverse of __mappingT.
        """

        Xp,Xm = XpXm
        if vw is None:
            Tp = np.arctan(Xp)*(self.TmaxGuess-self.TminGuess)/np.pi+ (self.TmaxGuess+ self.TminGuess)/2
            Tm = 0.5*Tp*(1+Xm/np.sqrt(1+Xm**2))
            return [Tp,Tm]
        else:
            Tm = self.Tnucl*(Xm+1) if Xm > 0 else self.Tnucl/(1-Xm)
            vmsq = min(vw**2,self.thermodynamics.csqLowT(Tm))
            r = Tm*(1/np.sqrt(1-vmsq)-1)
            Tp = Tm + 0.5*r*(1+Xp/np.sqrt(1+Xp**2))
            return [Tp,Tm]
