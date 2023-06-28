import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar


class HydroTemplateModel:
    """
    Class for solving the matching equations and computing vw in LTE by fitting to the template model, 
    where the speeds of sound are assumed to be constant. This generally offers a good approximation to realistic models, 
    while being much faster to treat.
    
    References
    ----------
    Wen-Yuan Ai, Benoit Laurent, and Jorinde van de Vis. 
    Model-independent bubble wall velocities in local thermal equilibrium.
    arXiv:2303.10171 (2023).
    """
    
    def __init__(self,model,Tnucl):
        r"""
        Initialize the HydroTemplateModel class. Computes :math:`\alpha_n,\ \Psi_n,\ c_s,\ c_b`.

        Parameters
        ----------
        model : Model
            An object of the class Model.
        Tnucl : double
            Nucleation temperature.

        Returns
        -------
        None.

        """
        self.model = model
        pSym,pBrok = model.pSym(Tnucl),model.pBrok(Tnucl)
        wSym,wBrok = model.wSym(Tnucl),model.wBrok(Tnucl)
        eSym,eBrok = wSym-pSym,wBrok-pBrok
        self.cb2 = model.csqBrok(Tnucl)
        self.cs2 = model.csqSym(Tnucl)
        self.alN = (eSym-eBrok-(pSym-pBrok)/self.cb2)/(3*wSym)
        self.psiN = wBrok/wSym
        self.cb = np.sqrt(self.cb2)
        self.cs = np.sqrt(self.cs2)
        self.wN = wSym
        self.pN = pSym
        self.Tnucl = Tnucl
        
        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2
        self.vJ = self.findJouguetVelocity()
        
    def findJouguetVelocity(self): 
        """
        Computes the Jouguet velocity.

        """
        return self.cb*(1+np.sqrt(3*self.alN*(1-self.cb2+3*self.cb2*self.alN)))/(1+3*self.cb2*self.alN)
    
    def get_vp(self,vm,al,branch=-1):
        r"""
        Solves the matching equation for :math: `v_+`.

        Parameters
        ----------
        vm : double
            Plasma velocity behind the wall :math: `v_-`.
        al : double
            alpha parameter in front of the wall :math: `\alpha_+`.
        branch : int, optional
            Select the branch of the matching equation's solution. Can either be 1 for detonation or -1 for deflagration/hybrid.
            The default is -1.

        Returns
        -------
        double
            :math: `v_+`.

        """
        disc = max(0, vm**4-2*self.cb2*vm**2*(1-6*al)+self.cb2**2*(1-12*vm**2*al*(1-3*al)))
        return 0.5*(self.cb2+vm**2+branch*np.sqrt(disc))/(vm+3*self.cb2*vm*al)
    
    def w_from_alpha(self,al):
        r"""
        Finds the enthlapy :math:`\omega_+` corresponding to :math:`\alpha_+`.

        Parameters
        ----------
        al : double
            alpha parameter in front of the wall :math: `\alpha_+`.

        Returns
        -------
        double
            :math: `\omega_+`.

        """
        return (abs((1-3*self.alN)*self.mu-self.nu)+1e-100)/(abs((1-3*al)*self.mu-self.nu)+1e-100)
    
    def __find_Tm(self,vm,vp,Tp):
        r"""
        Finds :math:`T_-` as a function of :math:`v_-,\ v_+,\ T_+`.
        """
        epsilon = (self.nu-self.mu+3*self.alN*self.mu)/(self.mu*self.nu)
        ap = 3/(self.mu*self.Tnucl**self.mu)
        am = 3/(self.mu*self.psiN*self.Tnucl**self.nu)
        return ((3*(vm+vp)*epsilon-ap*Tp**self.mu*(vm+vp-vp*self.mu))/(am*(vm*self.nu-vm-vp)))**(1/self.nu)
    
    def __eqWall(self,al,vm,branch=-1):
        """
        Matching equation at the bubble wall.
        """
        vp = self.get_vp(vm,al,branch)
        ga2m,ga2p= 1/(1-vm**2),1/(1-vp**2)
        psi = self.psiN*self.w_from_alpha(al)**(self.nu/self.mu-1)
        return vp*vm*al/(1-(self.nu-1)*vp*vm)-(1-3*al-(ga2p/ga2m)**(self.nu/2)*psi)/(3*self.nu)
    
    def solve_alpha(self,vw, constraint=True):
        r"""
        Finds the value of :math:`\alpha_+` that solves the matching equation at the wall.

        Parameters
        ----------
        vw : double
            Wall velocity at which to solve the matching equation.
        constraint : bool, optional
            If True, imposes :math:`v_+<\min(c_s^2/v_w,v_w)` on the solution. Otherwise, the 
            constraint :math:`v_+<v_-` is used instead. Default is True.

        Returns
        -------
        double
            Value of :math:`\alpha_+` that solves the matching equation.

        """
        vm = min(self.cb,vw)
        vp_max = min(self.cs2/vw,vw) if constraint else vm
        al_min = max((vm-vp_max)*(self.cb2-vm*vp_max)/(3*self.cb2*vm*(1-vp_max**2)),(self.mu-self.nu)/(3*self.mu))
        al_max = 1/3
        branch = -1
        if self.__eqWall(al_min,vm)*self.__eqWall(al_max,vm)>0:
            branch = 1
        sol = root_scalar(self.__eqWall,(vm,branch),bracket=(al_min,al_max),rtol=1e-6,xtol=1e-6)
        return sol.root
    
    def __dfdv(self,v,X):
        """
        Fluid equations in the shock wave.
        """
        xi,w = X
        mu_xiv = (xi-v)/(1-xi*v)
        dxidv = xi*(1-v*xi)*(mu_xiv**2/self.cs2-1)
        dxidv /= 2*v*(1-v**2)
        dwdv = w*(1+1/self.cs2)*mu_xiv/(1-v**2)
        return [dxidv,dwdv]
    
    def integrate_plasma(self,v0,vw,wp):
        """
        Integrates the fluid equations in the shock wave until it reaches the shock front.

        Parameters
        ----------
        v0 : double
            Plasma velocity just in front of the wall (in the frame of the bubble's center).
        vw : double
            Wall velocity.
        wp : double
            Enthalpy just in front of the wall.

        Returns
        -------
        Bunch object returned by the scipy function integrate.solve_ivp containing the solution of the fluid equations.

        """
        def event(v,X):
            xi,w = X
            return xi*(xi-v)/(1-xi*v) - self.cs2
        event.terminal = True
        sol = solve_ivp(self.__dfdv,(v0,1e-20),[vw,wp],events=event,rtol=1e-6,atol=1e-6)
        return sol
    
    def __shooting(self,vw,al):
        """
        Integrates through the shock wave and returns the residual of the matching equation at the shock front.
        """
        vm = min(self.cb,vw)
        vp = self.get_vp(vm, al)
        wp = self.w_from_alpha(al)
        if abs(vp*vw-self.cs2) < 1e-12: 
            # If the wall is already very close to the shock front, we do not integrate through the shock wave 
            # to avoid any error due to rounding error.
            vp_sw = vw
            vm_sw = vp
            wm_sw = wp
        else:
            sol = self.integrate_plasma((vw-vp)/(1-vw*vp), vw, wp)
            vp_sw = sol.y[0,-1]
            vm_sw = (vp_sw-sol.t[-1])/(1-vp_sw*sol.t[-1])
            wm_sw = sol.y[1,-1]
        return vp_sw/vm_sw - ((self.mu-1)*wm_sw+1)/((self.mu-1)+wm_sw)
    
    def findvwLTE(self):
        """
        Computes the wall velocity for a deflagration or hybrid solution.
        """
        func = lambda vw: self.__shooting(vw,self.solve_alpha(vw))
        if self.alN < (1-self.psiN)/3 or self.alN <= (self.mu-self.nu)/(3*self.mu):
            print('alN too small')
            return 0
        if self.alN > self.max_al(100) or func(self.vJ) < 0:
            print('alN too large')
            return 1
        sol = root_scalar(func,bracket=[1e-3,self.vJ],rtol=1e-6,xtol=1e-6)
        return sol.root
    
    def findMatching(self,vw):
        r"""
        Computes :math:`v_-,\ v_+,\ T_-,\ T_+` for a deflagration or hybrid solution when the wall velocity is vw.
        """
        if vw > self.vJ:
            return self.detonation_vAndT(vw)
        
        func = lambda al: self.__shooting(vw,al)
        
        vm = min(self.cb,vw)
        al_max = 1/3
        vp_max = min(self.cs2/vw,vw)
        al_min = max((vm-vp_max)*(self.cb2-vm*vp_max)/(3*self.cb2*vm*(1-vp_max**2)),(self.mu-self.nu)/(3*self.mu))
        sol = root_scalar(func,bracket=(al_min,al_max),rtol=1e-6,xtol=1e-6)
        wp = self.w_from_alpha(sol.root)
        vp = self.get_vp(vm, sol.root)
        Tp = self.Tnucl*wp**(1/self.mu)
        Tm = self.__find_Tm(vm, vp, Tp)
        return vp,vm,Tp,Tm
    
    def matchDeflagOrHybInitial(self,vw,vp):
        r"""
        Returns initial guess for the solver in the matchDeflagOrHyb function.
        """
        vm = min(vw,self.cb)
        al = None
        if vp is not None:
            al = ((vm-vp)*(self.cb2-vm*vp))/(3*self.cb2*vm*(1-vp**2))
        else:
            al = self.solve_alpha(vw, False)
            vp = self.get_vp(vm, al)
        wp = self.w_from_alpha(al)
        Tp = self.Tnucl*wp**(1/self.mu)
        Tm = self.__find_Tm(vm, vp, Tp)
        return [Tp, Tm]
    
    def findHydroBoundaries(self, vwTry):
        r"""
        Returns :math:`c_1, c_2, T_+, T_-` for a given wall velocity and nucleation temperature.
        """
        vp,vm,Tp,Tm = self.findMatching(vwTry)
        wSym = self.wN*(Tp/self.Tnucl)**self.mu
        pSym = self.pN+((Tp/self.Tnucl)**self.mu-1)*self.wN/self.mu
        c1 = wSym*vp/(1-vp**2)
        c2 = pSym+wSym*vp**2/(1-vp**2)
        return (c1, c2, Tp, Tm)
    
    def max_al(self,upper_limit=100):
        r"""
        Computes the highest value of :math:`\alpha_n` at which a hybrid solution can be found.

        Parameters
        ----------
        upper_limit : double, optional
            Largest value of :math:`\alpha_n` at which the solver will look. If the true value is above upper_limit,
            returns upper_limit. The default is 100.

        """
        vm = self.cb
        def func(alN):
            vw = self.vJ
            vp = self.cs2/vw
            ga2p,ga2m = 1/(1-vp**2),1/(1-vm**2)
            wp = (vp+vw-vw*self.mu)/(vp+vw-vp*self.mu)
            psi = self.psiN*wp**(self.nu/self.mu-1)
            al = (self.mu-self.nu)/(3*self.mu)+(alN-(self.mu-self.nu)/(3*self.mu))/wp
            return vp*vm*al/(1-(self.nu-1)*vp*vm)-(1-3*al-(ga2p/ga2m)**(self.nu/2)*psi)/(3*self.nu)
        if func(upper_limit) < 0:
            return upper_limit
        sol = root_scalar(func,bracket=((1-self.psiN)/3,upper_limit),rtol=1e-6,xtol=1e-6)
        return sol.root
    
    def detonation_vw(self):
        """
        Computes the wall velocity for a detonation solution.
        """
        def matching_eq(vw):
            A = vw**2+self.cb2*(1-3*self.alN*(1-vw**2))
            vm = (A+np.sqrt(A**2-4*vw**2*self.cb2))/(2*vw)
            ga2w,ga2m = 1/(1-vw**2),1/(1-vm**2)
            return vw*vm*self.alN/(1-(self.nu-1)*vw*vm)-(1-3*self.alN-(ga2w/ga2m)**(self.nu/2)*self.psiN)/(3*self.nu)
        if matching_eq(self.vJ+1e-10)*matching_eq(1-1e-10) > 0:
            print('No detonation solution')
            return 0
        sol = root_scalar(matching_eq,bracket=(self.vJ+1e-10,1-1e-10),rtol=1e-6,xtol=1e-6)
        return sol.root
    
    def detonation_vAndT(self,vw):
        r"""
        Computes :math:`v_-,\ v_+,\ T_-,\ T_+` for a detonation solution.
        """
        vp = vw
        X = vp**2+self.cb2*(1-3*(1-vp**2)*self.alN)
        vm = (X+np.sqrt(X**2-4*self.cb2*vp**2))/(2*vp)
        Tm = self.__find_Tm(vm, vp, self.Tnucl)
        return vp,vm,self.Tnucl,Tm
    
    
    
    
    