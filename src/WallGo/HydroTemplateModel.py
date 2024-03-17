import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar,minimize_scalar
from .helpers import gammaSq

from .WallGoExceptions import WallGoError

class HydroTemplateModel:
    """
    Class for solving the matching equations and computing vw in LTE by fitting to the template model,
    where the speeds of sound are assumed to be constant. This generally offers a good approximation to realistic models,
    while being much faster to treat.

    References
    ----------
    Felix Giese, Thomas Konstandin, Kai Schmitz and Jorinde van de Vis
    Model-independent energy budget for LISA
    arXiv:2010.09744 (2020)

    Wen-Yuan Ai, Benoit Laurent, and Jorinde van de Vis.
    Model-independent bubble wall velocities in local thermal equilibrium.
    arXiv:2303.10171 (2023).

    NOTE: We use the conventions that the velocities are always positive, even in the wall frame (vp and vm).
    These conventions are consistent with the literature, e.g. with arxiv:1004.4187.
    These conventions differ from the conventions used in the EOM and Boltzmann part of the code.
    The conversion is made in findHydroBoundaries.

    """

    def __init__(self,thermodynamics,rtol=1e-6,atol=1e-6):
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
        self.thermodynamics = thermodynamics
        self.rtol,self.atol = rtol,atol
        pHighT,pLowT = thermodynamics.pHighT(thermodynamics.Tnucl),thermodynamics.pLowT(thermodynamics.Tnucl)
        wHighT,wLowT = thermodynamics.wHighT(thermodynamics.Tnucl),thermodynamics.wLowT(thermodynamics.Tnucl)
        eHighT,eLowT = wHighT-pHighT,wLowT-pLowT

        ## Calculate sound speed squared in both phases, needs to be > 0
        self.cb2 = thermodynamics.csqLowT(thermodynamics.Tnucl)
        self.cs2 = thermodynamics.csqHighT(thermodynamics.Tnucl)

        if (self.cb2 < 0 or self.cs2 < 0):
            raise WallGoError("Invalid sound speed at nucleation temperature",
                              data = {"csqLowT" : self.cb2, "csqHighT" : self.cs2})


        self.alN = (eHighT-eLowT-(pHighT-pLowT)/self.cb2)/(3*wHighT)
        self.psiN = wLowT/wHighT
        self.cb = np.sqrt(self.cb2)
        self.cs = np.sqrt(self.cs2)
        self.wN = wHighT
        self.pN = pHighT
        self.Tnucl = thermodynamics.Tnucl

        self.nu = 1+1/self.cb2
        self.mu = 1+1/self.cs2
        self.vJ = self.findJouguetVelocity()
        self.vMin = self.minVelocity()
        self.vMax = self.maxVelocity()

    def findJouguetVelocity(self, alN=None):
        r"""
        Finds the Jouguet velocity, corresponding to the phase transition strength :math:`\alpha_n`,
        using :math:`v_J = c_b \frac{1 + \sqrt{3 \alpha_n(1 - c_b^2 + 3 c_b^2 \alpha_n)}}{1+ 3 c_b^2 \alpha_n}`
        (eq. (25) of arXiv:2303.10171).

        Parameters
        ----------
        alN : double
            phase transition strength at the nucleation temperature, :math:`\alpha_n`.
            If :math:`\alpha_n` is not specified, the value defined by the model is used.

        Returns
        -------
        vJ: double
            The value of the Jouguet velocity.

        """

        if alN is None:
            alN = self.alN
        return self.cb*(1+np.sqrt(3*alN*(1-self.cb2+3*self.cb2*alN)))/(1+3*self.cb2*alN)
        
    
    def minVelocity(self):
        r"""
        Finds the minimum velocity that is possible for a given nucleation temeperature. 
        It is found by shooting in vp with :math:`\alpha_+ = 1/3` at the wall. This is the maximum value of :math:`\alpha_+` possible.
        The wall velocity which yields :math:`\alpha_+ = 1/3` for a given :math:`\alpha_N` is the minimum possible wall velocity.

        It is possible that no solution can be found, in this case there is no minimum value of the wall velocity
        and the function returns zero.

        Parameters
        ----------

        Returns
            vmin: double
                The minimum value of the wall velocity for which a solution can be found
        """
        def shootingalphamax(vw):
            vm = min(self.cb,vw)
            vp = self.get_vp(vm, 1/3.)
            return self.__shooting(vw,vp)

        try:
            return root_scalar(shootingalphamax,bracket=(1e-6,self.vJ),rtol=self.rtol,xtol=self.atol).root
        except:
            return 0
        

    def maxVelocity(self):
        r"""
        Finds the maximum velocity that is possible for a given nucleation temeperature. 
        It is found by use of some function that is also used below, and I don't know where it 
        came from. 
        TODO: figure out where that came from!

        Parameters
        ----------

        Returns
            vmax: double
                The maximum value of the wall velocity for which a solution can be found
        """
        def minalpha(vw):
            vm = min(vw,self.cb)
            vp_max = min(self.cs2,vw)            
            return max((vm-vp_max)*(self.cb2-vm*vp_max)/(3*self.cb2*vm*(1-vp_max**2)),(self.mu-self.nu)/(3*self.mu))-self.alN

        try:
            return root_scalar(minalpha,bracket=(1e-2,self.vJ),rtol=self.rtol,xtol=self.atol).root

        except:
            return self.vJ

    def get_vp(self,vm,al,branch=-1):
        r"""
        Solves the matching equation for :math:`v_+`.

        Parameters
        ----------
        vm : double
            Plasma velocity in the wall frame right behind the wall :math:`v_-`.
        al : double
            phase transition strength at the temperature right in front of the wall :math:`\alpha_+`.
        branch : int, optional
            Select the branch of the matching equation's solution. Can either be 1 for detonation or -1 for deflagration/hybrid.
            The default is -1.

        Returns
        -------
        vp : double
        double
            Plasma velocity in the wall frame right in front of the the wall :math:`v_+`.

        """
        disc = max(0, vm**4-2*self.cb2*vm**2*(1-6*al)+self.cb2**2*(1-12*vm**2*al*(1-3*al)))
        return 0.5*(self.cb2+vm**2+branch*np.sqrt(disc))/(vm+3*self.cb2*vm*al)

    def w_from_alpha(self,al):
        r"""
        Finds the enthlapy :math:`\omega_+` corresponding to :math:`\alpha_+` using the equation of state of the template model.

        Parameters
        ----------
        al : double
            alpha parameter at the temperature :math:`T_+` in front of the wall :math:`\alpha_+`.

        Returns
        -------
        double
            :math:`\omega_+`.

        """
        return (abs((1-3*self.alN)*self.mu-self.nu)+1e-100)/(abs((1-3*al)*self.mu-self.nu)+1e-100)

    def __find_Tm(self,vm,vp,Tp):
        r"""
        Finds :math:`T_-` as a function of :math:`v_-,\ v_+,\ T_+` using the matching equations.

        Parameters
        ----------
        vm : double
            Value of the fluid velocity in the wall frame, right behind the bubble wall
        vp : double
            Value of the fluid velocity in the wall frame, right in front of the bubble
        Tp : double
            Plasma temperature right in front of the bubble wall

        Returns
        -------
        Tm : double
            Plasma temperature right behind the bubble wall
        """
        ap = 3/(self.mu*self.Tnucl**self.mu)
        am = 3*self.psiN/(self.nu*self.Tnucl**self.nu)
        return ((ap*vp*self.mu*(1-vm**2)*Tp**self.mu)/(am*vm*self.nu*(1-vp**2)))**(1/self.nu)

    def __eqWall(self,al,vm,branch=-1):
        """
        Matching equation at the bubble wall.

        Parameters
        ----------
        al : double
            phase transition strength at the temperature right in front of the wall :math:`\alpha_+`.
        vm : double
            Value of the fluid velocity in the wall frame, right behind the bubble wall
        """
        vp = self.get_vp(vm,al,branch)
        psi = self.psiN*self.w_from_alpha(al)**(self.nu/self.mu-1)
        return vp*vm*al/(1-(self.nu-1)*vp*vm)-(1-3*al-(gammaSq(vp)/gammaSq(vm))**(self.nu/2)*psi)/(3*self.nu)

    def solve_alpha(self,vw, constraint=True):
        r"""
        Finds the value of :math:`\alpha_+` that solves the matching equation at the wall by varying :math:`v_-`.

        Parameters
        ----------
        vw : double
            Wall velocity at which to solve the matching equation.
        constraint : bool, optional
            If True, imposes :math:`v_+<\min(c_s^2/v_w,v_w)` on the solution. Otherwise, the
            constraint :math:`v_+<v_-` is used instead. Default is True.

        Returns
        -------
        alp : double
            Value of :math:`\alpha_+` that solves the matching equation.

        """
        vm = min(self.cb,vw)
        vp_max = min(self.cs2/vw,vw) if constraint else vm
        al_min = max((vm-vp_max)*(self.cb2-vm*vp_max)/(3*self.cb2*vm*(1-vp_max**2)),(self.mu-self.nu)/(3*self.mu),1e-10)
        al_max = 1/3
        branch = -1
        if self.__eqWall(al_min,vm)*self.__eqWall(al_max,vm)>0:
            branch = 1
        sol = root_scalar(self.__eqWall,(vm,branch),bracket=(al_min,al_max),rtol=self.rtol,xtol=self.atol)
        return sol.root

    def __dfdv(self,v,X):
        """
        Fluid equations in the shock wave.
        """
        xi,w = X
        mu_xiv = (xi-v)/(1-xi*v)
        dwdv = w*(1+1/self.cs2)*mu_xiv/(1-v**2)
        dxidv = xi*(1-v*xi)*(mu_xiv**2/self.cs2-1)/(2*v*(1-v**2)) if v != 0  else 1e50 #If v = 0, dxidv is set to a very high value
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
            return (xi*(xi-v)/(1-xi*v) - self.cs2)*v
        event.terminal = True
        sol = solve_ivp(self.__dfdv,(v0,1e-10),[vw,wp],events=event,rtol=self.rtol,atol=0)
        return sol
    
    def __shooting(self,vw,vp):
        """
        Integrates through the shock wave and returns the residual of the matching equation at the shock front.
        """
        vm = min(self.cb,vw)
        al = (vp/vm-1.)*(vp*vm/self.cb2 - 1.)/(1-vp**2)/3.
        wp = self.w_from_alpha(al)
        if abs(vp*vw-self.cs2) < 1e-12:
            # If the wall is already very close to the shock front, we do not integrate through the shock wave
            # to avoid any error due to rounding error.
            vp_sw = vw
            vm_sw = vp
            wm_sw = wp
        elif vw == vp:
            # If the plasma is at rest in front of the wall, there is no variation of plasma velocity and temperature in the shock wave
            vp_sw = self.cs
            vm_sw = self.cs
            wm_sw = wp
        else:
            self.temp = [vw,vp,(vw-vp)/(1-vw*vp),wp]
            sol = self.integrate_plasma((vw-vp)/(1-vw*vp), vw, wp)
            vp_sw = sol.y[0,-1]
            vm_sw = (vp_sw-sol.t[-1])/(1-vp_sw*sol.t[-1])
            wm_sw = sol.y[1,-1]
        return vp_sw/vm_sw - ((self.mu-1)*wm_sw+1)/((self.mu-1)+wm_sw)


    def findvwLTE(self):
        """
        Computes the wall velocity for a deflagration or hybrid solution.
        TODO: Explain the logic

        Parameters
        ----------
        

        Returns
        -------
            vwLTE : double
        """

        def shootingInLTE(vw):
            vm = min(self.cb,vw)
            al = self.solve_alpha(vw)
            vp = self.get_vp(vm, al)
            return self.__shooting(vw, vp)

        if self.alN < (1-self.psiN)/3 or self.alN <= (self.mu-self.nu)/(3*self.mu):
            # print('alN too small')
            return 0
        if self.alN > self.max_al(100) or shootingInLTE(self.vJ) < 0:
            # print('alN too large')
            return 1
        sol = root_scalar(shootingInLTE,bracket=[1e-3,self.vJ],rtol=self.rtol,xtol=self.atol)
        return sol.root

    def findMatching(self,vw):
        r"""
        Computes :math:`v_-,\ v_+,\ T_-,\ T_+` for a deflagration or hybrid solution when the wall velocity is vw.
        
        Parameters
        ----------
        vw : double
            Wall velocity at which to solve the matching equation.

        Returns
        -------
        vp : double
            Value of :math:`v_+` that solves the matching equation.
        vm : double
            Value of :math:`v_-` that solves the matching equation.
        Tp : double
            Value of :math:`T_+` that solves the matching equation.
        Tm : double
            Value of :math:`T_-` that solves the matching equation.

        """
        if vw > self.vJ:
            return self.detonation_vAndT(vw)
        
        vm = min(self.cb,vw)

        if vw > self.vMax:
            # alN too small for shock 
            return (None,None,None,None)
        
        if vw < self.vMin:
            # alN too large for shock
            return (None,None,None,None)

        shockIntegrator = lambda vp: self.__shooting(vw,vp)

        ## Please add reference to a paper where these can be found (with eq numbers) 

        vp_max = min(self.cs2/vw,vw) #Follows from  v+max v- = 1/self.cs2, see page 6 of arXiv:1004.4187

        try:
            sol = root_scalar(shockIntegrator, bracket=(0,vp_max), rtol=self.rtol, xtol=self.atol)

        except Exception as e:
#            print("!!! Exception in HydroTemplateModel.findMatching():")
#            print(e)
#            print()
            return (None,None,None,None) # If no deflagration solution exists, returns None.
        
        vp = sol.root
        alp = (vp/vm-1.)*(vp*vm/self.cb2 - 1.)/(1-vp**2)/3. #This is equation 20a of arXiv:2303.10171 solved for alpha_+
        wp = self.w_from_alpha(alp)
        Tp = self.Tnucl*wp**(1/self.mu) #This follows from equation 22 and 23 of arXiv:2303.10171, and setting wn = 1
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

        NOTE: the sign of c1 is chosen to match the convention for the fluid velocity used in EOM and
        Hydro. In those conventions, vp would be negative, and therefore c1 has to be negative as well.
        """
        if vwTry < self.vMin:
            print('This wall velocity is too small for the chosen nucleation temperature. findHydroBoundaries will return zero.')
            return (0,0,0,0,0)

        vp,vm,Tp,Tm = self.findMatching(vwTry)
        if vp is None:
            return (vp,vm,Tp,Tm, None)
        wHighT = self.wN*(Tp/self.Tnucl)**self.mu
        pHighT = self.pN+((Tp/self.Tnucl)**self.mu-1)*self.wN/self.mu
        c1 = -wHighT*vp/(1-vp**2)
        c2 = pHighT+wHighT*vp**2/(1-vp**2)
        vMid = -0.5*(vm+vp)  # minus sign for convention change
        return (c1, c2, Tp, Tm, vMid)

    def max_al(self,upper_limit=100):
        r"""
        Computes the highest value of :math:`\alpha_n` at which a hybrid solution can be found.

        Parameters
        ----------
        upper_limit : double, optional
            Largest value of :math:`\alpha_n` at which the solver will look. If the true value is above upper_limit,
        
                
        Returns
        -------
        upper_limit : double
            Upper limit for :math:`\alpha_n`. The default is 100.


        """
        vm = self.cb
        lower_limit = (1-self.psiN)/3
        def func(alN):
            vw = self.findJouguetVelocity(alN)
            vp = self.cs2/vw
            ga2p,ga2m = 1/(1-vp**2),1/(1-vm**2)
            wp = (vp+vw-vw*self.mu)/(vp+vw-vp*self.mu)
            psi = self.psiN*wp**(self.nu/self.mu-1)
            al = (self.mu-self.nu)/(3*self.mu)+(alN-(self.mu-self.nu)/(3*self.mu))/wp
            return vp*vm*al/(1-(self.nu-1)*vp*vm)-(1-3*al-(ga2p/ga2m)**(self.nu/2)*psi)/(3*self.nu)

        if func(upper_limit) < 0:
            maximum = minimize_scalar(lambda x: -func(x),bounds=[(1-self.psiN)/3,upper_limit],method='Bounded')
            if maximum.fun > 0:
                return upper_limit
            else:
                upper_limit = maximum.x
        if func(lower_limit) > 0:
            minimum = minimize_scalar(func,bounds=[lower_limit,upper_limit],method='Bounded')
            if minimum.fun > 0:
                return lower_limit
            else:
                upper_limit = minimum.x

        sol = root_scalar(func,bracket=(lower_limit,upper_limit),rtol=self.rtol,xtol=self.atol)
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
            # print('No detonation solution')
            return 0
        sol = root_scalar(matching_eq,bracket=(self.vJ+1e-10,1-1e-10),rtol=self.rtol,xtol=self.atol)
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
