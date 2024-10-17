"""
Class for solving the hydrodynamic equations for the fluid velocity and temperature by 
approximating the equation of state by the template model.
"""

import warnings
import logging
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar, minimize_scalar, OptimizeResult
from .exceptions import WallGoError
from .helpers import gammaSq
from .thermodynamics import Thermodynamics


class HydrodynamicsTemplateModel:
    """
    Class for solving the matching equations and computing vw in LTE by fitting to the 
    template model, where the speeds of sound are assumed to be constant. This generally 
    offers a good approximation to realistic models, while being much faster to treat.

    References
    ----------
    .. [GKSvdV20] Felix Giese, Thomas Konstandin, Kai Schmitz and Jorinde van de Vis
        Model-independent energy budget for LISA
        arXiv:2010.09744 (2020)

    .. [ALvdV23] Wen-Yuan Ai, Benoit Laurent, and Jorinde van de Vis.
        Model-independent bubble wall velocities in local thermal equilibrium.
        arXiv:2303.10171 (2023).

    NOTE: We use the conventions that the velocities are always positive, even in the 
    wall frame (vp and vm).
    These conventions are consistent with the literature, e.g. with arxiv:1004.4187.
    These conventions differ from the conventions used in the EOM and Boltzmann part of 
    the code. The conversion is made in findHydroBoundaries.

    """

    def __init__(
        self, thermodynamics: Thermodynamics, rtol: float = 1e-6, atol: float = 1e-10
    ) -> None:
        r"""
        Initialize the HydroTemplateModel class.
        Computes :math:`\alpha_n,\ \Psi_n,\ c_s,\ c_b` and other thermodynamics
        quantities (see [ALvdV23]_ for the definitions of these variables).

        Parameters
        ----------
        thermodynamics : class
        rtol : float, optional
            Default value is 1e-6.
        atol : float, optional
            Default value is 1e-10.

        Returns
        -------
        None

        """
        self.thermodynamics = thermodynamics
        self.rtol, self.atol = rtol, atol
        pHighT, pLowT = thermodynamics.pHighT(
            thermodynamics.Tnucl
        ), thermodynamics.pLowT(thermodynamics.Tnucl)
        wHighT, wLowT = thermodynamics.wHighT(
            thermodynamics.Tnucl
        ), thermodynamics.wLowT(thermodynamics.Tnucl)
        eHighT, eLowT = wHighT - pHighT, wLowT - pLowT

        ## Calculate sound speed squared in both phases, needs to be > 0
        self.cb2 = float(thermodynamics.csqLowT(thermodynamics.Tnucl))
        self.cs2 = float(thermodynamics.csqHighT(thermodynamics.Tnucl))

        if self.cb2 < 0 or self.cs2 < 0:
            raise WallGoError(
                "Invalid sound speed at nucleation temperature",
                data={"csqLowT": self.cb2, "csqHighT": self.cs2},
            )
        if self.cb2 > 0.4 or self.cs2 > 0.4:
            warnings.warn(f"Warning: One of the sound speed at Tnucl is unusually"\
                          f" large (cb^2={self.cb2}, cs^2={self.cs2}). This can lead"\
                          f" to errors later.")

        ## Calculate other thermodynamics quantities like alpha and Psi
        self.alN = float((eHighT - eLowT - (pHighT - pLowT) / self.cb2) / (3 * wHighT))
        self.psiN = float(wLowT / wHighT)
        self.cb = np.sqrt(self.cb2)
        self.cs = np.sqrt(self.cs2)
        ## Enthalpy outside the bubble at Tn
        self.wN = float(wHighT)
        ## Pressure outside the bubble at Tn
        self.pN = float(pHighT)
        self.Tnucl = thermodynamics.Tnucl

        self.nu = 1 + 1 / self.cb2
        self.mu = 1 + 1 / self.cs2
        self.vJ = self.findJouguetVelocity()
        self.vMin = self.minVelocity()
        self.epsilon = self.wN*(1/self.mu-(1-3*self.alN)/self.nu)

    def findJouguetVelocity(self, alN: float | None = None) -> float:
        r"""
        Finds the Jouguet velocity, corresponding to the phase transition strength 
        :math:`\alpha_n`, using 
        :math:`v_J = c_b \frac{1 + \sqrt{3 \alpha_n(1 - c_b^2 + 3 c_b^2 \alpha_n)}}
        {1+ 3 c_b^2 \alpha_n}`
        (eq. (25) of [ALvdV23]_).

        Parameters
        ----------
        alN : float or None
            phase transition strength at the nucleation temperature, :math:`\alpha_n`.
            If :math:`\alpha_n` is not specified, the value defined by the model is
            used. Default is None.

        Returns
        -------
        vJ: float
            The value of the Jouguet velocity.

        """

        if alN is None:
            alN = self.alN
        return float(
            self.cb
            * (1 + np.sqrt(3 * alN * (1 - self.cb2 + 3 * self.cb2 * alN)))
            / (1 + 3 * self.cb2 * alN)
        )

    def minVelocity(self) -> float:
        r"""
        Finds the minimum velocity that is possible for a given nucleation temperature.
        It is found by shooting in vp with :math:`\alpha_+ = 1/3` at the wall. This is
        the maximum value of :math:`\alpha_+` possible. The wall velocity which yields 
        :math:`\alpha_+ = 1/3` for a given :math:`\alpha_N` is the minimum possible wall
        velocity.

        It is possible that no solution can be found, in this case there is no minimum
        value of the wall velocity and the function returns zero.

        Returns
        -------
        vmin: float
            The minimum value of the wall velocity for which a solution can be found
        """
        if self.alN < 1 / 3:
            return 0.0
        return float(
            root_scalar(
                lambda vw: self._shooting(vw, 0),
                bracket=(1e-6, self.vJ),
                rtol=self.rtol,
                xtol=self.atol,
            ).root
        )

    def getVp(self, vm: float, al: float, branch: int = -1) -> float:
        r"""
        Solves the matching equation for :math:`v_+`.

        Parameters
        ----------
        vm : float
            Plasma velocity in the wall frame right behind the wall :math:`v_-`.
        al : float
            phase transition strength at the temperature right in front of the wall
            :math:`\alpha_+`.
        branch : int, optional
            Select the branch of the matching equation's solution. Can either be 1 for
            detonation or -1 for deflagration/hybrid. Default is -1.

        Returns
        -------
        vp : float
            Plasma velocity in the wall frame right in front of the the wall
            :math:`v_+`.

        """
        disc = max(
            0,
            vm**4
            - 2 * self.cb2 * vm**2 * (1 - 6 * al)
            + self.cb2**2 * (1 - 12 * vm**2 * al * (1 - 3 * al)),
        )
        return float(
            0.5
            * (self.cb2 + vm**2 + branch * np.sqrt(disc))
            / (vm + 3 * self.cb2 * vm * al)
        )

    def wFromAlpha(self, al: float) -> float:
        r"""
        Finds the enthlapy :math:`w_+` corresponding to :math:`\alpha_+` using the
        equation of state of the template model.

        Parameters
        ----------
        al : float
            alpha parameter at the temperature :math:`T_+` in front of the wall
            :math:`\alpha_+`.

        Returns
        -------
        float
            :math:`w_+`.

        """
        # Add 1e-100 to avoid having something like 0/0
        sign = np.sign((1-3*self.alN)*self.mu-self.nu)*np.sign((1-3*al)*self.mu-self.nu)
        return sign*(abs((1 - 3 * self.alN) * self.mu - self.nu) + 1e-100) / (
            abs((1 - 3 * al) * self.mu - self.nu) + 1e-100
        )

    def _findTm(self, vm: float, vp: float, Tp: float) -> float:
        r"""
        Finds :math:`T_-` as a function of :math:`v_-,\ v_+,\ T_+` using the matching
        equations.

        Parameters
        ----------
        vm : float
            Value of the fluid velocity in the wall frame, right behind the bubble wall
        vp : float
            Value of the fluid velocity in the wall frame, right in front of the bubble
        Tp : float
            Plasma temperature right in front of the bubble wall

        Returns
        -------
        Tm : float
            Plasma temperature right behind the bubble wall
        """
        # a paramaters appearing in the definition of the template model
        try:
            ap = 3 / (self.mu * self.Tnucl**self.mu)
        except OverflowError:
            # If self.mu is large, the exponential can overflow and trigger an error
            ap = 0
        try:
            am = 3 * self.psiN / (self.nu * self.Tnucl**self.nu)
        except OverflowError:
            # Same thing
            am = 0
        return float(
            (
                (ap * vp * self.mu * (1 - vm**2) * Tp**self.mu)
                / (am * vm * self.nu * (1 - vp**2))
            )
            ** (1 / self.nu)
        )

    def _eqWall(self, al: float, vm: float, branch: int = -1) -> float:
        """
        Residual of the matching equation at the bubble wall.

        Parameters
        ----------
        al : float
            phase transition strength at the temperature right in front of the wall
            :math:`\alpha_+`.
        vm : float
            Value of the fluid velocity in the wall frame, right behind the bubble wall
        branch : int, optional
            Select the branch of the matching equation's solution. Can either be 1 for
            detonation or -1 for deflagration/hybrid. Default is -1.

        Returns
        -------
        float
            Residual of the matching equation
        """
        vp = self.getVp(vm, al, branch)
        psi = self.psiN * self.wFromAlpha(al) ** (self.nu / self.mu - 1)
        return float(
            vp * vm * al / (1 - (self.nu - 1) * vp * vm)
            - (1 - 3 * al - (gammaSq(vp) / gammaSq(vm)) ** (self.nu / 2) * psi)
            / (3 * self.nu)
        )

    def solveAlpha(self, vw: float, constraint: bool = True) -> float:
        r"""
        Finds the value of :math:`\alpha_+` that solves the matching equation at the
        wall by varying :math:`v_-`.

        Parameters
        ----------
        vw : float
            Wall velocity at which to solve the matching equation.
        constraint : bool, optional
            If True, imposes :math:`v_+<\min(c_s^2/v_w,v_w)` on the solution. Otherwise,
            the constraint :math:`v_+<v_-` is used instead. Default is True.

        Returns
        -------
        alp : float
            Value of :math:`\alpha_+` that solves the matching equation.

        """
        vm = min(self.cb, vw)
        vpMax = min(self.cs2 / vw, vw) if constraint else vm

        # Find lower and upper bounds on alpha
        alMin = max(
            (vm - vpMax)
            * (self.cb2 - vm * vpMax)
            / (3 * self.cb2 * vm * (1 - vpMax**2)),
            (self.mu - self.nu) / (3 * self.mu),
            0,
        ) + 1e-10
        alMax = 1 / 3
        branch = -1

        if self._eqWall(alMin, vm) * self._eqWall(alMax, vm) > 0 and vm > self.cb2:
            # If alMin and alMax don't bracket the deflagration solution, try with the
            # detonation one, which only exists when vm > cb^2.
            branch = 1
        try:
            sol = root_scalar(
                self._eqWall,
                (vm, branch),
                bracket=(alMin, alMax),
                rtol=self.rtol,
                xtol=self.atol,
            )
            return float(sol.root)
        except ValueError as exc:
            raise WallGoError("alpha can not be found", data={"vw": vw}) from exc

    def _dxiAndWdv(self, v: float, xiAndW: np.ndarray) -> np.ndarray:
        """
        Fluid equations in the shock wave as a function of v.
        """
        xi, w = xiAndW
        muXiV = (xi - v) / (1 - xi * v)
        dwdv = w * (1 + 1 / self.cs2) * muXiV / (1 - v**2)
        if v != 0:
            dxidv = (
                xi * (1 - v * xi) * (muXiV**2 / self.cs2 - 1) / (2 * v * (1 - v**2))
            )
        else:
            # If v = 0, dxidv is set to a very high value
            dxidv = 1e50
        return np.array([dxidv, dwdv])

    def integratePlasma(self, v0: float, vw: float, wp: float) -> OptimizeResult:
        """
        Integrates the fluid equations in the shock wave until it reaches the shock
        front.

        Parameters
        ----------
        v0 : float
            Plasma velocity just in front of the wall (in the frame of the bubble's
                                                       center).
        vw : float
            Wall velocity.
        wp : float
            Enthalpy just in front of the wall.

        Returns
        -------
        Bunch object returned by the scipy function integrate.solve_ivp containing the
        solution of the fluid equations.

        """

        def event(v: float, xiAndW: np.ndarray) -> float:
            # Function that is 0 at the shock wave front. Is used by solve_ivp to stop
            # the integration
            xi = xiAndW[0]
            return float((xi * (xi - v) / (1 - xi * v) - self.cs2) * v)

        event.terminal = True
        sol = solve_ivp(
            self._dxiAndWdv, (v0, 1e-10), [vw, wp], events=event, rtol=self.rtol, atol=0
        )
        return sol

    def _shooting(self, vw: float, vp: float) -> float:
        """
        Integrates through the shock wave and returns the residual of the matching
        equation at the shock front.
        """
        vm = min(self.cb, vw)
        al = (vp / vm - 1.0) * (vp * vm / self.cb2 - 1.0) / (1 - vp**2) / 3.0
        wp = self.wFromAlpha(al)
        if abs(vp * vw - self.cs2) < 1e-12:
            # If the wall is already very close to the shock front, we do not integrate
            # through the shock wave to avoid any error due to rounding error.
            vpSW = vw
            vmSW = vp
            wmSW = wp
        elif vw == vp:
            # If the plasma is at rest in front of the wall, there is no variation of
            # plasma velocity and temperature in the shock wave
            vpSW = self.cs
            vmSW = self.cs
            wmSW = wp
        else:
            sol = self.integratePlasma((vw - vp) / (1 - vw * vp), vw, wp)
            vpSW = sol.y[0, -1]
            vmSW = (vpSW - sol.t[-1]) / (1 - vpSW * sol.t[-1])
            wmSW = sol.y[1, -1]
        return vpSW / vmSW - ((self.mu - 1) * wmSW + 1) / ((self.mu - 1) + wmSW)

    def findvwLTE(self) -> float:
        """
        Computes the wall velocity for a deflagration or hybrid solution. Uses the 
        method described in [ALvdV23]_.

        Returns
        -------
        vwLTE : float
            Wall velocity in local thermal equilibrium.
        """

        def shootingInLTE(vw: float) -> float:
            vm = min(self.cb, vw)
            al = self.solveAlpha(vw)
            vp = self.getVp(vm, al)
            return self._shooting(vw, vp)

        if self.alN < (1 - self.psiN) / 3 or self.alN <= (self.mu - self.nu) / (
            3 * self.mu
        ):
            # alpha is too small
            return 0.0
        if self.alN > self.maxAl(100) or shootingInLTE(self.vJ) < 0:
            # alpha is too large
            return 1.0

        sol = root_scalar(
            shootingInLTE, bracket=[1e-3, self.vJ], rtol=self.rtol, xtol=self.atol
        )
        return float(sol.root)

    def findMatching(self, vw: float) -> tuple[float, ...] | tuple[None, ...]:
        r"""
        Computes :math:`v_-,\ v_+,\ T_-,\ T_+` for a deflagration or hybrid solution
        when the wall velocity is vw.

        Parameters
        ----------
        vw : float
            Wall velocity at which to solve the matching equation.

        Returns
        -------
        tuple[float | None, ...]
            Tuple containing :math:`v_+`, :math:`v_-`, :math:`T_+`, :math:`T_-` and 
            velocityMid. If the solver wasn't able to find a solution, returns a tuple
            of None.

        """
        if vw > self.vJ:
            return self.detonationVAndT(vw)

        vm = min(self.cb, vw)

        if vw < self.vMin:
            # alN too large for shock
            return (None, None, None, None)

        vpMax = min(
            self.cs2 / vw, vw
        )  # Follows from  v+max v- = 1/self.cs2, see page 6 of arXiv:1004.4187
        vpMin = 0

        # Change vpMin or vpMax in case wp is negative between vpMin and vpMax
        sqrtDisc = (self.mu+vm**2*self.mu*(self.nu-1))**2-4*vm**2*self.nu**2*(self.mu-1)
        if sqrtDisc >= 0:
            # vp at which wp changes sign
            vpSignChangeWp = (self.mu*(1-vm**2*(1-self.nu))-np.sqrt(sqrtDisc))/(
                2*vm*self.nu*(self.mu-1))
            if not np.isnan(vpSignChangeWp):
                if vpMin < vpSignChangeWp < vpMax:
                    vpMax = vpSignChangeWp-1e-10

        try:
            sol = root_scalar(
                lambda vp: self._shooting(vw, vp),
                bracket=(vpMin, vpMax),
                rtol=self.rtol,
                xtol=self.atol,
            )

        except ValueError:
            return (
                None,
                None,
                None,
                None,
            )  # If no deflagration solution exists, returns None.

        vp = sol.root
        alp = (
            (vp / vm - 1.0) * (vp * vm / self.cb2 - 1.0) / (1 - vp**2) / 3.0
        )  # This is equation 20a of arXiv:2303.10171 solved for alpha_+
        wp = self.wFromAlpha(alp)
        Tp = self.Tnucl * wp ** (
            1 / self.mu
        )  # This follows from equation 22-23 of arXiv:2303.10171, and setting wn = 1
        Tm = self._findTm(vm, vp, Tp)
        return vp, vm, Tp, Tm

    def matchDeflagOrHybInitial(self, vw: float, vp: float | None) -> list[float]:
        r"""
        Returns initial guess [Tp, Tm] for the solver in the 
        Hydrodynamics.matchDeflagOrHyb function by computing the corresponding 
        quantities in the template models. See Refs. [GKSvdV20]_ and [ALvdV23]_ for
        details.

        Parameters
        ----------
        vw : float
            Wall velocity.
        vp : float or None
            Plasma velocity in front of the wall :math:`v_+`. If None, vp is
            determined from conservation of entropy.

        Returns
        -------
        list[float]
            List containing Tp and Tm, the temperature in front and behind the wall.

        """
        vm = min(vw, self.cb)
        al: float
        if vp is not None:
            al = ((vm - vp) * (self.cb2 - vm * vp)) / (
                3 * self.cb2 * vm * (1 - vp**2)
            )
        else:
            try:
                al = self.solveAlpha(vw, False)
            except WallGoError as exc:
                raise WallGoError(
                    "Failed to find alpha!", data={"vw": vw, "vm": vm}
                ) from exc

            vp = self.getVp(vm, al)
        wp = self.wFromAlpha(al)
        Tp = self.Tnucl * wp ** (1 / self.mu)
        Tm = self._findTm(vm, vp, Tp)
        return [Tp, Tm]

    def findHydroBoundaries(self, vwTry: float) -> tuple[float | None, ...]:
        r"""
        Returns :math:`c_1, c_2, T_+, T_-` for a given wall velocity and nucleation
        temperature.

        NOTE: the sign of c1 is chosen to match the convention for the fluid velocity
        used in EquationOfMotion and Hydrodynamics. In those conventions, vp would be
        negative, and therefore c1 has to be negative as well.
        
        Parameters
        ----------
        vwTry : float
            Wall velocity
        
        Returns
        -------
        tuple[float | None, ...]
            Tuple containing c1, c2, Tp, Tm and velocityMid. If the solver wasn't able 
            to find a solution, returns a tuple of None.
        """
        if vwTry < self.vMin:
            logging.warning(
                """This wall velocity is too small for the chosen nucleation
                temperature. findHydroBoundaries will return zero."""
            )
            return (0, 0, 0, 0, 0)

        vp, vm, Tp, Tm = self.findMatching(vwTry)
        if vp is None or vm is None or Tp is None or Tm is None:
            return (vp, vm, Tp, Tm, None)
        vp, vm, Tp, Tm = float(vp), float(vm), float(Tp), float(Tm)
        wHighT = self.wN * (Tp / self.Tnucl) ** self.mu
        pHighT = self.pN + ((Tp / self.Tnucl) ** self.mu - 1) * self.wN / self.mu
        c1 = -wHighT * vp / (1 - vp**2)
        c2 = pHighT + wHighT * vp**2 / (1 - vp**2)
        velocityMid = -0.5 * (vm + vp)  # minus sign for convention change
        return (c1, c2, Tp, Tm, velocityMid)

    def maxAl(self, upperLimit: float = 100.0) -> float:
        r"""
        Computes the highest value of :math:`\alpha_n` at which a hybrid solution can be
        found by finding the value that gives a solution with :math:`v_w=v_J`.

        Parameters
        ----------
        upperLimit : float, optional
            Largest value of :math:`\alpha_n` at which the solver will look. Default is
            100.

        Returns
        -------
        float
            Maximal value for :math:`\alpha_n`. If the true value is above upperLimit,
            returns upperLimit.


        """
        vm = self.cb
        lowerLimit = (1 - self.psiN) / 3

        ## This function returns the residual of the matching equation when vw=vJ for a
        ## given value of alpha_n.
        def matching(alN: float) -> float:
            vw = self.findJouguetVelocity(alN)
            vp = self.cs2 / vw
            ga2p, ga2m = gammaSq(vp), gammaSq(vm)
            wp = (vp + vw - vw * self.mu) / (vp + vw - vp * self.mu)
            psi = self.psiN * wp ** (self.nu / self.mu - 1)
            al = (self.mu - self.nu) / (3 * self.mu) + (
                alN - (self.mu - self.nu) / (3 * self.mu)
            ) / wp
            return float(
                vp * vm * al / (1 - (self.nu - 1) * vp * vm)
                - (1 - 3 * al - (ga2p / ga2m) ** (self.nu / 2) * psi) / (3 * self.nu)
            )

        ## Finds the min or max of matching  to bracket the root in case it is not
        ## monotonous.
        if matching(upperLimit) < 0:
            maximum = minimize_scalar(
                lambda x: -matching(x),
                bounds=[(1 - self.psiN) / 3, upperLimit],
                method="Bounded",
            )
            if maximum.fun > 0:
                return upperLimit
            upperLimit = maximum.x
        if matching(lowerLimit) > 0:
            minimum = minimize_scalar(
                matching, bounds=[lowerLimit, upperLimit], method="Bounded"
            )
            if minimum.fun > 0:
                return lowerLimit
            upperLimit = minimum.x

        ## Finds the solution with 0 residual.
        sol = root_scalar(
            matching, bracket=(lowerLimit, upperLimit), rtol=self.rtol, xtol=self.atol
        )
        return float(sol.root)

    def detonationVAndT(self, vw: float) -> tuple[float, ...]:
        r"""
        Computes :math:`v_-,\ v_+,\ T_-,\ T_+` for a detonation solution.

        Parameters
        ----------
        vw : float
            Wall velocity.

        Returns
        -------
        vp : float
            Plasma velocity in front of the wall.
        vm : float
            Plasma velocity behind the wall.
        Tp : float
            Temperature in front of the wall.
        Tm : float
            Temperature behind the wall.

        """
        vp = vw
        part = vp**2 + self.cb2 * (1 - 3 * (1 - vp**2) * self.alN)
        vm = (part + np.sqrt(part**2 - 4 * self.cb2 * vp**2)) / (2 * vp)
        Tm = self._findTm(vm, vp, self.Tnucl)
        return vp, vm, self.Tnucl, Tm
