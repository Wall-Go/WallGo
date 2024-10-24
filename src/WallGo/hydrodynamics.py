"""
Classes for solving the hydrodynamic equations for the fluid velocity and temperature.
"""

from typing import Tuple
import logging
import numpy as np
import numpy.typing as npt
from scipy.optimize import root_scalar, root, minimize_scalar
from scipy.integrate import solve_ivp
from .exceptions import WallGoError
from .thermodynamics import Thermodynamics
from .hydrodynamicsTemplateModel import HydrodynamicsTemplateModel
from .helpers import gammaSq, boostVelocity
from .results import HydroResults


class Hydrodynamics:
    """
    Class for solving the hydrodynamic equations of the plasma, at distances far enough
    from the wall such that the wall can be treated as infinitesimally thin.

    NOTE: We use the conventions that the velocities are always positive, even in the
    wall frame (vp and vm). These conventions are consistent with the literature, e.g.
    with arxiv:1004.4187. These conventions differ from the conventions used in the
    EOM and Boltzmann part of the code. The conversion is made in findHydroBoundaries.
    """

    def __init__(
        self,
        thermodynamics: Thermodynamics,
        tmax: float,
        tmin: float,
        rtol: float,
        atol: float,
    ):
        """
        Initialisation

        Parameters
        ----------
        thermodynamics : class
        tmax : float
        tmin : float
        rtol : float
        atol : float

        Returns
        -------
        cls: Hydrodynamics
            An object of the Hydrodynamics class.

        """

        self.thermodynamics = thermodynamics
        self.Tnucl = thermodynamics.Tnucl

        self.TMaxHighT = thermodynamics.freeEnergyHigh.maxPossibleTemperature[0]
        self.TMinHighT = thermodynamics.freeEnergyHigh.minPossibleTemperature[0]
        self.TMaxLowT = thermodynamics.freeEnergyLow.maxPossibleTemperature[0]
        self.TMinLowT = thermodynamics.freeEnergyLow.minPossibleTemperature[0]

        self.TMaxHydro = tmax * self.Tnucl
        self.TMinHydro = tmin * self.Tnucl

        self.rtol, self.atol = rtol, atol

        self.template = HydrodynamicsTemplateModel(thermodynamics, rtol=rtol, atol=atol)

        try:
            self.vJ = self.findJouguetVelocity()
        except WallGoError:
            logging.warning(
                "Couldn't find Jouguet velocity, we continue "
                "with the Jouguet velocity of the template model"
            )
            self.vJ = self.template.vJ

        self.vBracketLow = 1e-3
        # Minimum velocity that allows a shock with the given nucleation temperature
        self.vMin = max(self.vBracketLow, self.minVelocity())

        # Bool which is set to true if the upper temperature in the phase tracing
        # limits the allowed velocity range. First (second) value corresponds to
        # the high (low )temperature phase
        self.doesPhaseTraceLimitvmax = [False, False]

        self.success = False

    def findJouguetVelocity(self) -> float:
        r"""
        Finds the Jouguet velocity for a thermal effective potential, defined by
        thermodynamics, and at the model's nucleation temperature, using that the
        derivative of :math:`v_+` with respect to :math:`T_-` is zero at the
        Jouguet velocity.

        Returns
        -------
        vJ: float
            The value of the Jouguet velocity for this model.

        """
        pHighT = self.thermodynamics.pHighT(self.Tnucl)
        eHighT = self.thermodynamics.eHighT(self.Tnucl)

        def vpDerivNum(tm: float) -> float:  # The numerator of the derivative of v+^2
            pLowT = self.thermodynamics.pLowT(tm)
            eLowT = self.thermodynamics.eLowT(tm)
            num1 = pHighT - pLowT  # First factor in the numerator of v+^2
            num2 = pHighT + eLowT
            den1 = eHighT - eLowT  # First factor in the denominator of v+^2
            den2 = eHighT + pLowT
            dnum1 = -self.thermodynamics.dpLowT(
                tm
            )  # T-derivative of first factor wrt tm
            dnum2 = self.thermodynamics.deLowT(tm)
            dden1 = -dnum2  # T-derivative of second factor wrt tm
            dden2 = -dnum1
            return (
                dnum1 * num2 * den1 * den2
                + num1 * dnum2 * den1 * den2
                - num1 * num2 * dden1 * den2
                - num1 * num2 * den1 * dden2
            )

        # For detonations, Tm has a lower bound of Tn, but no upper bound.
        # We make a guess for Tmax, and if it does not work we use the secant method

        Tmin = self.Tnucl
        Tmax = min(max(2 * Tmin, self.TMaxLowT), self.TMaxHydro)

        bracket1, bracket2 = vpDerivNum(Tmin), vpDerivNum(Tmax)
        while bracket1 * bracket2 > 0 and Tmax < self.TMaxHydro:
            Tmin = Tmax
            Tmax = min(Tmax + self.Tnucl, self.TMaxHydro)
            bracket1, bracket2 = vpDerivNum(Tmin), vpDerivNum(Tmax)

        tmSol: float
        if bracket1 * bracket2 <= 0:
            # If Tmin and Tmax bracket our root, use the 'brentq' method.
            rootResult = root_scalar(
                vpDerivNum,
                bracket=[self.Tnucl, Tmax],
                method="brentq",
                xtol=self.atol,
                rtol=self.rtol,
            )
        else:
            # If we cannot bracket the root, use the 'secant' method instead.
            rootResult = root_scalar(
                vpDerivNum,
                method="secant",
                x0=self.Tnucl,
                x1=Tmax,
                xtol=self.atol,
                rtol=self.rtol,
            )

        if rootResult.converged:
            tmSol = rootResult.root
        else:
            raise WallGoError(
                "Failed to solve Jouguet velocity at \
                              input temperature!",
                data={"flag": rootResult.flag, "Root result": rootResult},
            )

        vp = np.sqrt(
            (pHighT - self.thermodynamics.pLowT(tmSol))
            * (pHighT + self.thermodynamics.eLowT(tmSol))
            / (eHighT - self.thermodynamics.eLowT(tmSol))
            / (eHighT + self.thermodynamics.pLowT(tmSol))
        )
        return float(vp)

    def fastestDeflag(self) -> float:
        r"""
        Finds the largest wall velocity for which the temperature of the plasma is
        within the allowed regime, by finding the velocity for which
        Tm = TMaxLowT or Tp = TMaxHighT.
        Returns the Jouguet velocity if no solution can be found.

        Returns
        -------
        vmax: float
            The value of the fastest deflagration/hybrid solution for this model

        """

        def TpTm(vw: float) -> list[float]:
            _, _, Tp, Tm = self.findMatching(vw)
            return [Tp, Tm]

        if (
            TpTm(self.vJ - self.vBracketLow)[1] < self.TMaxLowT
            and TpTm(self.vJ - self.vBracketLow)[0] < self.TMaxHighT
        ):
            return self.vJ

        def TmMax(vw: float) -> float:
            return TpTm(vw)[1] - self.TMaxLowT

        try:
            vmax1 = root_scalar(
                TmMax,
                bracket=[self.vMin + self.vBracketLow, self.vJ - self.vBracketLow],
                method="brentq",
                xtol=self.atol,
                rtol=self.rtol,
            ).root

            if not self.thermodynamics.freeEnergyLow.maxPossibleTemperature[1]:
                self.doesPhaseTraceLimitvmax[1] = True

        except ValueError:
            vmax1 = self.vJ
            self.doesPhaseTraceLimitvmax[1] = False

        def TpMax(vw: float) -> float:
            return TpTm(vw)[0] - self.TMaxHighT

        try:
            vmax2 = root_scalar(
                TpMax,
                bracket=[self.vMin + self.vBracketLow, self.vJ - self.vBracketLow],
                method="brentq",
                xtol=self.atol,
                rtol=self.rtol,
            ).root
            if not self.thermodynamics.freeEnergyHigh.maxPossibleTemperature[1]:
                self.doesPhaseTraceLimitvmax[0] = True

        except ValueError:
            vmax2 = self.vJ
            self.doesPhaseTraceLimitvmax[0] = False

        return float(min(vmax1, vmax2))

    def slowestDeton(self) -> float:
        r"""
        Finds the smallest detonation wall velocity for which the temperature of the
        plasma is within the allowed range, by finding the velocity for which
        Tm = TMaxLowT. For detonations, Tp = Tn, so always in the allowed range.
        Returns 1 if Tm is above TMaxLowT for vw = 1, and returns the Jouguet velocity
        if no solution can be found.

        Returns
        -------
        vmin: float
            The value of the slowest detonation solution for this model
        """

        def TpTm(vw: float) -> list[float]:
            _, _, Tp, Tm = self.findMatching(vw)
            return [Tp, Tm]

        if TpTm(1)[1] > self.TMaxLowT:
            return 1

        def TmMax(vw: float) -> float:
            return TpTm(vw)[1] - self.TMaxLowT

        try:
            vmin = root_scalar(
                TmMax,
                bracket=[self.vJ + 1e-4, 1],
                method="brentq",
                xtol=self.atol,
                rtol=self.rtol,
            ).root
            # We add 0.01 because the functions in EOM become unstable at vmin
            return float(min(1, vmin + 0.01))

        except ValueError:
            return self.vJ

    def vpvmAndvpovm(self, Tp: float, Tm: float) -> Tuple[float, float]:
        r"""
        Finds :math:`v_+v_-` and :math:`v_+/v_-` as a function of :math:`T_+, T_-`,
        from the matching conditions.

        Parameters
        ----------
        Tp : float
            Plasma temperature right in front of the bubble wall
        Tm : float
            Plasma temperature right behind the bubble wall

        Returns
        -------
        vpvm, vpovm: float
            `v_+v_-` and :math:`v_+/v_-`
        """

        pHighT, pLowT = self.thermodynamics.pHighT(Tp), self.thermodynamics.pLowT(Tm)
        eHighT, eLowT = self.thermodynamics.eHighT(Tp), self.thermodynamics.eLowT(Tm)
        vpvm = (
            (pHighT - pLowT) / (eHighT - eLowT)
            if eHighT != eLowT
            else (pHighT - pLowT) * 1e50
        )
        vpovm = (eLowT + pHighT) / (eHighT + pLowT)
        return (vpvm, vpovm)

    def matchDeton(self, vw: float) -> Tuple[float, float, float, float]:
        r"""
        Solves the matching conditions for a detonation. In this case, :math:`v_w = v_+`
        and :math:`T_+ = T_n` and :math:`v_-,T_-` are found from the matching equations.

        Parameters
        ---------
        vw : float
            Wall velocity

        Returns
        -------
        vp,vm,Tp,Tm : float
            The value of the fluid velocities in the wall frame and the temperature
            right in front of the wall and right behind the wall.

        """
        vp = vw
        Tp = self.Tnucl
        pHighT, wHighT = self.thermodynamics.pHighT(Tp), self.thermodynamics.wHighT(Tp)
        eHighT = wHighT - pHighT

        def tmFromvpsq(tm: float) -> float:
            pLowT, wLowT = self.thermodynamics.pLowT(tm), self.thermodynamics.wLowT(tm)
            eLowT = wLowT - pLowT
            return float(
                vp**2 * (eHighT - eLowT)
                - (pHighT - pLowT) * (eLowT + pHighT) / (eHighT + pLowT)
            )

        minimizeResult = minimize_scalar(
            tmFromvpsq,
            bounds=[self.Tnucl, self.TMaxHydro],
            method="Bounded",
        )

        if minimizeResult.success:
            Tmax = minimizeResult.x
        else:
            raise WallGoError(minimizeResult.message, minimizeResult)
        if minimizeResult.fun > 0:
            raise WallGoError(
                "No solutions to the matching equations were found. This can be "
                "caused by a bad interpolation of the free energy. Try decreasing "
                "phaseTracerTol.",
                minimizeResult,
            )
        rootResult = root_scalar(
            tmFromvpsq,
            bracket=[self.Tnucl, Tmax],
            method="brentq",
            xtol=self.atol,
            rtol=self.rtol,
        )
        if rootResult.converged:
            Tm = rootResult.root
        else:
            raise WallGoError(rootResult.flag, rootResult)
        vpvm, vpovm = self.vpvmAndvpovm(Tp, Tm)
        vm = np.sqrt(vpvm / vpovm)
        if vp == 1:
            vm = 1
        return (vp, vm, Tp, Tm)

    def matchDeflagOrHyb(
        self, vw: float, vp: float | None = None
    ) -> Tuple[float, float, float, float]:
        r"""
        Obtains the matching parameters :math:`v_+, v_-, T_+, T_-` for a deflagration
        or hybrid by solving the matching relations.

        Parameters
        ----------
        vw : float
            Wall velocity.
        vp : float or None, optional
            Plasma velocity in front of the wall :math:`v_+`. If None, vp is
            determined from conservation of entropy. Default is None.

        Returns
        -------
        vp,vm,Tp,Tm : float
            The value of the fluid velocities in the wall frame and the temperature
            right in front of the wall and right behind the wall.

        """

        def matching(
            mappedTpTm: list[float],
        ) -> Tuple[float, float]:  # Matching relations at the wall interface
            Tpm = self._inverseMappingT(mappedTpTm)
            vmsq = min(vw**2, self.thermodynamics.csqLowT(Tpm[1]))

            if vp is None:
                # Determine vp from entropy conservation, e.g. eq. (15) of 2303.10171
                vpsq = (Tpm[1] ** 2 - Tpm[0] ** 2 * (1 - vmsq)) / Tpm[1] ** 2
            else:
                vpsq = vp**2
            vpvm, vpovm = self.vpvmAndvpovm(Tpm[0], Tpm[1])
            eq1 = vpvm * vpovm - vpsq
            eq2 = vpvm / vpovm - vmsq

            # We multiply the equations by c to make sure the solver
            # does not explore arbitrarly small or large values of Tm and Tp.
            c = (2**2 + (Tpm[0] / Tpm0[0]) ** 2 + (Tpm[1] / Tpm0[1]) ** 2) * (
                2**2 + (Tpm0[0] / Tpm[0]) ** 2 + (Tpm0[1] / Tpm[1]) ** 2
            )
            return (eq1 * c, eq2 * c)

        # Finds an initial guess for Tp and Tm using the template model and make
        # sure it satisfies all the relevant bounds.
        try:
            if vw > self.template.vMin:
                vwTemplate = min(vw, self.template.vJ - 1e-6)
                vpTemplate = vp
                if vp is not None:
                    vpTemplate = min(vp, vwTemplate)
                Tpm0 = self.template.matchDeflagOrHybInitial(vwTemplate, vpTemplate)
            else:
                Tpm0 = [self.Tnucl, 0.99 * self.Tnucl]
        except WallGoError:
            Tpm0 = [
                min(1.1, 1 / np.sqrt(1 - min(vw**2, self.template.cb2))) * self.Tnucl,
                self.Tnucl,
            ]  # The temperature in front of the wall Tp will be above Tnucl,
            # so we use the smallest of 1.1*Tnucl or gamma_-*Tnucl as initial guess
            # (the latter being close to the LTE value of (gamma_-/gamma_+)*T_-).

        if np.any(np.isnan(Tpm0)):
            Tpm0 = [
                min(1.1, 1 / np.sqrt(1 - min(vw**2, self.template.cb2))) * self.Tnucl,
                self.Tnucl,
            ]
        if (vp is not None) and (Tpm0[0] <= Tpm0[1]):
            Tpm0[0] = 1.01 * Tpm0[1]
        if (vp is None) and (
            Tpm0[0] <= Tpm0[1]
            or Tpm0[0]
            > Tpm0[1] / np.sqrt(1 - min(vw**2, self.thermodynamics.csqLowT(Tpm0[1])))
        ):
            Tpm0[0] = (
                Tpm0[1]
                * (
                    1
                    + 1 / np.sqrt(1 - min(vw**2, self.thermodynamics.csqLowT(Tpm0[1])))
                )
                / 2
            )

        # We map Tm and Tp, which we assume to lie between TMinHydro and TMaxHydro,
        # to the interval (-inf,inf) which is used by the solver.
        sol = root(
            matching, self._mappingT(Tpm0), method="hybr", options={"xtol": self.atol}
        )
        self.success = (
            sol.success or np.sum(sol.fun**2) < 1e-6
        )  # If the error is small enough,
        # we consider that root has converged even if it returns False.
        [Tp, Tm] = self._inverseMappingT(sol.x)

        vmsq = min(vw**2, self.thermodynamics.csqLowT(Tm))
        vm = np.sqrt(max(vmsq, 0))
        if vp is None:
            vp = np.sqrt((Tm**2 - Tp**2 * (1 - vm**2))) / Tm

        if np.isnan(vp):
            raise WallGoError(
                "Hydrodynamics error: Not able to find vp in matchDeflagOrHyb. "
                "Can sometimes be caused by a negative sound speed squared. If that is"
                " the case, try decreasing phaseTracerTol or the temperature scale, "
                "which will improve the potential's interpolation.",
                {
                    "vw": vw,
                    "vm": vm,
                    "Tp": Tp,
                    "Tm": Tm,
                    "csq": self.thermodynamics.csqLowT(Tm),
                },
            )
        return vp, vm, Tp, Tm

    def shockDE(
        self, v: float, xiAndT: np.ndarray
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        r"""
        Hydrodynamic equations for the self-similar coordinate :math:`\xi = r/t` and
        the fluid temperature :math:`T` in terms of the fluid velocity :math:`v`
        See e.g. eq. (B.10, B.11) of 1909.10040

        Parameters
        ----------
        v : array
            Fluid velocities.
        xiAndT : array
            Values of the self-similar coordinate :math:`\xi = r/t` and
            the temperature :math:`T`

        Returns
        -------
        eq1, eq2 : array
            The expressions for :math:`\frac{\partial \xi}{\partial v}`
            and :math:`\frac{\partial T}{\partial v}`
        """
        xi, T = xiAndT

        if T <= 0:
            raise WallGoError(
                "Hydrodynamics error: The temperature in the shock wave became "
                "negative during the integration. This can be caused by a too coarse "
                "integration. Try decreasing Hydrodynamics's relative tolerance.",
                {"v": v, "xi": xi, "T": T},
            )

        eq1 = (
            gammaSq(v)
            * (1.0 - v * xi)
            * (boostVelocity(xi, v) ** 2 / self.thermodynamics.csqHighT(T) - 1.0)
            * xi
            / 2.0
            / v
        )
        eq2 = (
            self.thermodynamics.wHighT(T)
            / self.thermodynamics.dpHighT(T)
            * gammaSq(v)
            * boostVelocity(xi, v)
        )
        return [eq1, eq2]

    def solveHydroShock(self, vw: float, vp: float, Tp: float) -> float:
        r"""
        Solves the hydrodynamic equations in the shock for a given wall
        velocity :math:`v_w` and matching parameters :math:`v_+,T_+`
        and returns the corresponding nucleation temperature :math:`T_n`,
        which is the temperature of the plasma in front of the shock.

        Parameters
        ----------
        vw : float
            Wall velocity
        vp : float
            Value of the fluid velocity in the wall frame, right in front of the bubble
        Tp : float
            Value of the fluid temperature right in front of the bubble

        Returns
        -------
        Tn : float
            Nucleation temperature

        """

        def shock(v: float, xiAndT: np.ndarray | list) -> float:
            xi, T = xiAndT
            return float(boostVelocity(xi, v) * xi - self.thermodynamics.csqHighT(T))

        shock.terminal = True  # What's happening here?
        xi0T0 = [vw, Tp]
        vpcent = boostVelocity(vw, vp)
        if shock(vpcent, xi0T0) > 0:
            vmShock = vpcent
            xiShock = vw
            TmShock = Tp
        elif vw == vp:
            vmShock = 0
            xiShock = self.thermodynamics.csqHighT(Tp) ** 0.5
            TmShock = Tp
        else:
            solshock = solve_ivp(
                self.shockDE,
                [vpcent, 1e-8],
                xi0T0,
                events=shock,
                rtol=self.rtol,
                atol=0,
            )  # solve differential equation all the way from v = v+ to v = 0
            vmShock = solshock.t[-1]
            xiShock, TmShock = solshock.y[:, -1]

        # continuity of the ii-compontent of the energy-momentum tensor
        def TiiShock(tn: float) -> float:
            return self.thermodynamics.wHighT(tn) * xiShock / (
                1 - xiShock**2
            ) - self.thermodynamics.wHighT(TmShock) * boostVelocity(
                xiShock, vmShock
            ) * gammaSq(
                boostVelocity(xiShock, vmShock)
            )

        # Make an initial guess for the temperature range in which Tnucl will be found
        Tmin, Tmax = max(self.Tnucl / 2, self.TMinHydro), TmShock
        bracket1, bracket2 = TiiShock(Tmin), TiiShock(Tmax)

        # If the range does not capture the shock, we lower Tmin
        while bracket1 * bracket2 > 0 and Tmin > self.TMinHydro:
            Tmax = Tmin
            bracket2 = bracket1
            Tmin = max(Tmin / 1.5, self.TMinHydro)
            bracket1 = TiiShock(Tmin)

        if bracket1 * bracket2 <= 0:
            # If Tmin and Tmax bracket our root, use the 'brentq' method.
            TnRootResult = root_scalar(
                TiiShock,
                bracket=[Tmin, Tmax],
                method="brentq",
                xtol=self.atol,
                rtol=self.rtol,
            )
        else:
            # If we cannot bracket the root, use the 'secant' method instead.
            TnRootResult = root_scalar(
                TiiShock,
                method="secant",
                x0=self.Tnucl,
                x1=TmShock,
                xtol=self.atol,
                rtol=self.rtol,
            )

        if not TnRootResult.converged:
            raise WallGoError(TnRootResult.flag, TnRootResult)
        return float(TnRootResult.root)

    def strongestShock(self, vw: float) -> float:
        r"""
        Finds the smallest nucleation temperature possible for a given wall velocity vw.
        The strongest shock is found by finding the value of Tp for which vp = 0 and
        Tm is TMinHydro (very small). The correspdoning nucleation temperature is
        obtained from solveHydroShock at this value of Tp and vp=0.

        Parameters
        ----------
        vw: float
            Value of the wall velocity.

        Returns
        -------
        Tnucl: float
            The nucleation temperature corresponding to the strongest shock.

        """

        def matchingStrongest(Tp: float) -> float:
            return self.thermodynamics.pHighT(Tp) - self.thermodynamics.pLowT(
                self.TMinHydro
            )

        try:
            TpStrongestRootResult = root_scalar(
                matchingStrongest,
                bracket=[self.TMinHydro, self.TMaxHydro],
                rtol=self.rtol,
                xtol=self.atol,
            )
            if not TpStrongestRootResult.converged:
                raise WallGoError(
                    TpStrongestRootResult.flag,
                    TpStrongestRootResult,
                )
            Tpstrongest = TpStrongestRootResult.root
            return self.solveHydroShock(vw, 0, Tpstrongest)
        except ValueError:
            return 0

    def minVelocity(self) -> float:
        r"""
        Finds the smallest velocity for which a deflagration/hybrid is possible for the
        given nucleation temperature. Returns 0 if no solution can be found.

        Returns
        -------
        vMin: float
            The smallest velocity that allows for a deflagration/hybrid.

        """

        def strongestshockTnucl(vw: float) -> float:
            return self.strongestShock(vw) - self.Tnucl

        try:
            vMinRootResult = root_scalar(
                strongestshockTnucl,
                bracket=(self.vBracketLow, self.vJ),
                rtol=self.rtol,
                xtol=self.atol,
            )
            if not vMinRootResult.converged:
                raise WallGoError(vMinRootResult.flag, vMinRootResult)
            return float(vMinRootResult.root)
        except ValueError:
            return 0

    def findMatching(self, vwTry: float) -> Tuple[float, float, float, float]:
        r"""
        Finds the matching parameters :math:`v_+, v_-, T_+, T_-` as a function
        of the wall velocity and for the nucleation temperature of the model.
        For detonations, these follow directly from :func:`matchDeton`,
        for deflagrations and hybrids, the code varies :math:`v_+` until the
        temperature in front of the shock equals the nucleation temperature

        Parameters
        ----------
        vwTry : float
            The value of the wall velocity

        Returns
        -------
        vp,vm,Tp,Tm : float
            The value of the fluid velocities in the wall frame and the
            temperature right in front of the wall and right behind the wall.

        """

        if vwTry > self.vJ:  # Detonation
            vp, vm, Tp, Tm = self.matchDeton(vwTry)

        else:  # Hybrid or deflagration
            # Loop over v+ until the temperature in front of the shock matches
            # the nucleation temperature.

            vpmin = self.vBracketLow
            # The speed of sound below should really be evaluated at Tp, but we use Tn
            # here to save time. We will use Tp later if it doesn't work.
            vpmax = min(vwTry, self.thermodynamics.csqHighT(self.Tnucl) / vwTry)

            def shockTnuclDiff(vpTry: float) -> float:
                _, _, Tp, _ = self.matchDeflagOrHyb(vwTry, vpTry)
                return self.solveHydroShock(vwTry, vpTry, Tp) - self.Tnucl

            shockTnuclDiffMin = shockTnuclDiff(vpmin)
            shockTnuclDiffMax = shockTnuclDiff(vpmax)

            # If no solution was found between vpmin and vpmax, it might be because
            # vpmax was evaluated at Tn instead of Tp.  We thus reevaluate vpmax by
            # solving 'vpmax = cs(Tp(vpmax))^2/vwTry'
            if shockTnuclDiffMin * shockTnuclDiffMax > 0:

                def solveVpmax(vpTry: float) -> float:
                    _, _, Tp, _ = self.matchDeflagOrHyb(vwTry, vpTry)
                    return vpTry - self.thermodynamics.csqHighT(Tp) / vwTry

                if solveVpmax(vwTry) * solveVpmax(vpmax) <= 0:
                    vpmax = root_scalar(
                        solveVpmax,
                        bracket=[vpmax, vwTry],
                        xtol=self.atol,
                        rtol=self.rtol,
                    ).root
                    shockTnuclDiffMax = shockTnuclDiff(vpmax)

            if shockTnuclDiffMin * shockTnuclDiffMax <= 0:
                sol = root_scalar(
                    shockTnuclDiff,
                    bracket=[vpmin, vpmax],
                    xtol=self.atol,
                    rtol=self.rtol,
                )
            else:
                extremum = minimize_scalar(
                    lambda x: np.sign(shockTnuclDiffMax) * shockTnuclDiff(x),
                    bounds=[vpmin, vpmax],
                    method="Bounded",
                )

                if extremum.fun > 0:
                    # In this case, use the template model to compute the matching.
                    # Because the Jouguet velocity can be slightly different in the
                    # template model, we make sure that vwTemplate corresponds to
                    # the appropriate type of solution.
                    if vwTry <= self.vJ:
                        vwTemplate = min(vwTry, self.template.vJ - 1e-6)
                    else:
                        vwTemplate = max(vwTry, self.template.vJ + 1e-6)
                    return self.template.findMatching(vwTemplate)

                sol = root_scalar(
                    shockTnuclDiff,
                    bracket=[vpmin, extremum.x],
                    xtol=self.atol,
                    rtol=self.rtol,
                )
            vp, vm, Tp, Tm = self.matchDeflagOrHyb(vwTry, sol.root)

        return (vp, vm, Tp, Tm)

    def findHydroBoundaries(
        self, vwTry: float
    ) -> Tuple[float, float, float, float, float]:
        r"""
        Finds the relevant boundary conditions (:math:`c_1,c_2,T_+,T_-` and the fluid
        velocity in right in front of the wall) for the scalar and plasma equations of
        motion for a given wall velocity and the model's nucletion temperature.

        NOTE: the sign of :math:`c_1` is chosen to match the convention for the
        fluid velocity used in EOM and Hydro. In those conventions,
        math:`v_+` would be negative, and therefore :math:`c_1` has to
        be negative as well.

        Parameters
        ----------
        vwTry : float
            The value of the wall velocity

        Returns
        -------
        c1,c2,Tp,Tm,velocityMid : float
            The boundary conditions for the scalar field and plasma equation of motion

        """
        if vwTry < self.vMin:
            logging.warning(
                "This wall velocity is too small for the chosen nucleation temperature,"
                " findHydroBoundaries will return zero."
            )
            return (0, 0, 0, 0, 0)

        vp, vm, Tp, Tm = self.findMatching(vwTry)
        if vp is None:
            return (vp, vm, Tp, Tm, None)
        wHighT = self.thermodynamics.wHighT(Tp)
        c1 = -wHighT * gammaSq(vp) * vp
        c2 = self.thermodynamics.pHighT(Tp) + wHighT * gammaSq(vp) * vp**2
        velocityMid = -0.5 * (vm + vp)  # NOTE: minus sign for convention change
        return (c1, c2, Tp, Tm, velocityMid)

    def findvwLTE(self) -> float:
        r"""
        Returns the wall velocity in local thermal equilibrium for the model's
        nucleation temperature. The wall velocity is determined by solving the
        matching condition :math:`T_+ \gamma_+= T_-\gamma_-`. For small wall
        velocity :math:`T_+ \gamma_+> T_-\gamma_-`, and -- if a solution exists --
        :math:`T_+ \gamma_+< T_-\gamma_-` for large wall velocity. If the phase
        transition is too weak for a solution to exist, returns 0. If it is too
        strong, returns 1. The solution is always a deflagration or hybrid.

        Parameters
        ----------

        Returns
        -------
        vwLTE
            The value of the wall velocity for this model in local thermal equilibrium.
        """

        # Function given to the root finder.
        def shockTnuclDiff(
            vw: float,
        ) -> float:
            vp, _, Tp, _ = self.matchDeflagOrHyb(vw)
            Tntry = self.solveHydroShock(vw, vp, Tp)
            return Tntry - self.Tnucl

        # Equation to find the position of the shock front.
        # If shock(vw) < 0, the front is ahead of vw.
        def shock(
            vw: float,
        ) -> float:
            vp, _, Tp, _ = self.matchDeflagOrHyb(vw)
            return vp * vw - self.thermodynamics.csqHighT(Tp)

        self.success = True
        vmin = self.vMin
        vmax = self.vJ - 1e-10

        if (
            shock(vmax) > 0
        ):  # Finds the maximum vw such that the shock front is ahead of the wall.
            try:
                vmax = root_scalar(
                    shock,
                    bracket=[
                        self.thermodynamics.csqHighT(self.Tnucl) ** 0.5,
                        self.vJ,
                    ],
                    xtol=self.atol,
                    rtol=self.rtol,
                ).root
                vmax = vmax - 1e-6  # HACK! why?
            except ValueError:
                return 1  # No shock can be found, e.g. when the PT is too strong --
            # is there a risk here of returning 1 when it should be 0?

        shockTnuclDiffMax = shockTnuclDiff(vmax)
        if (
            shockTnuclDiffMax > 0 or not self.success
        ):  # There is no deflagration or hybrid solution, we return 1.
            return 1

        shockTnuclDiffMin = shockTnuclDiff(vmin)
        if shockTnuclDiffMin < 0:  # vw is smaller than vmin, we return 0.
            return 0

        sol = root_scalar(
            shockTnuclDiff,
            bracket=(vmin, vmax),
            xtol=self.atol,
            rtol=self.rtol,
        )
        return float(sol.root)

    def _mappingT(self, TpTm: list[float]) -> list[float]:
        """
        Maps the variables Tp and Tm, which are constrained to
        TMinHydro < Tm,Tp < TMaxHydro to the interval (-inf,inf) to allow root
        finding algorithms to explore different values of (Tp,Tm), without going
        outside of the bounds above.

        Parameters
        ----------
        TpTm : array_like, shape (2,)
            List containing Tp and Tm.
        """

        Tp, Tm = TpTm
        mappedTm = np.tan(
            np.pi
            / (self.TMaxHydro - self.TMinHydro)
            * (Tm - (self.TMaxHydro + self.TMinHydro) / 2)
        )  # Maps Tm =TminGuess to -inf and Tm = TmaxGuess to inf
        mappedTp = np.tan(
            np.pi
            / (self.TMaxHydro - self.TMinHydro)
            * (Tp - (self.TMaxHydro + self.TMinHydro) / 2)
        )  # Maps Tp=TminGuess to -inf and Tp =TmaxGuess to +inf
        return [mappedTp, mappedTm]

    def _inverseMappingT(self, mappedTpTm: list[float]) -> list[float]:
        """
        Inverse of _mappingT.
        """

        mappedTp, mappedTm = mappedTpTm
        Tp = (
            np.arctan(mappedTp) * (self.TMaxHydro - self.TMinHydro) / np.pi
            + (self.TMaxHydro + self.TMinHydro) / 2
        )
        Tm = (
            np.arctan(mappedTm) * (self.TMaxHydro - self.TMinHydro) / np.pi
            + (self.TMaxHydro + self.TMinHydro) / 2
        )
        return [Tp, Tm]
