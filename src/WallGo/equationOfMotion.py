"""
Class for solving the EOM and the hydrodynamic equations.
"""

import warnings
from typing import Tuple
import copy  # for deepcopy
import numpy as np
import numpy.typing as npt

import scipy.optimize
from scipy.interpolate import UnivariateSpline

from .boltzmann import BoltzmannSolver
from .Fields import Fields, FieldPoint
from .Grid3Scales import Grid3Scales
from .helpers import gammaSq  # derivatives for callable functions
from .hydrodynamics import Hydrodynamics
from .Polynomial import Polynomial
from .thermodynamics import Thermodynamics
from .containers import (
    BoltzmannDeltas,
    BoltzmannBackground,
    WallParams,
)
from .results import (
    BoltzmannResults,
    HydroResults,
    WallGoResults,
    WallGoInterpolationResults,
)


class EOM:

    """
    Class that solves the energy-momentum conservation equations and the scalar
    EOMs to determine the wall velocity.
    """

    def __init__(
        self,
        boltzmannSolver: BoltzmannSolver,
        thermodynamics: Thermodynamics,
        hydrodynamics: Hydrodynamics,
        grid: Grid3Scales,
        nbrFields: int,
        meanFreePath: float,
        wallThicknessBounds: tuple[float, float],
        wallOffsetBounds: tuple[float, float],
        includeOffEq: bool = False,
        forceImproveConvergence: bool = False,
        errTol: float = 1e-3,
        maxIterations: int = 10,
        pressRelErrTol: float = 0.3679,
    ):
        """
        Initialization

        Parameters
        ----------
        boltzmannSolver : BoltzmannSolver
            BoltzmannSolver instance.
        thermodynamics : Thermodynamics
            Thermodynamics object
        hydrodynamics : Hydrodynamics
            Hydrodynamics object
        grid : Grid3Scales
            Object of the class Grid3Scales.
        nbrFields : int
            Number of scalar fields on which the scalar potential depends.
        meanFreePath : float
            Estimate of the mean free path of the particles in the plasma.
        wallThicknessBounds : tuple
            Tuple containing the bounds the wall thickness (in units of 1/Tnucl).
            The solver will never explore outside of this interval.
        wallOffsetBounds : tuple
            Tuple containing the bounds the wall offset. The solver will never
            explore outside of this interval.
        includeOffEq : bool, optional
            If False, all the out-of-equilibrium contributions are neglected.
            The default is False.
        forceImproveConvergence : bool, optional
            If True, uses a slower algorithm that improves the convergence when
            computing the pressure. The improved algorithm is automatically used
            for detonation. Default is False.
        errTol : float, optional
            Absolute error tolerance. The default is 1e-3.
        maxIterations : int, optional
            Maximum number of iterations for the convergence of pressure.
            The default is 10.
        pressRelErrTol : float, optional
            Relative tolerance in pressure when finding its root.

        Returns
        -------
        None.

        """

        assert isinstance(boltzmannSolver, BoltzmannSolver)
        assert isinstance(thermodynamics, Thermodynamics)
        assert isinstance(hydrodynamics, Hydrodynamics)
        assert isinstance(grid, Grid3Scales)
        assert (
            grid is boltzmannSolver.grid
        ), "EOM and BoltzmannSolver must have the same instance of the Grid object."

        self.boltzmannSolver = boltzmannSolver
        self.grid = grid
        self.nbrFields = nbrFields
        self.meanFreePath = meanFreePath
        self.wallThicknessBounds = wallThicknessBounds
        self.wallOffsetBounds = wallOffsetBounds
        self.includeOffEq = includeOffEq
        self.forceImproveConvergence = forceImproveConvergence

        self.thermo = thermodynamics
        self.hydrodynamics = hydrodynamics

        self.particles = self.boltzmannSolver.offEqParticles

        ## Tolerances
        self.errTol = errTol
        self.maxIterations = maxIterations
        self.pressRelErrTol = pressRelErrTol
        self.pressAbsErrTol = 0.0

    def findWallVelocityDeflagrationHybrid(
        self, wallThicknessIni: float | None = None
    ) -> WallGoResults:
        """
        Finds the wall velocity by minimizing the action and solving for the
        solution with 0 total pressure on the wall. This function only looks for
        deflagration or hybrid solutions. Returns a velocity of 1 if the pressure
        peak at vw = vJ is not large enough to stop the wall.
        For detonation solutions, use solveInterpolation().

        Parameters
        ----------
        wallThicknessIni : float or None, optional
            Initial thickness used for all the walls. If None, uses 5/Tnucl.
            Default is None.

        Returns
        -------
        WallGoResults
            WallGoResults object containing the solution of the EOM.

        """

        # If no initial wall thickness was provided, starts with a reasonable guess
        if wallThicknessIni is None:
            wallThicknessIni = 5 / self.thermo.Tnucl

        wallParams = WallParams(
            widths=wallThicknessIni * np.ones(self.nbrFields),
            offsets=np.zeros(self.nbrFields),
        )

        # In some cases, no deflagration solution can exist below or above some
        # velocity. That's why we need to look in the smaller interval [vmin,vmax]
        # (which is computed by Hydrodynamics) instead of the naive interval [0,vJ].
        vmin = self.hydrodynamics.vMin
        vmax = min(self.hydrodynamics.vJ, self.hydrodynamics.fastestDeflag())
        return self.solveWall(vmin, vmax, wallParams)

    def solveWall(
        self,
        wallVelocityMin: float,
        wallVelocityMax: float,
        wallParamsGuess: WallParams,
    ) -> WallGoResults:
        r"""
        Solves the equation :math:`P_{\rm tot}(\xi_w)=0` for the wall velocity
        and wall thicknesses/offsets. The solver only looks between wallVelocityMin
        and wallVelocityMax

        Parameters
        ----------
        wallVelocityMin : float
            Lower bound of the bracket in which the root finder will look for a
            solution. Should satisfy
            :math:`0<{\rm wallVelocityMin}<{\rm wallVelocityMax}`.
        wallVelocityMax : float
            Upper bound of the bracket in which the root finder will look for a
            solution. Should satisfy
            :math:`{\rm wallVelocityMin}<{\rm wallVelocityMax}\leq\xi_J`.
        wallParamsGuess : WallParams
            Contains a guess of the wall thicknesses and wall offsets.

        Returns
        -------
        results : WallGoResults
            Data class containing results.

        """
        results = WallGoResults()

        self.pressAbsErrTol = 1e-8

        # Get the pressure at vw = wallVelocityMax
        (
            pressureMax,
            wallParamsMax,
            boltzmannResultsMax,
            boltzmannBackgroundMax,
            hydroResultsMax,
        ) = self.wallPressure(wallVelocityMax, wallParamsGuess)

        # also getting the LTE results
        wallVelocityLTE = self.hydrodynamics.findvwLTE()

        # The pressure peak is not enough to stop the wall: no deflagration or
        # hybrid solution
        if pressureMax < 0:
            print("Maximum pressure on wall is negative!")
            print(f"{pressureMax=} {wallParamsMax=}")
            results.setWallVelocities(1, 0, wallVelocityLTE)
            results.setWallParams(wallParamsMax)
            results.setHydroResults(hydroResultsMax)
            results.setBoltzmannBackground(boltzmannBackgroundMax)
            results.setBoltzmannResults(boltzmannResultsMax)
            return results

        # Get the pressure at vw = wallVelocityMin
        (
            pressureMin,
            wallParamsMin,
            boltzmannResultsMin,
            boltzmannBackgroundMin,
            hydroResultsMin,
        ) = self.wallPressure(wallVelocityMin, wallParamsGuess)
        if pressureMin > 0:
            print(
                """EOM warning: the pressure at vw = 0 is positive which indicates the 
                phase transition cannot proceed. Something might be wrong with your 
                potential."""
            )
            results.setWallVelocities(0, 0, wallVelocityLTE)
            results.setWallParams(wallParamsMin)
            results.setHydroResults(hydroResultsMin)
            results.setBoltzmannBackground(boltzmannBackgroundMin)
            results.setBoltzmannResults(boltzmannResultsMin)
            return results

        self.pressAbsErrTol = (
            0.01
            * self.errTol
            * (1 - self.pressRelErrTol)
            * np.minimum(np.abs(pressureMin), np.abs(pressureMax))
            / 4
        )

        ## This computes pressure on the wall with a given wall speed and WallParams
        def pressureWrapper(vw: float) -> float:  # pylint: disable=invalid-name
            """Small optimization here: the root finder below calls this first at the
            bracket endpoints, for which we already computed the pressure above.
            So make use of those.
            """
            if np.abs(vw - wallVelocityMin) < 1e-10 or vw < wallVelocityMin:
                return pressureMin
            if np.abs(vw - wallVelocityMax) < 1e-10 or vw > wallVelocityMax:
                return pressureMax

            # Use linear interpolation to get a better first guess for the initial wall
            # parameters
            fractionVw = (vw - wallVelocityMin) / (wallVelocityMax - wallVelocityMin)
            newWallParams = wallParamsMin + (wallParamsMax - wallParamsMin) * fractionVw
            return self.wallPressure(vw, newWallParams)[0]

        optimizeResult = scipy.optimize.root_scalar(
            pressureWrapper,
            method="brentq",
            bracket=[wallVelocityMin, wallVelocityMax],
            xtol=self.errTol,
        )
        wallVelocity = optimizeResult.root

        # Get wall params, and other results
        fractionWallVelocity = (wallVelocity - wallVelocityMin) / (
            wallVelocityMax - wallVelocityMin
        )
        newWallParams = (
            wallParamsMin + (wallParamsMax - wallParamsMin) * fractionWallVelocity
        )
        (
            _,
            wallParams,
            boltzmannResults,
            boltzmannBackground,
            hydroResults,
        ) = self.wallPressure(
            wallVelocity,
            newWallParams,
        )

        # minimum possible error in the wall speed
        wallVelocityMinError = self.errTol * optimizeResult.root

        # estimating errors from truncation and comparison to finite differences
        if self.includeOffEq:
            finiteDifferenceBoltzmannResults = self.getBoltzmannFiniteDifference()
            # assuming nonequilibrium errors proportional to deviation from LTE
            wallVelocityDeltaLTE = abs(wallVelocity - wallVelocityLTE)
            # the truncation error in the spectral method within Boltzmann
            wallVelocityTruncationError = (
                boltzmannResults.truncationError * wallVelocityDeltaLTE
            )
            # the deviation from the finite difference method within Boltzmann
            delta00 = boltzmannResults.Deltas.Delta00.coefficients[0]
            delta00FD = finiteDifferenceBoltzmannResults.Deltas.Delta00.coefficients[0]
            errorFD = np.linalg.norm(delta00 - delta00FD) / np.linalg.norm(delta00)
            wallVelocityDerivativeError = errorFD * wallVelocityDeltaLTE

            # if truncation waringin large, raise a warning
            if (
                wallVelocityTruncationError > wallVelocityDerivativeError
                and wallVelocityTruncationError > self.errTol
            ):
                warnings.warn("Truncation error large, increase N or M", RuntimeWarning)

            # estimating the error by the largest of these
            wallVelocityError = max(
                wallVelocityMinError,
                wallVelocityTruncationError,
            )
        else:
            finiteDifferenceBoltzmannResults = boltzmannResults
            wallVelocityError = wallVelocityMinError

        # setting results
        results.setWallVelocities(
            wallVelocity=wallVelocity,
            wallVelocityError=wallVelocityError,
            wallVelocityLTE=wallVelocityLTE,
        )

        results.setHydroResults(hydroResults)
        results.setWallParams(wallParams)
        results.setBoltzmannBackground(boltzmannBackground)
        results.setBoltzmannResults(boltzmannResults)
        results.setFiniteDifferenceBoltzmannResults(finiteDifferenceBoltzmannResults)

        # return collected results
        return results

    def wallPressure(
        self,
        wallVelocity: float,
        wallParams: WallParams,
        atol: float | None = None,
        rtol: float | None = None,
        boltzmannResultsInput: BoltzmannResults | None = None,
    ) -> tuple[float, WallParams, BoltzmannResults, BoltzmannBackground, HydroResults]:
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action. Can use two different iteration algorithms
        to find the pressure. If self.forceImproveConvergence=False and
        wallVelocity<self.hydrodynamics.vJ, uses a fast algorithm that sometimes fails
        to converge. Otherwise, or if the previous algorithm converges slowly,
        uses a slower, but more robust algorithm.

        Parameters
        ----------
        wallVelocity : float
            Wall velocity at which the pressure is computed.
        wallParams : WallParams
            Contains a guess of the wall thicknesses and wall offsets.
        atol : float or None, optional
            Absolute tolerance. If None, uses self.pressAbsErrTol. Default is None.
        rtol : float or None, optional
            Relative tolerance. If None, uses self.pressRelErrTol. Default is None.
        boltzmannResultsInput : BoltzmannResults or None, optional
            Object of the BoltzmannResults class containing the initial solution
            of the Boltzmann equation. If None, sets the initial deltaF to 0.
            Default is None.

        Returns
        -------
        pressure : float
            Total pressure on the wall.
        wallParams : WallParams
            WallParams object containing the wall thicknesses and wall offsets
            that minimize the action and solve the EOM. Only returned if
            returnExtras is True.
        boltzmannResults : BoltzmannResults
            BoltzmannResults object containing the solution of the Boltzmann
            equation. Only returned if returnExtras is True
        boltzmannBackground : BoltzmannBackground
            BoltzmannBackground object containing the solution of the hydrodynamic
            equations and the scalar field profiles. Only returned if returnExtras
            is True.
        hydroResults : HydroResults
            HydroResults object containing the solution obtained from Hydrodynamics.
            Only returned if returnExtras is True

        """

        if atol is None:
            atol = self.pressAbsErrTol
        if rtol is None:
            rtol = self.pressRelErrTol

        improveConvergence = self.forceImproveConvergence
        if wallVelocity > self.hydrodynamics.vJ:
            improveConvergence = True

        print(f"\nTrying {wallVelocity=}")

        # Initialize the different data class objects and arrays
        zeroPoly = Polynomial(
            np.zeros((len(self.particles), self.grid.M - 1)),
            self.grid,
            direction=("Array", "z"),
            basis=("Array", "Cardinal"),
        )
        offEquilDeltas = BoltzmannDeltas(
            Delta00=zeroPoly,
            Delta02=zeroPoly,
            Delta20=zeroPoly,
            Delta11=zeroPoly,
        )
        deltaF = Polynomial(
            np.zeros(
                (
                    len(self.particles),
                    (self.grid.M - 1),
                    (self.grid.N - 1),
                    (self.grid.N - 1),
                )
            ),
            self.grid,
            basis=("Array", "Cardinal", "Chebyshev", "Chebyshev"),
            direction=("Array", "z", "pz", "pp"),
            endpoints=False,
        )

        boltzmannResults: BoltzmannResults
        if boltzmannResultsInput is None:
            boltzmannResults = BoltzmannResults(
                deltaF=deltaF,
                Deltas=offEquilDeltas,
                truncationError=0.0,
                linearizationCriterion1=0.0,
                linearizationCriterion2=0.0,
            )
        else:
            boltzmannResults = boltzmannResultsInput

        # Find the boundary conditions of the hydrodynamic equations
        (
            c1,
            c2,
            Tplus,
            Tminus,
            velocityMid,
        ) = self.hydrodynamics.findHydroBoundaries(  # pylint: disable=invalid-name
            wallVelocity
        )
        hydroResults = HydroResults(
            temperaturePlus=Tplus,
            temperatureMinus=Tminus,
            velocityJouguet=self.hydrodynamics.vJ,
        )

        # Positions of the phases
        vevLowT = self.thermo.freeEnergyLow(Tminus).fieldsAtMinimum
        vevHighT = self.thermo.freeEnergyHigh(Tplus).fieldsAtMinimum

        ##Estimate the new grid parameters
        widths = wallParams.widths
        offsets = wallParams.offsets
        ## Distance between the right and left edges of the walls at the boundaries
        wallThicknessGrid = (
            np.max((1 - offsets) * widths) - np.min((-1 - offsets) * widths)
        ) / 2
        ## Center between these two edges
        ## The source and pressure are proportional to d(m^2)/dz, which peaks at
        ## -wallThicknessGrid*np.log(2)/2. This is why we substract this value.
        wallCenterGrid = (
            np.max((1 - offsets) * widths) + np.min((-1 - offsets) * widths)
        ) / 2 - wallThicknessGrid * np.log(2) / 2
        gammaWall = 1 / np.sqrt(1 - velocityMid**2)
        """ The length of the tail inside typically scales like gamma, while the one
        outside like 1/gamma. We take the max because the tail lengths must be larger
        than wallThicknessGrid*(1+2*smoothing)/ratioPointsWall """
        tailInside = max(
            self.meanFreePath * gammaWall * self.includeOffEq,
            wallThicknessGrid
            * (1 + 2.1 * self.grid.smoothing)
            / self.grid.ratioPointsWall,
        )
        tailOutside = max(
            self.meanFreePath / gammaWall * self.includeOffEq,
            wallThicknessGrid
            * (1 + 2.1 * self.grid.smoothing)
            / self.grid.ratioPointsWall,
        )
        self.grid.changePositionFalloffScale(
            tailInside, tailOutside, wallThicknessGrid, wallCenterGrid
        )

        (
            pressure,
            wallParams,
            boltzmannResults,
            boltzmannBackground,
        ) = self._intermediatePressureResults(
            wallParams,
            vevLowT,
            vevHighT,
            c1,
            c2,
            velocityMid,
            boltzmannResults,
            Tplus,
            Tminus,
        )

        pressures = [pressure]

        """
        The 'multiplier' parameter is used to reduce the size of the wall
        parameters updates during the iteration procedure. The next iteration
        will use multiplier*newWallParams+(1-multiplier)*oldWallParams.
        Can be used when the iterations do not converge, even close to the
        true solution. For small enough values, we should always be able to converge.
        The value will be reduced if the algorithm doesn't converge.
        """
        multiplier = 1.0

        i = 0
        while True:
            if improveConvergence:
                # Use the improved algorithm (which converges better but slowly)
                (
                    pressure,
                    wallParams,
                    boltzmannResults,
                    boltzmannBackground,
                    errorSolver,
                ) = self._getNextPressure(
                    pressure,
                    wallParams,
                    vevLowT,
                    vevHighT,
                    c1,
                    c2,
                    velocityMid,
                    boltzmannResults,
                    Tplus,
                    Tminus,
                    multiplier=multiplier,
                )
            else:
                (
                    pressure,
                    wallParams,
                    boltzmannResults,
                    boltzmannBackground,
                ) = self._intermediatePressureResults(
                    wallParams,
                    vevLowT,
                    vevHighT,
                    c1,
                    c2,
                    velocityMid,
                    boltzmannResults,
                    Tplus,
                    Tminus,
                    multiplier=multiplier,
                )
                errorSolver = 0
            pressures.append(pressure)

            error = np.abs(pressures[-1] - pressures[-2])
            errTol = np.maximum(rtol * np.abs(pressure), atol) * multiplier

            print(f"{pressure=} {error=} {errTol=} {improveConvergence=} {multiplier=}")
            i += 1

            if error < errTol or (errorSolver < errTol and improveConvergence):
                """
                Even if two consecutive call to _getNextPressure() give similar
                pressures, it is possible that the internal calls made to
                _intermediatePressureResults() do not converge. This is measured
                by 'errorSolver'. If _getNextPressure() converges but
                _intermediatePressureResults() doesn't, 'multiplier' is probably too
                large. We therefore continue the iteration procedure with a smaller
                value of 'multiplier'.
                """
                if errorSolver > errTol:
                    multiplier /= 2.0
                else:
                    break
            elif i >= self.maxIterations - 1:
                print(
                    "Pressure for a wall velocity has not converged to "
                    "sufficient accuracy with the given maximum number "
                    "for iterations."
                )
                break

            if len(pressures) > 2:
                if error > abs(pressures[-2] - pressures[-3]) / 1.5:
                    # If the error decreases too slowly, use the improved algorithm
                    improveConvergence = True

        return (
            pressure,
            wallParams,
            boltzmannResults,
            boltzmannBackground,
            hydroResults,
        )

    def _getNextPressure(
        self,
        pressure1: float,
        wallParams1: WallParams,
        vevLowT: Fields,
        vevHighT: Fields,
        c1: float,  # pylint: disable=invalid-name
        c2: float,  # pylint: disable=invalid-name
        velocityMid: float,
        boltzmannResults1: BoltzmannResults,
        Tplus: float,
        Tminus: float,
        temperatureProfile: np.ndarray | None = None,
        velocityProfile: np.ndarray | None = None,
        multiplier: float = 1.0,
    ) -> tuple:
        """
        Performs the next iteration to solve the EOM and Boltzmann equation.
        First computes the pressure twice, updating the wall parameters and
        Boltzmann results each time. If the iterations overshot the true solution
        (the two pressure updates go in opposite directions), uses linear
        interpolation to find a better estimate of the true solution. This function is
        called only when improveConvergence=True in wallPressure().
        """
        (
            pressure2,
            wallParams2,
            boltzmannResults2,
            _,
        ) = self._intermediatePressureResults(
            wallParams1,
            vevLowT,
            vevHighT,
            c1,
            c2,
            velocityMid,
            boltzmannResults1,
            Tplus,
            Tminus,
            temperatureProfile,
            velocityProfile,
            multiplier,
        )
        (
            pressure3,
            wallParams3,
            boltzmannResults3,
            boltzmannBackground3,
        ) = self._intermediatePressureResults(
            wallParams2,
            vevLowT,
            vevHighT,
            c1,
            c2,
            velocityMid,
            boltzmannResults2,
            Tplus,
            Tminus,
            temperatureProfile,
            velocityProfile,
            multiplier,
        )

        ## If the last iteration does not overshoot the real pressure (the two
        ## last update go in the same direction), returns the last iteration.
        if (pressure3 - pressure2) * (pressure2 - pressure1) >= 0:
            err = abs(pressure3 - pressure2)
            return pressure3, wallParams3, boltzmannResults3, boltzmannBackground3, err

        ## If the last iteration overshot, uses linear interpolation to find a
        ## better estimate of the true solution.
        interpPoint = (pressure1 - pressure2) / (pressure1 - 2 * pressure2 + pressure3)
        (
            pressure4,
            wallParams4,
            boltzmannResults4,
            boltzmannBackground4,
        ) = self._intermediatePressureResults(
            wallParams1 + (wallParams2 - wallParams1) * interpPoint,
            vevLowT,
            vevHighT,
            c1,
            c2,
            velocityMid,
            boltzmannResults1 + (boltzmannResults2 - boltzmannResults1) * interpPoint,
            Tplus,
            Tminus,
            temperatureProfile,
            velocityProfile,
            multiplier,
        )
        err = abs(pressure4 - pressure2)
        return pressure4, wallParams4, boltzmannResults4, boltzmannBackground4, err

    def _intermediatePressureResults(
        self,
        wallParams: WallParams,
        vevLowT: Fields,
        vevHighT: Fields,
        c1: float,  # pylint: disable=invalid-name
        c2: float,  # pylint: disable=invalid-name
        velocityMid: float,
        boltzmannResults: BoltzmannResults,
        Tplus: float,
        Tminus: float,
        temperatureProfileInput: np.ndarray | None = None,
        velocityProfileInput: np.ndarray | None = None,
        multiplier: float = 1.0,
    ) -> tuple[float, WallParams, BoltzmannResults, BoltzmannBackground]:
        """
        Performs one step of the iteration procedure to update the pressure,
        wall parameters and Boltzmann solution. This is done by first solving
        the Boltzmann equation and then minimizing the action to solve the EOM.
        """

        ## here dfieldsdz are z-derivatives of the fields
        fields, dfieldsdz = self.wallProfile(
            self.grid.xiValues, vevLowT, vevHighT, wallParams
        )

        temperatureProfile: np.ndarray
        velocityProfile: np.ndarray
        if temperatureProfileInput is None or velocityProfileInput is None:
            temperatureProfile, velocityProfile = self.findPlasmaProfile(
                c1,
                c2,
                velocityMid,
                fields,
                dfieldsdz,
                boltzmannResults.Deltas,
                Tplus,
                Tminus,
            )
        else:
            temperatureProfile = temperatureProfileInput
            velocityProfile = velocityProfileInput

        ## Prepare a new background for Boltzmann
        TWithEndpoints: np.ndarray = np.concatenate(
            (np.array([Tminus]), np.array(temperatureProfile), np.array([Tplus]))
        )
        fieldsWithEndpoints: Fields = np.concatenate(
            (vevLowT, fields, vevHighT), axis=fields.overFieldPoints
        ).view(Fields)
        vWithEndpoints: np.ndarray = np.concatenate(
            (
                np.array([velocityProfile[0]]),
                np.array(velocityProfile),
                np.array([velocityProfile[-1]]),
            )
        )
        boltzmannBackground = BoltzmannBackground(
            velocityMid,
            vWithEndpoints,
            fieldsWithEndpoints,
            TWithEndpoints,
        )
        if self.includeOffEq:
            ## ---- Solve Boltzmann equation to get out-of-equilibrium contributions
            self.boltzmannSolver.setBackground(boltzmannBackground)
            boltzmannResults = (
                multiplier * self.boltzmannSolver.getDeltas()
                + (1 - multiplier) * boltzmannResults
            )

        # Need to solve wallWidth and wallOffset. For this, put wallParams in a 1D array
        # NOT including the first offset which we keep at 0.
        wallArray: np.ndarray = np.concatenate(
            (wallParams.widths, wallParams.offsets[1:])
        )  ## should work even if offsets is just 1 element

        ## first width, then offset
        lowerBounds: np.ndarray = np.concatenate(
            (
                self.nbrFields * [self.wallThicknessBounds[0] / self.thermo.Tnucl],
                (self.nbrFields - 1) * [self.wallOffsetBounds[0]],
            )
        )
        upperBounds: np.ndarray = np.concatenate(
            (
                self.nbrFields * [self.wallThicknessBounds[1] / self.thermo.Tnucl],
                (self.nbrFields - 1) * [self.wallOffsetBounds[1]],
            )
        )
        bounds = scipy.optimize.Bounds(lb=lowerBounds, ub=upperBounds)

        ## And then a wrapper that puts the inputs back in WallParams
        def actionWrapper(
            wallArray: np.ndarray, *args: Fields | npt.ArrayLike | Polynomial
        ) -> float:
            return self.action(self._toWallParams(wallArray), *args)

        Delta00 = boltzmannResults.Deltas.Delta00  # pylint: disable=invalid-name
        sol = scipy.optimize.minimize(
            actionWrapper,
            wallArray,
            args=(vevLowT, vevHighT, temperatureProfile, Delta00),
            method="Nelder-Mead",
            bounds=bounds,
        )

        ## Put the resulting width, offset back in WallParams format
        wallParams = (
            multiplier * self._toWallParams(sol.x) + (1 - multiplier) * wallParams
        )

        fields, dPhidz = self.wallProfile(
            self.grid.xiValues, vevLowT, vevHighT, wallParams
        )
        dVdPhi = self.thermo.effectivePotential.derivField(fields, temperatureProfile)

        # Out-of-equilibrium term of the EOM
        dVout = (
            np.sum(
                [
                    particle.totalDOFs
                    * particle.msqDerivative(fields)
                    * Delta00.coefficients[i, :, None]
                    for i, particle in enumerate(self.particles)
                ],
                axis=0,
            )
            / 2
        )

        ## EOM for field i is d^2 phi_i + dVfull == 0, the latter term is dVdPhi + dVout
        dVfull: Fields = dVdPhi + dVout

        dVdz = np.sum(np.array(dVfull * dPhidz), axis=1)

        # Create a Polynomial object to represent dVdz. Will be used to integrate it.
        eomPoly = Polynomial(dVdz, self.grid)

        dzdchi, _, _ = self.grid.getCompactificationDerivatives()
        pressure = eomPoly.integrate(w=-dzdchi)

        return pressure, wallParams, boltzmannResults, boltzmannBackground

    def gridPressure(
        self,
        vmin: float,
        vmax: float,
        nbrPoints: int,
        wallThicknessIni: float | None = None,
        rtol: float = 1e-3,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        list[WallParams],
        list[BoltzmannResults],
        list[BoltzmannBackground],
        list[HydroResults],
    ]:
        """
        Computes the pressure on a linearly spaced grid of velocities between
        vmin and vmax. Typically used to find detonation solutions.

        Parameters
        ----------
        vmin : float
            Lower bound of the interpolation interval.
        vmax : float
            Upper bound of the interpolation interval.
        nbrPoints : int
            Number of points on the grid.
        wallThicknessIni : float or None, optional
            Initial wall thickness used to compute the first pressure at vmin. If None,
            uses 5/Tnucl. The default is None.
        rtol : float, optional
            Relative tolerance. The default is 1e-3.

        Returns
        -------
        wallVelocities : ndarray
            Velocity grid.
        pressures: ndarray
            Pressure evaluated on the grid.
        wallParamsList : list[WallParams]
            WallParams evaluated on the grid.
        boltzmannResultsList : list[BoltzmannResults]
            BoltzmannResults evaluated on the grid.
        boltzmannBackgroundList : list[BoltzmannBackground]
            BoltzmannBackground evaluated on the grid.
        hydroResultsList : list[HydroResults]
            HydroResults evaluated on the grid.

        """
        # Defining the velocity grid
        wallVelocities = np.linspace(vmin, vmax, nbrPoints)

        # Initializing the wall parameters
        if wallThicknessIni is None:
            wallThicknessIni = 5 / self.thermo.Tnucl

        wallParams = WallParams(
            widths=wallThicknessIni * np.ones(self.nbrFields),
            offsets=np.zeros(self.nbrFields),
        )

        boltzmannResults = None

        pressure, wallParams, boltzmannResults, _, hydroResults = self.wallPressure(
            vmin, wallParams, 0, rtol, boltzmannResults
        )

        pressures: list[float] = []
        boltzmannBackgroundList: list[BoltzmannBackground] = []
        boltzmannResultsList: list[BoltzmannResults] = []
        hydroResultsList: list[HydroResults] = []
        wallParamsList: list[WallParams] = []
        # Computing the pressure on the velocity grid
        for i, wallVelocity in enumerate(wallVelocities):
            if i > 1:
                # Use linear extrapolation to get a more accurate initial value of wall
                # parameters
                wallParamsTry = wallParamsList[-1] + (
                    wallParamsList[-1] - wallParamsList[-2]
                ) * (wallVelocity - wallVelocities[i - 1]) / (
                    wallVelocities[i - 1] - wallVelocities[i - 2]
                )
                boltzmannResultsTry = boltzmannResultsList[-1] + (
                    boltzmannResultsList[-1] - boltzmannResultsList[-2]
                ) * (
                    (wallVelocity - wallVelocities[i - 1])
                    / (wallVelocities[i - 1] - wallVelocities[i - 2])
                )
            else:
                wallParamsTry = wallParams
                boltzmannResultsTry = boltzmannResults

            (
                pressure,
                wallParams,
                boltzmannResults,
                boltzmannBackground,
                hydroResults,
            ) = self.wallPressure(
                wallVelocity, wallParamsTry, 0, rtol, boltzmannResultsTry
            )

            pressures.append(pressure)
            wallParamsList.append(wallParams)
            boltzmannResultsList.append(boltzmannResults)
            boltzmannBackgroundList.append(boltzmannBackground)
            hydroResultsList.append(hydroResults)

        return (
            wallVelocities,
            np.array(pressures),
            wallParamsList,
            boltzmannResultsList,
            boltzmannBackgroundList,
            hydroResultsList,
        )

    def solveInterpolation(
        self,
        vmin: float,
        vmax: float,
        wallThicknessIni: float | None = None,
        desiredPressure: float = 0.0,
        rtol: float = 1e-3,
        dvMin: float = 0.02,
    ) -> WallGoInterpolationResults:
        """
        Finds all the EOM solutions in some interval by computing the pressure
        on a grid and interpolating to get the roots.

        Parameters
        ----------
        vmin : float
            Lower bound of the interpolation interval.
        vmax : float
            Upper bound of the interpolation interval.
        wallThicknessIni : float or None, optional
            Initial wall thickness used to compute the first pressure at vmin. If None,
            uses 5/Tnucl. The default is None.
        desiredPressure : float, optional
            The solver finds the velocities for which the pressure is equal to
            desiredPressure. The default is 0.
        rtol : float, optional
            Relative tolerance. The default is 1e-3.
        dvMin : float, optional
            Minimal spacing between each grid points. The default is 0.02.

        Returns
        -------
        wallGoInterpolationResults : WallGoInterpolationResults

        """
        if vmin < 0.99:
            """
            Chooses the number of points on the grid to reach the tolerance goal
            assuming the spline error scales as Delta v^4. Always uses at least
            5 points for the spline to be accurate.
            """
            nbrPoints = max(1 + int((vmax - vmin) / min(dvMin, rtol**0.25)), 5)
            # Computing the pressure on the grid
            (
                wallVelocities,
                pressures,
                wallParamsList,
                boltzmannResultsList,
                boltzmannBackgroundList,
                hydroResultsList,
            ) = self.gridPressure(vmin, vmax, nbrPoints, wallThicknessIni, rtol)
            # Splining the result
            pressuresSpline = UnivariateSpline(
                wallVelocities, pressures - desiredPressure, s=0.0
            )

            # Finding the roots of the spline and classifying the result as stable or
            # unstable solutions
            roots = pressuresSpline.roots()
            stableRoots, unstableRoots = [], []
            for root in roots:
                if pressuresSpline.derivative()(root) > 0:
                    stableRoots.append(root)
                else:
                    unstableRoots.append(root)

            # Storing the result in a WallGoInterpolationResults class
            wallGoInterpolationResults = WallGoInterpolationResults(
                wallVelocities=stableRoots,
                unstableWallVelocities=unstableRoots,
                velocityGrid=wallVelocities.tolist(),
                pressures=pressures.tolist(),
                pressureSpline=pressuresSpline,
                wallParams=wallParamsList,
                boltzmannResults=boltzmannResultsList,
                boltzmannBackgrounds=boltzmannBackgroundList,
                hydroResults=hydroResultsList,
            )
            return wallGoInterpolationResults

        wallGoInterpolationResults = WallGoInterpolationResults(
            wallVelocities=[],
            unstableWallVelocities=[],
            velocityGrid=[],
            pressures=[],
            pressureSpline=[],
            wallParams=[],
            boltzmannResults=[],
            boltzmannBackgrounds=[],
            hydroResults=[],
        )
        return wallGoInterpolationResults

    def _toWallParams(self, wallArray: np.ndarray) -> WallParams:
        offsets: np.ndarray = np.concatenate(
            (np.array([0.0]), wallArray[self.nbrFields :])
        )
        return WallParams(widths=wallArray[: self.nbrFields], offsets=offsets)

    def action(
        self,
        wallParams: WallParams,
        vevLowT: Fields,
        vevHighT: Fields,
        temperatureProfile: np.ndarray,
        offEquilDelta00: Polynomial,
    ) -> float:
        """
        Computes the action by using gaussian quadratrure to integrate the Lagrangian.

        Parameters
        ----------
        wallParams : WallParams
            WallParams object.
        vevLowT : Fields
            Field values in the low-T phase.
        vevHighT : Fields
            Field values in the high-T phase.
        temperatureProfile : ndarray
            Temperature profile on the grid.
        offEquilDelta00 : Polynomial
            Off-equilibrium function Delta00.

        Returns
        -------
        action : float
            Action spent by the scalar field configuration.

        """

        wallWidths = wallParams.widths

        # Computing the field profiles
        fields = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)[0]

        # Computing the potential
        potential = self.thermo.effectivePotential.evaluate(fields, temperatureProfile)

        # Computing the out-of-equilibrium term of the action
        potentialOut = (
            sum(
                [
                    particle.totalDOFs
                    * particle.msqVacuum(fields)
                    * offEquilDelta00.coefficients[i]
                    for i, particle in enumerate(self.particles)
                ]
            )
            / 2
        )

        # Values of the potential at the boundaries
        potentialLowT = self.thermo.effectivePotential.evaluate(
            vevLowT, temperatureProfile[0]
        )
        potentialHighT = self.thermo.effectivePotential.evaluate(
            vevHighT, temperatureProfile[-1]
        )

        potentialRef = (np.array(potentialLowT) + np.array(potentialHighT)) / 2.0

        # Integrating the potential to get the action
        # We substract Vref (which has no effect on the EOM) to make the integral finite
        potentialPoly = Polynomial(potential + potentialOut - potentialRef, self.grid)
        dzdchi, _, _ = self.grid.getCompactificationDerivatives()

        # Potential energy part of the action
        U = potentialPoly.integrate(w=dzdchi)  # pylint: disable=invalid-name
        # Kinetic part of the action
        K = np.sum(  # pylint: disable=invalid-name
            (vevHighT - vevLowT) ** 2 / (6 * wallWidths)
        )

        return float(U + K)

    def wallProfile(
        self,
        z: np.ndarray,  # pylint: disable=invalid-name
        vevLowT: Fields,
        vevHighT: Fields,
        wallParams: WallParams,
    ) -> Tuple[Fields, Fields]:
        """
        Computes the scalar field profile and its derivative with respect to
        the position in the wall.

        Parameters
        ----------
        z : ndarray
            Position grid on which to compute the scalar field profile.
        vevLowT : Fields
            Scalar field VEVs in the low-T phase.
        vevHighT : Fields
            Scalar field VEVs in the high-T phase.
        wallParams : WallParams
            WallParams object.

        Returns
        -------
        fields : Fields
            Scalar field profile.
        dPhidz : Fields
            Derivative with respect to the position of the scalar field profile.

        """

        if np.isscalar(z):
            zL = z / wallParams.widths  # pylint: disable=invalid-name
        else:
            ## Broadcast mess needed
            zL = z[:, None] / wallParams.widths[None, :]  # pylint: disable=invalid-name

        fields = vevLowT + 0.5 * (vevHighT - vevLowT) * (
            1 + np.tanh(zL + wallParams.offsets)
        )
        dPhidz = (
            0.5
            * (vevHighT - vevLowT)
            / (wallParams.widths * np.cosh(zL + wallParams.offsets) ** 2)
        )

        return Fields.CastFromNumpy(fields), Fields.CastFromNumpy(dPhidz)

    def findPlasmaProfile(
        self,
        c1: float,  # pylint: disable=invalid-name
        c2: float,  # pylint: disable=invalid-name
        velocityMid: float,
        fields: Fields,
        dPhidz: Fields,
        offEquilDeltas: BoltzmannDeltas,
        Tplus: float,
        Tminus: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of
        LHS.

        Parameters
        ----------
        c1 : float
            Value of the :math:`T^{30}` component of the energy-momentum tensor.
        c2 : float
            Value of the :math:`T^{33}` component of the energy-momentum tensor.
        velocityMid : float
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        fields : Fields
            Scalar field profiles.
        dPhidz : Fields
            Derivative with respect to the position of the scalar field profiles.
        offEquilDeltas : BoltzmannDeltas
            BoltzmannDeltas object containing the off-equilibrium Delta functions
        Tplus : float
            Plasma temperature in front of the wall.
        Tminus : float
            Plasma temperature behind the wall.

        Returns
        -------
        temperatureProfile : array-like
            Temperature profile in the wall.
        velocityProfile : array-like
            Plasma velocity profile in the wall.

        """
        temperatureProfile = np.zeros(len(self.grid.xiValues))
        velocityProfile = np.zeros(len(self.grid.xiValues))

        for index in range(len(self.grid.xiValues)):
            T, vPlasma = self.findPlasmaProfilePoint(
                index,
                c1,
                c2,
                velocityMid,
                fields.GetFieldPoint(index),
                dPhidz.GetFieldPoint(index),
                offEquilDeltas,
                Tplus,
                Tminus,
            )

            temperatureProfile[index] = T
            velocityProfile[index] = vPlasma

        return temperatureProfile, velocityProfile

    def findPlasmaProfilePoint(
        self,
        index: int,
        c1: float,  # pylint: disable=invalid-name
        c2: float,  # pylint: disable=invalid-name
        velocityMid: float,
        fields: FieldPoint,
        dPhidz: FieldPoint,
        offEquilDeltas: BoltzmannDeltas,
        Tplus: float,
        Tminus: float,
    ) -> Tuple[float, float]:
        r"""
        Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution,
        the minimum of LHS.

        Parameters
        ----------
        index : int
            Index of the grid point on which to find the plasma profile.
        c1 : float
            Value of the :math:`T^{30}` component of the energy-momentum tensor.
        c2 : float
            Value of the :math:`T^{33}` component of the energy-momentum tensor.
        velocityMid : float
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        fields : FieldPoint
            Scalar field profile.
        dPhidz : FieldPoint
            Derivative with respect to the position of the scalar field profile.
        offEquilDeltas : BoltzmannDeltas
            BoltzmannDeltas object containing the off-equilibrium Delta functions
        Tplus : float
            Plasma temperature in front of the wall.
        Tminus : float
            Plasma temperature behind the wall.

        Returns
        -------
        T : float
            Temperature at the point grid.xiValues[index].
        vPlasma : float
            Plasma velocity at the point grid.xiValues[index].

        """

        # Computing the out-of-equilibrium part of the energy-momentum tensor
        Tout30, Tout33 = self.deltaToTmunu(index, fields, velocityMid, offEquilDeltas)
        s1 = c1 - Tout30  # pylint: disable=invalid-name
        s2 = c2 - Tout33  # pylint: disable=invalid-name

        """
        The function we want to solve look in general like a parabola. In particular,
        it has two solutions, one deflagration and one detonation. To solve it,
        we first find the parabola's minimum, and then select the desired 
        solution on either side of the minimum.
        """
        minRes = scipy.optimize.minimize_scalar(
            lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2),
            method="Bounded",
            bounds=[0, 2 * max(Tplus, Tminus)],
        )

        # If the minimum is positive, there are no roots and we return the
        # minimum's position
        if self.temperatureProfileEqLHS(fields, dPhidz, minRes.x, s1, s2) >= 0:
            T = minRes.x
            vPlasma = self.plasmaVelocity(fields, T, s1)
            return T, vPlasma

        # Bracketing the root
        tempAtMinimum = minRes.x
        TMultiplier = max(Tplus / tempAtMinimum, 1.2)
        if (
            Tplus < Tminus
        ):  # If this is a detonation solution, finds a solution below TLowerBound
            TMultiplier = min(Tminus / tempAtMinimum, 0.8)

        testTemp = tempAtMinimum * TMultiplier
        while self.temperatureProfileEqLHS(fields, dPhidz, testTemp, s1, s2) < 0:
            tempAtMinimum *= TMultiplier
            testTemp *= TMultiplier

        # Solving for the root
        res = scipy.optimize.root_scalar(
            lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2),
            bracket=(tempAtMinimum, testTemp),
            xtol=1e-10,
            rtol=self.errTol / 10,
        ).root

        T = res
        vPlasma = self.plasmaVelocity(fields, T, s1)
        return T, vPlasma

    def plasmaVelocity(
        self, fields: FieldPoint, T: float, s1: float  # pylint: disable=invalid-name
    ) -> float:
        r"""
        Computes the plasma velocity as a function of the temperature.

        Parameters
        ----------
        fields : FieldPoint
            Scalar field profiles.
        T : float
            Temperature.
        s1 : float
            Value of :math:`T^{30}-T_{\rm out}^{30}`.

        Returns
        -------
        float
            Plasma velocity.

        """
        # Need enthalpy ouside a free-energy minimum (eq .(12) in arXiv:2204.13120v1)
        enthalpy = -T * self.thermo.effectivePotential.derivT(fields, T)

        return float((-enthalpy + np.sqrt(4 * s1**2 + enthalpy**2)) / (2 * s1))

    def temperatureProfileEqLHS(
        self,
        fields: FieldPoint,
        dPhidz: FieldPoint,
        T: float,
        s1: float,
        s2: float,  # pylint: disable=invalid-name
    ) -> float:
        r"""
        The LHS of Eq. (20) of arXiv:2204.13120v1.

        Parameters
        ----------
        fields : FieldPoint
            Scalar field profile.
        dPhidz : FieldPoint
            Derivative with respect to the position of the scalar field profile.
        T : float
            Temperature.
        s1 : float
            Value of :math:`T^{30}-T_{\rm out}^{30}`.
        s2 : float
            Value of :math:`T^{33}-T_{\rm out}^{33}`.

        Returns
        -------
        float
            LHS of Eq. (20) of arXiv:2204.13120v1.

        """
        # Need enthalpy ouside a free-energy minimum (eq (12) in the ref.)
        enthalpy = -T * self.thermo.effectivePotential.derivT(fields, T)

        kineticTerm = 0.5 * np.sum(dPhidz**2).view(np.ndarray)

        ## eff potential at this field point and temperature. NEEDS the T-dep constant
        veff = self.thermo.effectivePotential.evaluate(fields, T)

        result = (
            kineticTerm
            - veff
            - 0.5 * enthalpy
            + 0.5 * np.sqrt(4 * s1**2 + enthalpy**2)
            - s2
        )

        result = np.asarray(result)
        if result.shape == (1,) and len(result) == 1:
            return float(result[0])
        if result.shape == ():
            return float(result)
        raise TypeError(f"LHS has wrong type, {result.shape=}")

    def deltaToTmunu(
        self,
        index: int,
        fields: FieldPoint,
        velocityMid: float,
        offEquilDeltas: BoltzmannDeltas,
    ) -> Tuple[float, float]:
        r"""
        Computes the out-of-equilibrium part of the energy-momentum tensor. See eq. (14)
        of arXiv:2204.13120v1.

        Parameters
        ----------
        index : int
            Index of the grid point on which to find the plasma profile.
        fields : FieldPoint
            Scalar field profile.
        velocityMid : float
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        offEquilDeltas : BoltzmannDeltas
            BoltzmannDeltas object containing the off-equilibrium Delta functions

        Returns
        -------
        T30 : float
            Out-of-equilibrium part of :math:`T^{30}`.
        T33 : float
            Out-of-equilibrium part of :math:`T^{33}`.

        """
        Delta00 = offEquilDeltas.Delta00.coefficients[  # pylint: disable=invalid-name
            :, index
        ]
        Delta02 = offEquilDeltas.Delta02.coefficients[  # pylint: disable=invalid-name
            :, index
        ]
        Delta20 = offEquilDeltas.Delta20.coefficients[  # pylint: disable=invalid-name
            :, index
        ]
        Delta11 = offEquilDeltas.Delta11.coefficients[  # pylint: disable=invalid-name
            :, index
        ]

        u0 = np.sqrt(gammaSq(velocityMid))  # pylint: disable=invalid-name
        u3 = np.sqrt(gammaSq(velocityMid)) * velocityMid  # pylint: disable=invalid-name
        ubar0 = u3
        ubar3 = u0

        # Computing the out-of-equilibrium part of the energy-momentum tensor
        T30 = np.sum(
            [
                particle.totalDOFs
                * (
                    +(
                        3 * Delta20[i]
                        - Delta02[i]
                        - particle.msqVacuum(fields) * Delta00[i]
                    )
                    * u3
                    * u0
                    + (
                        3 * Delta02[i]
                        - Delta20[i]
                        + particle.msqVacuum(fields) * Delta00[i]
                    )
                    * ubar3
                    * ubar0
                    + 2 * Delta11[i] * (u3 * ubar0 + ubar3 * u0)
                )
                / 2.0
                for i, particle in enumerate(self.particles)
            ]
        )
        T33 = np.sum(
            [
                particle.totalDOFs
                * (
                    (
                        +(
                            3 * Delta20[i]
                            - Delta02[i]
                            - particle.msqVacuum(fields) * Delta00[i]
                        )
                        * u3
                        * u3
                        + (
                            3 * Delta02[i]
                            - Delta20[i]
                            + particle.msqVacuum(fields) * Delta00[i]
                        )
                        * ubar3
                        * ubar3
                        + 4 * Delta11[i] * u3 * ubar3
                    )
                    / 2.0
                    - (
                        particle.msqVacuum(fields) * Delta00[i]
                        + Delta02[i]
                        - Delta20[i]
                    )
                    / 2.0
                )
                for i, particle in enumerate(self.particles)
            ]
        )

        return T30, T33

    def getBoltzmannFiniteDifference(self) -> BoltzmannResults:
        """Mostly to estimate errors, recomputes Boltzmann stuff
        using finite difference derivatives.
        """
        # finite difference method requires everything to be in
        # the Cardinal basis
        boltzmannSolverFiniteDifference = copy.deepcopy(self.boltzmannSolver)
        boltzmannSolverFiniteDifference.derivatives = "Finite Difference"
        assert (
            boltzmannSolverFiniteDifference.basisM == "Cardinal"
        ), "Error in boltzmannFiniteDifference: must be in Cardinal basis"
        boltzmannSolverFiniteDifference.basisN = "Cardinal"
        boltzmannSolverFiniteDifference.collisionArray.changeBasis("Cardinal")
        # now computing results
        return boltzmannSolverFiniteDifference.getDeltas()
