import numpy as np
import numpy.typing as npt
import warnings
from typing import Tuple
import copy  # for deepcopy


import scipy.optimize
from scipy.interpolate import UnivariateSpline

from .Boltzmann import BoltzmannSolver
from .Fields import Fields, FieldPoint
from .Grid import Grid
from .helpers import gammaSq  # derivatives for callable functions
from .Hydro import Hydro
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .WallGoTypes import BoltzmannResults, BoltzmannDeltas, BoltzmannBackground, HydroResults, WallGoResults, WallParams, WallGoInterpolationResults


class EOM:

    """
    Class that solves the energy-momentum conservation equations and the scalar EOMs to determine the wall velocity.
    """
    def __init__(
        self,
        boltzmannSolver: BoltzmannSolver,
        thermodynamics: Thermodynamics,
        hydro: Hydro,
        grid: Grid,
        nbrFields: int,
        meanFreePath: float,
        includeOffEq: bool=False,
        forceImproveConvergence: bool=False,
        errTol=1e-3,
        maxIterations=10,
        pressRelErrTol=0.3679,
    ):
        """
        Initialization

        Parameters
        ----------
        boltzmannSolver : BoltzmannSolver
            BoltzmannSolver instance.
        model : GenericModel
            Object of model class that implements the GenericModel interface.
        grid : Grid
            Object of the class Grid.
        nbrFields : int
            Number of scalar fields on which the scalar potential depends.
        meanFreePath : float
            Estimate of the mean free path of the particles in the plasma.
        includeOffEq : bool, optional
            If False, all the out-of-equilibrium contributions are neglected.
            The default is False.
        forceImproveConvergence : bool, optional
            If True, uses a slower algorithm that improves the convergence when 
            computing the pressure. The improved algorithm is automatically used 
            for detonation. Default is False.
        errTol : double, optional
            Absolute error tolerance. The default is 1e-3.
        maxIterations : integer, optional
            Maximum number of iterations for the convergence of pressure. The default is 10.
        pressRelErrTol : float, optional
            Relative tolerance in pressure when finding its root.

        Returns
        -------
        None.

        """

        assert isinstance(boltzmannSolver, BoltzmannSolver)
        assert isinstance(thermodynamics, Thermodynamics)
        assert isinstance(hydro, Hydro)
        assert isinstance(grid, Grid)
        assert grid is boltzmannSolver.grid, "EOM and BoltzmannSolver must have the same instance of the Grid object."

        self.boltzmannSolver = boltzmannSolver
        self.grid = grid
        self.errTol = errTol
        self.nbrFields = nbrFields
        self.meanFreePath = meanFreePath
        self.includeOffEq = includeOffEq
        self.forceImproveConvergence = forceImproveConvergence

        self.thermo = thermodynamics
        self.hydro = hydro
        ## LN: Dunno if we want to store this here tbh
        self.Tnucl = self.thermo.Tnucl
        
        self.particles = self.boltzmannSolver.offEqParticles
        
        ## Tolerances
        self.errTol = errTol
        self.maxIterations = maxIterations
        self.pressRelErrTol = pressRelErrTol
        self.pressAbsErrTol = 0


    def findWallVelocityMinimizeAction(self, wallThicknessIni: float=None) -> WallGoResults:
        """
        Finds the wall velocity by minimizing the action and solving for the
        solution with 0 total pressure on the wall. This function only looks for 
        deflagration or hybrid solutions. Returns a velocity of 1 if the pressure
        peak at vw = vJ is not large enough to stop the wall. 
        For detonation solutions, use solveInterpolation().

        Returns
        -------
        WallGoResults object containing the solution of the EOM.

        """
        
        # If no initial wall thickness was provided, starts with a reasonable guess
        if wallThicknessIni is None:
            wallThicknessIni = 5 / self.Tnucl
            
        wallParams = WallParams(
            widths=wallThicknessIni * np.ones(self.nbrFields),
            offsets=np.zeros(self.nbrFields),
        )

        vmin = self.hydro.vMin
        vmax = min(self.hydro.vJ,self.hydro.fastestDeflag())
        return self.solveWall(vmin, vmax, wallParams)
    

    def solveWall(self, wallVelocityMin: float, wallVelocityMax: float, wallParamsGuess: WallParams) -> WallGoResults:
        r"""
        Solves the equation :math:`P_{\rm tot}(\xi_w)=0` for the wall velocity 
        and wall thicknesses/offsets. The solver only looks between wallVelocityMin
        and wallVelocityMax

        Parameters
        ----------
        wallVelocityMin : double
            Lower bound of the bracket in which the root finder will look for a
            solution. Should satisfy
            :math:`0<{\rm wallVelocityMin}<{\rm wallVelocityMax}`.
        wallVelocityMax : double
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
        
        # Get the pressure at vw = vJ
        pressureMax, wallParamsMax, boltzmannResultsMax, boltzmannBackgroundMax, hydroResultsMax = self.wallPressure(wallVelocityMax, wallParamsGuess, True)
        
        # also getting the LTE results
        wallVelocityLTE = self.hydro.findvwLTE()
        
        # The pressure peak is not enough to stop the wall: no deflagration/hybrid solution
        if pressureMax < 0:
            print('Maximum pressure on wall is negative!')
            print(f"{pressureMax=} {wallParamsMax=}")
            results.setWallVelocities(1, 0, wallVelocityLTE)
            results.setWallParams(wallParamsMax)
            results.setHydroResults(hydroResultsMax)
            results.setBoltzmannBackground(boltzmannBackgroundMax)
            results.setBoltzmannResults(boltzmannResultsMax)
            return results
    
        # Get the pressure at vw = 0
        pressureMin, wallParamsMin, boltzmannResultsMin, boltzmannBackgroundMin, hydroResultsMin = self.wallPressure(wallVelocityMin, wallParamsGuess, True)
        if pressureMin > 0:
            print("""EOM warning: the pressure at vw = 0 is positive which indicates the phase transition cannot proceed.
                  Something might be wrong with your potential.""")
            results.setWallVelocities(0, 0, wallVelocityLTE)
            results.setWallParams(wallParamsMin)
            results.setHydroResults(hydroResultsMin)
            results.setBoltzmannBackground(boltzmannBackgroundMin)
            results.setBoltzmannResults(boltzmannResultsMin)
            return results
        
        self.pressAbsErrTol = 0.01 * self.errTol * (1 - self.pressRelErrTol) * np.minimum(np.abs(pressureMin), np.abs(pressureMax)) / 4

        ## This computes pressure on the wall with a given wall speed and WallParams 
        def pressureWrapper(vw: float):

            """Small optimization here: the root finder below calls this first at the bracket endpoints,
            for which we already computed the pressure above. So make use of those.
            """ 
            if np.abs(vw - wallVelocityMin) < 1e-10 or vw < wallVelocityMin:
                return pressureMin
            elif np.abs(vw - wallVelocityMax) < 1e-10 or vw > wallVelocityMax:
                return pressureMax

           # Use linear interpolation to get a better first guess for the initial wall parameters
            fractionVw = (vw - wallVelocityMin) / (wallVelocityMax - wallVelocityMin)
            newWallParams = (
                wallParamsMin + (wallParamsMax - wallParamsMin) * fractionVw
            )
            return self.wallPressure(vw, newWallParams, False)

        optimizeResult = scipy.optimize.root_scalar(
            pressureWrapper,
            method='brentq',
            bracket=[wallVelocityMin, wallVelocityMax],
            xtol=self.errTol,
        )
        wallVelocity = optimizeResult.root


        # Get wall params, and other results
        fractionWallVelocity = (wallVelocity - wallVelocityMin) / (wallVelocityMax - wallVelocityMin)
        newWallParams = (
            wallParamsMin + (wallParamsMax - wallParamsMin) * fractionWallVelocity
        )
        _, wallParams, boltzmannResults, boltzmannBackground, hydroResults = self.wallPressure(
            wallVelocity, newWallParams, returnExtras=True,
        )

        # minimum possible error in the wall speed
        wallVelocityMinError = self.errTol * optimizeResult.root

        # estimating errors from truncation and comparison to finite differences
        if self.includeOffEq:
            finiteDifferenceBoltzmannResults = self.getBoltzmannFiniteDifference()
            # assuming nonequilibrium errors proportional to deviation from LTE
            wallVelocityDeltaLTE = abs(wallVelocity - wallVelocityLTE)
            # the truncation error in the spectral method within Boltzmann
            wallVelocityTruncationError = boltzmannResults.truncationError * wallVelocityDeltaLTE
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
        results.setFiniteDifferenceBoltzmannResults(
            finiteDifferenceBoltzmannResults
        )

        # return collected results
        return results


    def wallPressure(
        self,
        wallVelocity: float,
        wallParams: WallParams,
        returnExtras: bool = False,
        atol: float = None,
        rtol: float = None,
        boltzmannResults: BoltzmannResults = None
    ) -> tuple:
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action. Can use two different iteration algorithms 
        to find the pressure. If self.forceImproveConvergence=False and 
        wallVelocity<self.hydro.vJ, uses a fast algorithm that sometimes fails 
        to converge. Otherwise, or if the previous algorithm converges slowly,
        uses a slower, but more robust algorithm.

        Parameters
        ----------
        wallVelocity : double
            Wall velocity at which the pressure is computed.
        wallParams : WallParams
            Contains a guess of the wall thicknesses and wall offsets.
        returnExtras : bool, optional
            If False, only the pressure is returned. If True, the pressure,
            and WallParams, BoltzmannResults and HydroResults, objects are
            returned. The default is False.
        atol : float or None
            Absolute tolerance. If None, uses self.pressAbsErrTol. Default is None.
        rtol : float or None
            Relative tolerance. If None, uses self.pressRelErrTol. Default is None.
        boltzmannResults : BoltzmannResults or None
            Object of the BoltzmannResults class containing the initial solution
            of the Boltzmann equation. If None, sets the initial deltaF to 0.
            Default is None.

        Returns
        -------
        pressure : double
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
            HydroResults object containing the solution obtained from Hydro.
            Only returned if returnExtras is True

        """
        
        if atol is None:
            atol = self.pressAbsErrTol
        if rtol is None:
            rtol = self.pressRelErrTol
            
        improveConvergence = self.forceImproveConvergence
        if wallVelocity > self.hydro.vJ:
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
            np.zeros((len(self.particles), (self.grid.M - 1), (self.grid.N - 1), (self.grid.N - 1))),
            self.grid,
            basis=("Array", "Cardinal", "Chebyshev", "Chebyshev"),
            direction=("Array", "z", "pz", "pp"),
            endpoints=False,
        )
        
        if boltzmannResults is None:
            boltzmannResults = BoltzmannResults(
                deltaF=deltaF,
                Deltas=offEquilDeltas,
                truncationError=0,
                linearizationCriterion1=0,
                linearizationCriterion2=0,
            )
        
        # Find the boundary conditions of the hydrodynamic equations
        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)
        hydroResults = HydroResults(
            temperaturePlus=Tplus,
            temperatureMinus=Tminus,
            velocityJouget=self.hydro.vJ,
        )
        
        # Positions of the phases
        vevLowT = self.thermo.freeEnergyLow(Tminus).fieldsAtMinimum
        vevHighT = self.thermo.freeEnergyHigh(Tplus).fieldsAtMinimum
             
        ##Estimate the new grid parameters 
        widths = wallParams.widths
        offsets = wallParams.offsets
        ## Distance between the right and left edges of the walls at the boundaries
        wallThicknessGrid = (np.max((1-offsets)*widths)-np.min((-1-offsets)*widths))/2
        ## Center between these two edges
        ## The source and pressure are proportional to d(m^2)/dz, which peaks at -wallThicknessGrid*np.log(2)/2. This is why we substract this value.
        wallCenterGrid = (np.max((1-offsets)*widths)+np.min((-1-offsets)*widths))/2 - wallThicknessGrid*np.log(2)/2
        gammaWall = 1/np.sqrt(1-velocityMid**2)
        ## The tail inside typically scales like gamma, while the one outside like 1/gamma
        ## We take the max because the tail lengths must be larger than wallThicknessGrid*(1+2*smoothing)/ratioPointsWall
        tailInside = max(self.meanFreePath*gammaWall*self.includeOffEq, wallThicknessGrid*(1+2.1*self.grid.smoothing)/self.grid.ratioPointsWall)
        tailOutside = max(self.meanFreePath/gammaWall*self.includeOffEq, wallThicknessGrid*(1+2.1*self.grid.smoothing)/self.grid.ratioPointsWall)
        self.grid.changePositionFalloffScale(tailInside, tailOutside, wallThicknessGrid, wallCenterGrid)
        
        
        pressure, wallParams, boltzmannResults, boltzmannBackground = self.__intermediatePressureResults(
            wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus, 
        )
        
        pressures = [pressure]
        
        ## The 'multiplier' parameter is used to reduce the size of the wall
        ## parameters updates during the iteration procedure. The next iteration
        ## will use multiplier*newWallParams+(1-multiplier)*oldWallParams. 
        ## Can be used when the iterations do not converge, even close to the 
        ## true solution. For small enough values, we should always be able to converge.
        ## The value will be reduced if the algorithm doesn't converge.
        multiplier = 1

        i = 0
        while True:
            if improveConvergence:
                # Use the improved algorithm (which converges better but slowly)
                pressure, wallParams, boltzmannResults, boltzmannBackground, errorSolver = self.__getNextPressure(
                    pressure, wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus, multiplier=multiplier
                )
            else:
                pressure, wallParams, boltzmannResults, boltzmannBackground = self.__intermediatePressureResults(
                    wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus, multiplier=multiplier,
                )
                errorSolver = 0
            pressures.append(pressure)

            error = np.abs(pressures[-1]-pressures[-2])
            errTol = np.maximum(rtol * np.abs(pressure), atol)*multiplier
            
            print(f"{pressure=} {error=} {errTol=} {improveConvergence=} {multiplier=}")
            i += 1

            if (error < errTol or (errorSolver < errTol and improveConvergence)):
                ## Even if two consecutive call to __getNextPressure() give similar pressures, it is possible
                ## that the internal calls made to __intermediatePressureResults() do not converge. This is measured
                ## by 'errorSolver'. If __getNextPressure() converges but not __intermediatePressureResults() doesn't,
                ## 'multiplier' is probably too large. We therefore continue the iteration procedure with a smaller 
                ## value of 'multiplier'.
                if errorSolver > errTol:
                    multiplier /= 2
                else:
                    break
            elif i >= self.maxIterations-1:
                print(
                    "Pressure for a wall velocity has not converged to "
                    "sufficient accuracy with the given maximum number "
                    "for iterations."
                )
                break
            
            if len(pressures) > 2:
                if error > abs(pressures[-2]-pressures[-3])/1.5:
                    # If the error decreases too slowly, use the improved algorithm
                    improveConvergence = True

        if returnExtras:
            return pressure, wallParams, boltzmannResults, boltzmannBackground, hydroResults
        else:
            return pressure
        
    def __getNextPressure(
            self, 
            pressure1: float, 
            wallParams1: WallParams, 
            vevLowT: Fields, 
            vevHighT: Fields, 
            c1: float,
            c2: float,
            velocityMid: float,
            boltzmannResults1: BoltzmannResults,
            Tplus: float,
            Tminus: float,
            Tprofile: npt.ArrayLike=None,
            velocityProfile: npt.ArrayLike=None,
            multiplier: float=1.0
            ) -> tuple:
        """
        Performs the next iteration to solve the EOM and Boltzmann equation. 
        First computes the pressure twice, updating the wall parameters and 
        Boltzmann results each time. If the iterations overshot the true solution
        (the two pressure updates go in opposite directions), uses linear 
        interpolation to find a better estimate of the true solution.
        """
        pressure2, wallParams2, boltzmannResults2, _= self.__intermediatePressureResults(
            wallParams1, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults1, Tplus, Tminus, Tprofile, velocityProfile, multiplier
        )
        pressure3, wallParams3, boltzmannResults3, boltzmannBackground3 = self.__intermediatePressureResults(
            wallParams2, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults2, Tplus, Tminus, Tprofile, velocityProfile, multiplier
        )
        
        ## If the last iteration does not overshoot the real pressure (the two 
        ## last update go in the same direction), returns the last iteration.
        if (pressure3-pressure2)*(pressure2-pressure1) >= 0:
            err = abs(pressure3-pressure2)
            return pressure3, wallParams3, boltzmannResults3, boltzmannBackground3, err
        
        ## If the last iteration overshot, uses linear interpolation to find a 
        ## better estimate of the true solution.
        x = (pressure1-pressure2)/(pressure1-2*pressure2+pressure3)
        pressure4, wallParams4, boltzmannResults4, boltzmannBackground4 = self.__intermediatePressureResults(
            wallParams1+(wallParams2-wallParams1)*x, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults1+(boltzmannResults2-boltzmannResults1)*x, Tplus, Tminus, Tprofile, velocityProfile, multiplier
        )
        err = abs(pressure4-pressure2)
        return pressure4, wallParams4, boltzmannResults4, boltzmannBackground4, err
        
        

    def __intermediatePressureResults(
            self, 
            wallParams: WallParams,
            vevLowT: Fields,
            vevHighT: Fields,
            c1: float,
            c2: float,
            velocityMid: float,
            boltzmannResults: BoltzmannResults,
            Tplus: float,
            Tminus: float,
            Tprofile: npt.ArrayLike=None,
            velocityProfile: npt.ArrayLike=None,
            multiplier: float=1.0,
            ) -> tuple:
        """
        Performs one step of the iteration procedure to update the pressure,
        wall parameters and Boltzmann solution. This is done by first solving 
        the Boltzmann equation and then minimizing the action to solve the EOM.
        """

        ## here dPhidz are z-derivatives of the fields
        fields, dPhidz = self.wallProfile(
            self.grid.xiValues, vevLowT, vevHighT, wallParams
        )

        if Tprofile is None and velocityProfile is None:
            Tprofile, velocityProfile = self.findPlasmaProfile(
                c1, c2, velocityMid, fields, dPhidz, boltzmannResults.Deltas, Tplus, Tminus
            )

        ## Prepare a new background for Boltzmann
        TWithEndpoints = np.concatenate(([Tminus], Tprofile, [Tplus]))
        fieldsWithEndpoints = np.concatenate((vevLowT, fields, vevHighT), axis=fields.overFieldPoints).view(Fields)
        vWithEndpoints = np.concatenate(([velocityProfile[0]], velocityProfile, [velocityProfile[-1]])) 
        boltzmannBackground = BoltzmannBackground(
            velocityMid, vWithEndpoints, fieldsWithEndpoints, TWithEndpoints,
        ) 
        if self.includeOffEq:
              ## ---- Solve Boltzmann equation to get out-of-equilibrium contributions
            self.boltzmannSolver.setBackground(boltzmannBackground)
            boltzmannResults = multiplier*self.boltzmannSolver.getDeltas() + (1-multiplier)*boltzmannResults

        ## ---- Next need to solve wallWidth and wallOffset. For this, put wallParams in a np 1D array,
        ## NOT including the first offset which we keep at 0.
        wallArray = np.concatenate( (wallParams.widths, wallParams.offsets[1:]) ) ## should work even if offsets is just 1 element

        ## first width, then offset
        lowerBounds = np.concatenate((self.nbrFields * [0.1 / self.Tnucl] , (self.nbrFields-1) * [-10.] ))
        upperBounds = np.concatenate((self.nbrFields * [100. / self.Tnucl] , (self.nbrFields-1) * [10.] ))
        bounds = scipy.optimize.Bounds(lb = lowerBounds, ub = upperBounds)

        ## And then a wrapper that puts the inputs back in WallParams 
        def actionWrapper(wallArray: npt.ArrayLike, *args) -> float:
            return self.action( self.__toWallParams(wallArray), *args )
        
        Delta00 = boltzmannResults.Deltas.Delta00
        sol = scipy.optimize.minimize(
            actionWrapper,
            wallArray,
            args=(vevLowT, vevHighT, Tprofile, Delta00),
            method='Nelder-Mead',
            bounds=bounds,
        )

        ## Put the resulting width, offset back in WallParams format
        wallParams = multiplier*self.__toWallParams(sol.x) + (1-multiplier)*wallParams

        fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)
        dVdPhi = self.thermo.effectivePotential.derivField(fields, Tprofile)
        
        # Out-of-equilibrium term of the EOM
        dVout = np.sum([particle.totalDOFs * particle.msqDerivative(fields) * Delta00.coefficients[i,:,None]
                        for i,particle in enumerate(self.particles)], axis=0) / 2
        
        ## EOM for field i is d^2 phi_i + dVfull == 0, the latter term is dVdPhi + dVout
        dVfull: Fields = dVdPhi + dVout

        dVdz = np.sum(np.array(dVfull * dPhidz), axis=1)

        EOMPoly = Polynomial(dVdz, self.grid)

        dzdchi,_,_ = self.grid.getCompactificationDerivatives()
        pressure = EOMPoly.integrate(w=-dzdchi)

        return pressure, wallParams, boltzmannResults, boltzmannBackground
    
    def gridPressure(self, vmin: float, vmax: float, nbrPoints: int, wallThicknessIni: float=None, rtol: float=1e-3) -> tuple:
        """
        Computes the pressure on a linearly spaced grid of velocities between 
        vmin and vmax.

        Parameters
        ----------
        vmin : float
            Lower bound of the interpolation interval.
        vmax : float
            Upper bound of the interpolation interval.
        nbrPoints : int
            Number of points on the grid.
        wallThicknessIni : float, optional
            Initial wall thickness used to compute the first pressure at vmin.
            The default is None.
        rtol : float, optional
            Relative tolerance. The default is 1e-3.

        Returns
        -------
        wallVelocities : list[float]
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
            wallThicknessIni = 5 / self.Tnucl
            
        wallParams = WallParams(
            widths=wallThicknessIni * np.ones(self.nbrFields),
            offsets=np.zeros(self.nbrFields),
        )
        
        boltzmannResults = None
        
        pressure, wallParams, boltzmannResults, _, hydroResults = self.wallPressure(vmin, wallParams, True, 0, rtol, boltzmannResults)
        
        pressures = []
        boltzmannBackgroundList = []
        boltzmannResultsList = []
        hydroResultsList = []
        wallParamsList = []
        # Computing the pressure on the velocity grid
        for i,wallVelocity in enumerate(wallVelocities):
            if i > 1:
                # Use linear extrapolation to get a more accurate initial value of wall parameters
                wallParamsTry = wallParamsList[-1] + (wallParamsList[-1]-wallParamsList[-2])*(wallVelocity-wallVelocities[i-1])/(wallVelocities[i-1]-wallVelocities[i-2])
                boltzmannResultsTry = boltzmannResultsList[-1] + (boltzmannResultsList[-1]-boltzmannResultsList[-2])*((wallVelocity-wallVelocities[i-1])/(wallVelocities[i-1]-wallVelocities[i-2]))
            else:
                wallParamsTry = wallParams
                boltzmannResultsTry = boltzmannResults
                
            pressure, wallParams, boltzmannResults, boltzmannBackground, hydroResults = self.wallPressure(wallVelocity, wallParamsTry, True, 0, rtol, boltzmannResultsTry)
            
            pressures.append(pressure)
            wallParamsList.append(wallParams)
            boltzmannResultsList.append(boltzmannResults)
            boltzmannBackgroundList.append(boltzmannBackground)
            hydroResultsList.append(hydroResults)
        
        return wallVelocities, np.array(pressures), wallParamsList, boltzmannResultsList, boltzmannBackgroundList, hydroResultsList
    
    def solveInterpolation(self, vmin: float, vmax: float, wallThicknessIni: float=None, desiredPressure: float=0, rtol: float=1e-3, dvMin: float=0.02) -> WallGoInterpolationResults:
        """
        Finds all the EOM solutions in some interval by computing the pressure 
        on a grid and interpolating to get the roots.

        Parameters
        ----------
        vmin : float
            Lower bound of the interpolation interval.
        vmax : float
            Upper bound of the interpolation interval.
        wallThicknessIni : float, optional
            Initial wall thickness used to compute the first pressure at vmin. 
            The default is None.
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
            nbrPoints = max(1+int((vmax-vmin)/min(dvMin,rtol**0.25)), 5)
            # Computing the pressure on the grid
            wallVelocities, pressures, wallParamsList, boltzmannResultsList, boltzmannBackgroundList, hydroResultsList = self.gridPressure(vmin, vmax, nbrPoints, wallThicknessIni, rtol)
            # Splining the result
            pressuresSpline = UnivariateSpline(wallVelocities, pressures-desiredPressure, s=0)
            
            # Finding the roots of the spline and classifying the result as stable or unstable solutions
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
        
        else: 
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

        

    def __toWallParams(self, wallArray: npt.ArrayLike) -> WallParams:
        offsets = np.concatenate( ([0], wallArray[self.nbrFields:]) )
        return WallParams(widths = wallArray[:self.nbrFields], offsets = offsets)

    def action(self, wallParams: WallParams, vevLowT: Fields, vevHighT: Fields, Tprofile: npt.ArrayLike, offEquilDelta00: npt.ArrayLike) -> float:
        r"""
        Computes the action by using gaussian quadratrure to integrate the Lagrangian.

        Parameters
        ----------
        wallParams : array-like
            Array of size 2*N-1 containing :math:`(L_0,L_i,\delta_i)`.
        vevLowT : array-like
            Field values in the low-T phase.
        vevHighT : array-like
            Field values in the high-T phase.
        Tprofile : array-like
            Temperature profile on the grid.
        offEquilDelta00 : array-like
            Off-equilibrium function Delta00.

        Returns
        -------
        action : double
            Action spent by the scalar field configuration.

        """

        wallWidths = wallParams.widths

        # Computing the field profiles
        fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)

        # Computing the potential
        V = self.thermo.effectivePotential.evaluate(fields, Tprofile)

        # Computing the out-of-equilibrium term of the action
        VOut = sum([particle.totalDOFs*particle.msqVacuum(fields)*offEquilDelta00.coefficients[i] for i,particle in enumerate(self.particles)])/2
        
        # Values of the potential at the boundaries
        VLowT = self.thermo.effectivePotential.evaluate(vevLowT,Tprofile[0])
        VHighT = self.thermo.effectivePotential.evaluate(vevHighT,Tprofile[-1])

        Vref = (VLowT+VHighT)/2
        
        # Integrating the potential to get the action 
        # We substract Vref (which has no effect on the EOM) to make the integral finite
        VPoly = Polynomial(V+VOut-Vref, self.grid)
        dzdchi,_,_ = self.grid.getCompactificationDerivatives()
        U = VPoly.integrate(w = dzdchi)
        # Kinetic part of the action
        K = np.sum((vevHighT-vevLowT)**2/(6*wallWidths))

        return U + K  


    def wallProfile(self, z, vevLowT: Fields, vevHighT: Fields, wallParams: WallParams) -> Tuple[Fields, Fields]:
        """
        Computes the scalar field profile and its derivative with respect to
        the position in the wall.

        Parameters
        ----------
        z : array-like
            Position grid on which to compute the scalar field profile.
        vevLowT : array-like
            Scalar field VEVs in the low-T phase.
        vevHighT : array-like
            Scalar field VEVs in the high-T phase.
        wallWidths : array-like
            Array containing the wall widths.
        wallOffsets : array-like
            Array containing the wall offsets.

        Returns
        -------
        fields : array-like
            Scalar field profile.
        dPhidz : array-like
            Derivative with respect to the position of the scalar field profile.

        """

        if np.isscalar(z):
            z_L = z / wallParams.widths
        else:
            ## Broadcast mess needed
            z_L = z[:,None] / wallParams.widths[None,:]

        fields = vevLowT + 0.5*(vevHighT - vevLowT) * (1 + np.tanh( z_L + wallParams.offsets ))
        dPhidz = 0.5*(vevHighT-vevLowT) / ( wallParams.widths * np.cosh(z_L + wallParams.offsets)**2 )

        return Fields.CastFromNumpy(fields), Fields.CastFromNumpy(dPhidz)


    def findPlasmaProfile(self, c1: float , c2: float, velocityMid: float, fields: Fields, dPhidz: Fields, 
                          offEquilDeltas: list, Tplus: float, Tminus: float) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        r"""
        Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of LHS.

        Parameters
        ----------
        c1 : double
            Value of the :math:`T^{30}` component of the energy-momentum tensor.
        c2 : double
            Value of the :math:`T^{33}` component of the energy-momentum tensor.
        velocityMid : double
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        fields : array-like
            Scalar field profiles.
        dPhidz : array-like
            Derivative with respect to the position of the scalar field profiles.
        offEquilDeltas : list
            List of dictionaries containing the off-equilibrium Delta functions
        Tplus : double
            Plasma temperature in front of the wall.
        Tminus : double
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

            """Ensure that we got only one one (T, vPlasma) value from the above.
            Particularly the vPlasma tends to be in len=1 array format because our Veff is intended to work with arrays
            """
            T = np.asanyarray(T)
            vPlasma = np.asanyarray(vPlasma)
            assert (T.size == 1 and vPlasma.size == 1), "Invalid output from findPlasmaProfilePoint()! In EOM.findPlasmaProfile(), grid loop."
            # convert to float, this is OK and works for all shapes because we checked the size above
            temperatureProfile[index] = T.item()
            velocityProfile[index] = vPlasma.item()

        return temperatureProfile, velocityProfile
    

    def findPlasmaProfilePoint(self, index: int, c1: float, c2: float, velocityMid: float, fields: Fields, dPhidz: Fields, 
                               offEquilDeltas: list, Tplus: float, Tminus: float) -> Tuple[float, float]:
        r"""
        Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.

        Parameters
        ----------
        index : int
            Index of the grid point on which to find the plasma profile.
        c1 : double
            Value of the :math:`T^{30}` component of the energy-momentum tensor.
        c2 : double
            Value of the :math:`T^{33}` component of the energy-momentum tensor.
        velocityMid : double
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        fields : Fields
            Scalar field profile.
        dPhidz : Fields
            Derivative with respect to the position of the scalar field profile.
        offEquilDeltas : list
            List of dictionaries containing the off-equilibrium Delta functions
        Tplus : double
            Plasma temperature in front of the wall.
        Tminus : double
            Plasma temperature behind the wall.

        Returns
        -------
        T : double
            Temperature at the point grid.xiValues[index].
        vPlasma : double
            Plasma velocity at the point grid.xiValues[index].

        """

        # Computing the out-of-equilibrium part of the energy-momentum tensor
        Tout30, Tout33 = self.deltaToTmunu(index, fields, velocityMid, offEquilDeltas)
        s1 = c1 - Tout30
        s2 = c2 - Tout33
        
        """
        The function we want to solve look in general like a parabola. In particular,
        it has two solutions, one deflagration and one detonation. To solve it,
        we first find the parabola's minimum, and then select the desired 
        solution on either side of the minimum.
        """
        minRes = scipy.optimize.minimize_scalar(lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2), method='Bounded', bounds=[0,2*max(Tplus,Tminus)])
        
        # If the minimum is positive, there are no roots and we return the minimum's position
        if self.temperatureProfileEqLHS(fields, dPhidz, minRes.x, s1, s2) >= 0:
            T = minRes.x
            vPlasma = self.plasmaVelocity(fields, T, s1)
            return T, vPlasma

        # Bracketing the root
        T1 = minRes.x
        TMultiplier = max(Tplus/T1, 1.2)
        if Tplus < Tminus: # If this is a detonation solution, finds a solution below TLowerBound
            TMultiplier = min(Tminus/T1, 0.8)

        T2 = T1*TMultiplier
        while self.temperatureProfileEqLHS(fields, dPhidz, T2, s1, s2) < 0:
            T1 *= TMultiplier
            T2 *= TMultiplier

        # Solving for the root
        res = scipy.optimize.brentq(
            lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2),
            T1,
            T2,
            xtol=1e-9,
            rtol=1e-9, ## really???
        )

        T = res
        vPlasma = self.plasmaVelocity(fields, T, s1)
        return T, vPlasma

    def plasmaVelocity(self, fields: Fields, T: npt.ArrayLike, s1: float) -> npt.ArrayLike:
        r"""
        Computes the plasma velocity as a function of the temperature.

        Parameters
        ----------
        fields : Fields
            Scalar field profiles.
        T : npt.ArrayLike
            Temparature.
        s1 : double
            Value of :math:`T^{30}-T_{\rm out}^{30}`.

        Returns
        -------
        double
            Plasma velocity.

        """
        ## Need "enthalpy" but ouside a free-energy minimum! More precisely, eq (12) in the ref. 
        w = -T * self.thermo.effectivePotential.derivT(fields, T)

        return (-w  + np.sqrt(4*s1**2 + w**2)) / (2 * s1)

    def temperatureProfileEqLHS(self, fields: FieldPoint, dPhidz: FieldPoint, T: float, s1: float, s2: float):
        r"""
        The LHS of Eq. (20) of arXiv:2204.13120v1.

        Parameters
        ----------
        fields : array-like
            Scalar field profile.
        dPhidz : array-like
            Derivative with respect to the position of the scalar field profile.
        T : double
            Temperature.
        s1 : double
            Value of :math:`T^{30}-T_{\rm out}^{30}`.
        s2 : double
            Value of :math:`T^{33}-T_{\rm out}^{33}`.

        Returns
        -------
        double
            LHS of Eq. (20) of arXiv:2204.13120v1.

        """
        ## Need "enthalpy" but ouside a free-energy minimum! More precisely, eq (12) in the ref. 
        w = -T * self.thermo.effectivePotential.derivT(fields, T)

        kineticTerm = 0.5*np.sum(dPhidz**2).view(np.ndarray)

        ## eff potential at this field point and temperature. NEEDS the T-dep constant
        veff = self.thermo.effectivePotential.evaluate(fields, T)

        result = (
            kineticTerm
            - veff - 0.5*w + 0.5*np.sqrt(4*s1**2 + w**2) - s2
        )

        result = np.asarray(result)
        if result.shape == (1,) and len(result) == 1:
            return result[0]
        elif result.shape == ():
            return result
        else:
            raise TypeError(f"LHS has wrong type, {result.shape=}")


    def deltaToTmunu(self, index: int, fields: Fields, velocityMid: float, offEquilDeltas: list) -> Tuple[float, float]:
        r"""
        Computes the out-of-equilibrium part of the energy-momentum tensor.

        Parameters
        ----------
        index : int
            Index of the grid point on which to find the plasma profile.
        fields : Fields
            Scalar field profile.
        velocityMid : double
            Midpoint of plasma velocity in the wall frame, :math:`(v_+ + v_-)/2`.
        offEquilDeltas : list
            List of dictionaries containing the off-equilibrium Delta functions

        Returns
        -------
        T30 : double
            Out-of-equilibrium part of :math:`T^{30}`.
        T33 : double
            Out-of-equilibrium part of :math:`T^{33}`.

        """
        delta00 = offEquilDeltas.Delta00.coefficients[:,index]
        delta02 = offEquilDeltas.Delta02.coefficients[:,index]
        delta20 = offEquilDeltas.Delta20.coefficients[:,index]
        delta11 = offEquilDeltas.Delta11.coefficients[:,index]

        u0 = np.sqrt(gammaSq(velocityMid))
        u3 = np.sqrt(gammaSq(velocityMid))*velocityMid
        ubar0 = u3
        ubar3 = u0

        # Computing the out-of-equilibrium part of the energy-momentum tensor
        T30 = np.sum([particle.totalDOFs*(
            + (3*delta20[i] - delta02[i] - particle.msqVacuum(fields)*delta00[i])*u3*u0
            + (3*delta02[i] - delta20[i] + particle.msqVacuum(fields)*delta00[i])*ubar3*ubar0
            + 2*delta11[i]*(u3*ubar0 + ubar3*u0))/2. for i,particle in enumerate(self.particles)])
        T33 = np.sum([particle.totalDOFs*((
            + (3*delta20[i] - delta02[i] - particle.msqVacuum(fields)*delta00[i])*u3*u3
            + (3*delta02[i] - delta20[i] + particle.msqVacuum(fields)*delta00[i])*ubar3*ubar3
            + 4*delta11[i]*u3*ubar3)/2. 
            - (particle.msqVacuum(fields)*delta00[i]+ delta02[i]-delta20[i])/2.) for i,particle in enumerate(self.particles)])

        return T30, T33

    def getBoltzmannFiniteDifference(self) -> BoltzmannResults:
        """Mostly to estimate errors, recomputes Boltzmann stuff
        using finite difference derivatives.
        """
        # finite difference method requires everything to be in
        # the Cardinal basis
        boltzmannSolverFiniteDifference = copy.deepcopy(self.boltzmannSolver)
        boltzmannSolverFiniteDifference.derivatives = "Finite Difference"
        assert boltzmannSolverFiniteDifference.basisM == "Cardinal", \
            "Error in boltzmannFiniteDifference: must be in Cardinal basis"
        boltzmannSolverFiniteDifference.basisN = "Cardinal"
        boltzmannSolverFiniteDifference.collisionArray.changeBasis("Cardinal")
        # now computing results
        return boltzmannSolverFiniteDifference.getDeltas()