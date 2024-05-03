import numpy as np
import numpy.typing as npt
import warnings
from typing import Tuple
import copy  # for deepcopy


import scipy.optimize

from .Boltzmann import (
    BoltzmannBackground, BoltzmannDeltas, BoltzmannSolver
)
from .Fields import Fields, FieldPoint
from .GenericModel import GenericModel
from .Grid import Grid
from .helpers import gammaSq  # derivatives for callable functions
from .Hydro import Hydro
from .Polynomial import Polynomial
from .Thermodynamics import Thermodynamics
from .WallGoTypes import BoltzmannResults, HydroResults, WallGoResults, WallParams


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
        includeOffEq: bool=False,
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
        includeOffEq : bool, optional
            If False, all the out-of-equilibrium contributions are neglected.
            The default is False.
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

        self.boltzmannSolver = boltzmannSolver
        self.grid = grid
        self.errTol = errTol
        self.nbrFields = nbrFields
        self.includeOffEq = includeOffEq

        self.thermo = thermodynamics
        self.hydro = hydro
        ## LN: Dunno if we want to store this here tbh
        self.Tnucl = self.thermo.Tnucl
        
        self.particles = self.boltzmannSolver.offEqParticles
        
        ## Tolerances
        self.errTol = errTol
        self.maxIterations = maxIterations
        self.pressRelErrTol = pressRelErrTol


    def findWallVelocityMinimizeAction(self):
        """
        Finds the wall velocity by minimizing the action and solving for the
        solution with 0 total pressure on the wall.

        Returns
        -------
        wallVelocity : double
            Value of the wall velocity that solves the scalar EOMs.
        wallParams : array-like
            Array containing the wall thicknesses and wall offsets that
            minimize the action and solve the EOM.

        """
        
        assert self.grid is self.boltzmannSolver.grid, "EOM and BoltzmannSolver must have the same instance of the Grid object."

        # LN: note that I've made widths and offsets be same size. Previously offsets was one element shorter

        wallParams = WallParams(
            widths=(5 / self.Tnucl) * np.ones(self.nbrFields),
            offsets=np.zeros(self.nbrFields),
        )

        vmin = self.hydro.vMin
        vmax = min(self.hydro.vJ,self.hydro.fastestDeflag())

        return self.solveWall(vmin, vmax, wallParams)
    

    ## LN: Right so this actually solves wall properties and not the pressure! So changed the name
    def solveWall(self, wallVelocityMin: float, wallVelocityMax: float, wallParamsGuess: WallParams) -> WallGoResults:
        r"""
        Solves the equation :math:`P_{\rm tot}(\xi_w)=0` for the wall velocity and wall thicknesses/offsets.

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

        self.pressAbsErrTol = 1e-8  # why 1e-8?
        ## HACK! LN: Return values here need to be consistent. Can't sometimes have 1 number, sometimes tuple etc
        pressureMax, wallParamsMax, _, _, _ = self.wallPressure(
            wallVelocityMax, wallParamsGuess, True
        )
        if pressureMax < 0:
            print('Maximum pressure on wall is negative!')
            print(f"{pressureMax=} {wallParamsMax=}")
            #return 1
            return 1, wallParamsMax
    
        pressureMin, wallParamsMin, _, _, _ = self.wallPressure(wallVelocityMin, wallParamsGuess, True)
        if pressureMin > 0:
            ## If this is a bad outcome then we should warn about it. TODO
            #return 0
            return 0, wallParamsMin
        
        self.pressAbsErrTol = 0.01 * self.errTol * (1 - self.pressRelErrTol) * np.minimum(np.abs(pressureMin), np.abs(pressureMax)) / 4

        ## This computes pressure on the wall with a given wall speed and WallParams that looks hacky
        def pressureWrapper(vw: float):

            """Small optimization here: the root finder below calls this first at the bracket endpoints,
            for which we already computed the pressure above. So make use of those.
            In principle a direct float == float comparison could work here, but that's illegal.
            I also include the case where vw is outside [wallVelocityMin, wallVelocityMax] although it probably does not occur.
            """ 
            absTolerance = 1e-8  # why 1e-8?
            if np.abs(vw - wallVelocityMin) < absTolerance or vw < wallVelocityMin:
                return pressureMin
            elif np.abs(vw - wallVelocityMax) < absTolerance or vw > wallVelocityMax:
                return pressureMax

            # Don't return wall params. But this seems pretty evil: wallPressure() modifies the wallParams it gets as input!
            # HACK! This newWallParams thing needs explaining.
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

        # also getting the LTE results
        wallVelocityLTE = self.hydro.findvwLTE()

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
    ):
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action.

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

        Returns
        -------
        pressure : double
            Total pressure on the wall.
        wallParams : array-like
            Array containing the wall thicknesses and wall offsets that
            minimize the action and solve the EOM. Only returned if
            returnOptimalWallParams is True.

        """
        
        # Let's not allow this:
        """ 
        if wallParams is None:
            wallParams = np.append(self.nbrFields*[5/self.Tnucl], (self.nbrFields-1)*[0])
        """

        print(f"\nTrying {wallVelocity=}")

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
        boltzmannResults = BoltzmannResults(
            deltaF=deltaF,
            Deltas=offEquilDeltas,
            truncationError=0,
            linearizationCriterion1=0,
            linearizationCriterion2=0,
        )

        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)
        hydroResults = HydroResults(
            temperaturePlus=Tplus,
            temperatureMinus=Tminus,
            velocityJouget=self.hydro.vJ,
        )

        vevLowT = self.thermo.freeEnergyLow(Tminus).fieldsAtMinimum
        vevHighT = self.thermo.freeEnergyHigh(Tplus).fieldsAtMinimum
                
        # Estimate L_xi
        # msq1 = self.particle.msqVacuum(vevHighT)
        # msq2 = self.particle.msqVacuum(vevLowT)
        # L1,L2 = self.boltzmannSolver.collisionArray.estimateLxi(-velocityMid, Tplus, Tminus, msq1, msq2)
        # L_xi = max(L1/2, L2/2, 2*max(wallParams.widths))
        
        L_xi = 2*max(wallParams.widths)
        self.grid.changePositionFalloffScale(L_xi)

        pressure, wallParams, boltzmannResults, boltzmannBackground = self.intermediatePressureResults(
            wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus
        )

        i = 0
        while True:
            pressureOld = pressure
            
            pressure, wallParams, boltzmannResults, boltzmannBackground = self.intermediatePressureResults(
                wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus
            )

            error = np.abs(pressure-pressureOld)
            errTol = np.maximum(self.pressRelErrTol * np.abs(pressure), self.pressAbsErrTol)
            
            print(f"{pressure=} {error=} {errTol=}")
            i += 1

            if error < errTol:
                break
            elif i >= self.maxIterations-1:
                print(
                    "Pressure for a wall velocity has not converged to "
                    "sufficient accuracy with the given maximum number "
                    "for iterations."
                )
                break

        if returnExtras:
            return pressure, wallParams, boltzmannResults, boltzmannBackground, hydroResults
        else:
            return pressure
        

    def intermediatePressureResults(self, wallParams, vevLowT, vevHighT, c1, c2, velocityMid, boltzmannResults, Tplus, Tminus):

        ## here dPhidz are z-derivatives of the fields
        fields, dPhidz = self.wallProfile(
            self.grid.xiValues, vevLowT, vevHighT, wallParams
        )

        Tprofile, velocityProfile = self.findPlasmaProfile(
            c1, c2, velocityMid, fields, dPhidz, boltzmannResults.Deltas, Tplus, Tminus
        )

        """LN: If I'm reading this right, for Boltzmann we have to append endpoints to our field,T,velocity profile arrays.
        Doing this reshaping here every time seems not very performant => consider getting correct shape already from wallProfile(), findPlasmaProfile().
        TODO also BoltzmannSolver seems to drop the endpoints internally anyway!!
        """
        ## Prepare a new background for Boltzmann
        TWithEndpoints = np.concatenate(([Tminus], Tprofile, [Tplus]))
        fieldsWithEndpoints = np.concatenate((vevLowT, fields, vevHighT), axis=fields.overFieldPoints).view(Fields)
        vWithEndpoints = np.concatenate(([velocityProfile[0]], velocityProfile, [velocityProfile[-1]])) 
        boltzmannBackground = BoltzmannBackground(
            velocityMid, vWithEndpoints, fieldsWithEndpoints, TWithEndpoints,
        ) 
        if self.includeOffEq:
             ## ---- Solve Boltzmann equation to get out-of-equilibrium contributions
            """TODO I suggest handling background-related logic inside BoltzmannSolver. Here we could just pass necessary input
                and let BoltzmannSolver create/manage the actual BoltzmannBackground object
                """
            self.boltzmannSolver.setBackground(boltzmannBackground)
            boltzmannResults = self.boltzmannSolver.getDeltas()

        ## ---- Next need to solve wallWidth and wallOffset. For this, put wallParams in a np 1D array,
        ## NOT including the first offset which we keep at 0.
        wallArray = np.concatenate( (wallParams.widths, wallParams.offsets[1:]) ) ## should work even if offsets is just 1 element
        
        ## This gives WallParams back from the above array, putting 0 as the first offset
        def __toWallParams(_wallArray: np.ndarray) -> WallParams:
            offsets = np.concatenate( ([0], _wallArray[self.nbrFields:]) )
            return WallParams(widths = _wallArray[:self.nbrFields], offsets = offsets)

        ## LN: Where do these bounds come from!?
        ## first width, then offset
        lowerBounds = np.concatenate((self.nbrFields * [0.1 / self.Tnucl] , (self.nbrFields-1) * [-10.] ))
        upperBounds = np.concatenate((self.nbrFields * [100. / self.Tnucl] , (self.nbrFields-1) * [10.] ))
        bounds = scipy.optimize.Bounds(lb = lowerBounds, ub = upperBounds)

        ## And then a wrapper that puts the inputs back in WallParams (could maybe bypass this somehow...?)
        def actionWrapper(wallArray: np.ndarray, *args) -> float:
            return self.action( __toWallParams(wallArray), *args )
        
        Delta00 = boltzmannResults.Deltas.Delta00
        sol = scipy.optimize.minimize(
            actionWrapper,
            wallArray,
            args=(vevLowT, vevHighT, Tprofile, Delta00),
            method='Nelder-Mead',
            bounds=bounds,
        )

        ## Put the resulting width, offset back in WallParams format
        wallParams = __toWallParams(sol.x)

        fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)
        dVdPhi = self.thermo.effectivePotential.derivField(fields, Tprofile)
        
        # Out-of-equilibrium term of the EOM
        dVout = np.sum([particle.totalDOFs * particle.msqDerivative(fields) * Delta00.coefficients[i,:,None]
                        for i,particle in enumerate(self.particles)], axis=0) / 2
        
        ## EOM for field i is d^2 phi_i + dVfull == 0, the latter term is dVdPhi + dVout
        dVfull: Fields = dVdPhi + dVout

        """
        In principle, the following should be sumed over all the particles, but it turns 
        out that only the first particle has a nonzero contribution. This is 
        because the other particles have a wall offset over which the action is
        minimized. This sets their pressure to 0. The first field doesn't have 
        an offset, so its pressure is nonzero.
        """
        dVdz = (dVfull * dPhidz).GetField(0)

        EOMPoly = Polynomial(dVdz, self.grid)
        
        dzdchi,_,_ = self.grid.getCompactificationDerivatives()
        pressure = EOMPoly.integrate(w=-dzdchi)

        ## Observation: dV/dPhi derivative can be EXTREMELY sensitive to small changes in T. So if comparing things manually, do keep this in mind

        return pressure, wallParams, boltzmannResults, boltzmannBackground



    def action(self, wallParams: WallParams, vevLowT: Fields, vevHighT: Fields, Tprofile: np.ndarray, offEquilDelta00: np.ndarray) -> float:
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

        fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)

        V = self.thermo.effectivePotential.evaluate(fields, Tprofile)


        VOut = sum([particle.totalDOFs*particle.msqVacuum(fields)*offEquilDelta00.coefficients[i] for i,particle in enumerate(self.particles)])/2

        VLowT = self.thermo.effectivePotential.evaluate(vevLowT,Tprofile[0])
        VHighT = self.thermo.effectivePotential.evaluate(vevHighT,Tprofile[-1])

        Vref = (VLowT+VHighT)/2
        
        VPoly = Polynomial(V+VOut-Vref, self.grid)
        dzdchi,_,_ = self.grid.getCompactificationDerivatives()
        U = VPoly.integrate(w = dzdchi)
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
                          offEquilDeltas: list, Tplus: float, Tminus: float) -> Tuple[np.ndarray, np.ndarray]:
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

        ## TODO can this loop be numpified?
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

        ## What's going on in this function? Please explain your logic. TODO issue #114

        Tout30, Tout33 = self.deltaToTmunu(index, fields, velocityMid, offEquilDeltas)
        s1 = c1 - Tout30
        s2 = c2 - Tout33

        ## TODO figure out better bounds
        # HACK! shoved in this Tc as a short term hack
        Tc = self.thermo.findCriticalTemperature(
            dT=0.1, rTol=1e-6, paranoid=True,
        )
        #(TMin, TMax) = self.thermo.getCoexistenceRange()
        minRes = scipy.optimize.minimize_scalar(
            lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2),
            method='Bounded',
            bounds=[0, Tc],
        )
        # TODO: A fail safe

        ## Whats this? shouldn't we check that LHS == 0 ?
        if self.temperatureProfileEqLHS(fields, dPhidz, minRes.x, s1, s2) >= 0:
            T = minRes.x
            vPlasma = self.plasmaVelocity(fields, T, s1)
            return T, vPlasma


        TLowerBound = minRes.x
        TStep = np.abs(Tplus - TLowerBound)
        if TStep == 0:
            TStep = np.abs(Tminus - TLowerBound)

        TUpperBound = TLowerBound + TStep
        while self.temperatureProfileEqLHS(fields, dPhidz, TUpperBound, s1, s2) < 0:
            TStep *= 2
            TUpperBound = TLowerBound + TStep


        res = scipy.optimize.brentq(
            lambda T: self.temperatureProfileEqLHS(fields, dPhidz, T, s1, s2),
            TLowerBound,
            TUpperBound,
            xtol=1e-9,  # all these hard-coded tolerances are aweful
            rtol=1e-9, ## really???
        )
        # TODO: Can the function have multiple zeros?

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
        ## Need "enthalpy" but ouside a free-energy minimum! More precisely, eq (12) in the ref. So hack it here
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
        ## Need "enthalpy" but ouside a free-energy minimum! More precisely, eq (12) in the ref. So hack it here
        w = -T * self.thermo.effectivePotential.derivT(fields, T)

        ## TODO probably force axis here. But need to guarantee correct field type first, so need some refactoring
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

        ## Where do these come from?
        ## LN: needs rewrite, what happens with many out-of-eq particles?
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



    #### -------------- Begin old stuff, do not use! and TODO cleanup


    ## DEPRECATED, use findWallVelocityMinimizeAction() instead!
    # Jorinde: "findWallVelocityLoop was written first, but it didn't work so well, so Benoit wrote the other functiion" 
"""     def findWallVelocityLoop(self):
        '''
        Finds the wall velocity by solving hydrodynamics, the Boltzmann equation and
        the field equation of motion iteratively.
        '''

        # Initial conditions for velocity, hydro boundaries, wall parameters and
        # temperature profile

        if self.wallVelocityLTE < 1:
            wallVelocity = 0.9 * self.wallVelocityLTE
            maxWallVelocity = self.wallVelocityLTE
        else:
            wallVelocity = np.sqrt(1 / 3)
            maxWallVelocity = self.hydro.vJ

        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)

        wallWidthsGuess = (5/self.Tnucl)*np.ones(self.nbrFields)
        wallOffsetsGuess = np.zeros(self.nbrFields-1)
        wallWidths, wallOffsets = wallWidthsGuess, wallOffsetsGuess

        wallParameters = np.concatenate(([wallVelocity], wallWidths, wallOffsets))

        wallParameters = np.array([0.6,0.04,0.04,0.2])

        print(wallParameters)

        offEquilDeltas = {
            "00": np.zeros(self.grid.M-1),
            "02": np.zeros(self.grid.M-1),
            "20": np.zeros(self.grid.M-1),
            "11": np.zeros(self.grid.M-1)}

        error = self.errTol + 1
        while error > self.errTol:

            oldWallVelocity = wallParameters[0]
            oldWallWidths = wallParameters[1:1+self.nbrFields]
            oldWallOffsets = wallParameters[1+self.nbrFields:]
            oldError = error

            wallVelocity = wallParameters[0]
            wallWidths = wallParameters[1:self.nbrFields+1]
            wallOffsets = wallParameters[self.nbrFields+1:]

            c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)

            vevLowT = self.thermo.freeEnergyLow(Tminus)[:-1]
            vevHighT = self.thermo.freeEnergyHigh(Tplus)[:-1]

            fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)

            Tprofile, velocityProfile = self.findPlasmaProfile(c1, c2, velocityMid, fields, dPhidz, offEquilDeltas, Tplus, Tminus)

            boltzmannBackground = BoltzmannBackground(velocityMid, velocityProfile, fields, Tprofile)

            boltzmannSolver = BoltzmannSolver(self.grid, boltzmannBackground, self.particle)

            # TODO: getDeltas() is not working at the moment (it returns nan), so I turned it off to debug the rest of the loop.
            #print('NOTE: offEquilDeltas has been set to 0 to debug the main loop.')
            offEquilDeltas = boltzmannSolver.getDeltas()

            # for i in range(2): # Can run this loop several times to increase the accuracy of the approximation
            #     wallParameters = initialEOMSolution(wallParameters, offEquilDeltas, freeEnergy, hydro, particle, grid)
            #     print(f'Intermediate result: {wallParameters=}')

            intermediateRes = root(self.momentsOfWallEoM, wallParameters, args=(offEquilDeltas,))
            print(intermediateRes)

            wallParameters = intermediateRes.x

            error = 0#np.sqrt((1 - oldWallVelocity/wallVelocity)**2 + np.sum((1 - oldWallWidths/wallWidths)**2) + np.sum((wallOffsets - oldWallOffsets) ** 2))

        return wallParameters
    

    def momentsOfWallEoM(self, wallParameters, offEquilDeltas):
        wallVelocity = wallParameters[0]
        wallWidths = wallParameters[1:self.nbrFields+1]
        wallOffsets = wallParameters[self.nbrFields+1:]
        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.thermo.freeEnergyLow(Tminus)[:-1]
        vevHighT = self.thermo.freeEnergyHigh(Tplus)[:-1]

        fields, dPhidz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        Tprofile, vprofile = self.findPlasmaProfile(c1, c2, velocityMid, fields, dPhidz, offEquilDeltas, Tplus, Tminus)

        # Define a function returning the local temparature by interpolating through Tprofile.
        Tfunc = UnivariateSpline(self.grid.xiValues, Tprofile, k=3, s=0)

        # Define a function returning the local Delta00 function by interpolating through offEquilDeltas['00'].
        offEquilDelta00 = UnivariateSpline(self.grid.xiValues, offEquilDeltas['00'], k=3, s=0)

        pressures = self.pressureMoment(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, offEquilDelta00)
        stretchs = self.stretchMoment(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, offEquilDelta00)

        return np.append(pressures, stretchs)

    def equationOfMotions(self, fields, T, offEquilDelta00):
        dVdX = self.thermo.effectivePotential.derivField(fields, T)

        # TODO: need to generalize to more than 1 particle.
        def dmtdh(Y):
            Y = np.asanyarray(Y)
            # TODO: Would be nice to compute the mass derivative directly in particle.
            return derivative(lambda x: self.particle.msqVacuum(np.append(x,Y[1:])), Y[0], dx = 1e-3, n=1, order=4)

        dmtdX = np.zeros_like(fields)
        dmtdX[0] = dmtdh(fields)
        offEquil = 0.5 * 12 * dmtdX * offEquilDelta00

        return dVdX + offEquil

    def pressureLocal(self, z, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        fields, dPhidz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(fields, Tfunc(z), Delta00func(z))
        return -dPhidz*EOM

    def pressureMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        return quad_vec(self.pressureLocal, -1, 1, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]

    def stretchLocal(self, z, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        fields, dPhidz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(fields, Tfunc(z), Delta00func(z))

        return dPhidz*(2*(fields-vevLowT)/(vevHighT-vevLowT)-1)*EOM

    def stretchMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        kinetic = (2/15)*(vevHighT-vevLowT)**2/wallWidths**2
        return kinetic + quad_vec(self.stretchLocal, -np.inf, np.inf, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]
     """
