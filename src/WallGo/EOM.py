import numpy as np
from dataclasses import dataclass
from typing import Tuple

import scipy.optimize

from scipy.optimize import minimize, minimize_scalar, brentq, root, root_scalar
from scipy.integrate import quad_vec,quad
from scipy.interpolate import UnivariateSpline
from .Polynomial2 import Polynomial
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .GenericModel import GenericModel
from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .helpers import derivative, gammaSq, GCLQuadrature # derivatives for callable functions
from .Particle import Particle
from .Fields import Fields
from .Grid import Grid


@dataclass
class WallParams():
    ## Holds wall widths and wall offsets for all fields
    widths: np.ndarray ## 1D array
    offsets: np.ndarray ## 1D array


class EOM:

    model: GenericModel
    hydro: Hydro
    thermo: Thermodynamics

    """LN: Very counterintuitive that this requires particle input even if includeOffEq=False. Here are some things to consider:
        1. Is it possible to remove includeOffEq from the constructor and instead have a dedicated function for solving the EOM
        without out-of-eq contributions?
        2. If most of all functions here are considerably simpler when out-of-eq are not included, should there be a separate (child?) class 
        for handling that case?
        3. Would this class make sense if by default it doesn't have a particle associated with it. 
        Could they instead be added on demand at runtime on demand?
    """

    """
    Class that solves the energy-momentum conservation equations and the scalar EOMs to determine the wall velocity.
    """
    def __init__(self, particle: Particle, thermodynamics: Thermodynamics, hydro: Hydro, grid: Grid, nbrFields: int, includeOffEq: bool=False, errTol=1e-6):
        """
        Initialization

        Parameters
        ----------
        particle : Particle
            Object of the class Particle, which contains the information about
            the out-of-equilibrium particles for which the Boltzmann equation
            will be solved.
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
            Error tolerance. The default is 1e-6.

        Returns
        -------
        None.

        """
        self.particle = particle
        self.grid = grid
        self.errTol = errTol
        self.nbrFields = nbrFields
        self.includeOffEq = includeOffEq
        
        self.thermo = thermodynamics
        self.hydro = hydro
        # I feel this is error prone: we should always read Tnucl from self.thermo
        self.Tnucl = self.thermo.Tnucl

    ## OLD STUFF, use findWallVelocityMinimizeAction() instead! 
    # Jorinde: "findWallVelocityLoop was written first, but it didn't work so well, so Benoit wrote the other functiion"
    '''
    def findWallVelocityLoop(self):
        """
        Finds the wall velocity by solving hydrodynamics, the Boltzmann equation and
        the field equation of motion iteratively.
        """

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

            fields, dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)

            Tprofile, velocityProfile = self.findPlasmaProfile(c1, c2, velocityMid, fields, dXdz, offEquilDeltas, Tplus, Tminus)

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
    '''
    


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
        # ## ??????????? why is wallOffsets different size?
        # wallWidths = (5/self.Tnucl)*np.ones(self.nbrFields)
        # wallOffsets = np.zeros(self.nbrFields-1)
        
        ## I guess we here give initial guesses for wallWidths and wallOffsets. Offset = how much field i is shifted from field 0 in position space
        ## For some reason offsets was different length previously

        wallParams = WallParams(widths = (5/self.Tnucl)*np.ones(self.nbrFields), 
                                offsets = np.zeros(self.nbrFields))

        alpha = self.thermo.alpha(self.Tnucl)
        vmin = max(1-(3*alpha)**(-10./13.),0.01) #based on eq (103) of 1004.4187

        return self.solvePressure(vmin, self.hydro.vJ-1e-6, wallParams)
    

    def solvePressure(self, wallVelocityMin: float, wallVelocityMax: float, wallParams: WallParams):
        r"""
        Solves the equation :math:`P_{\rm tot}(\xi_w)=0` for the wall velocity.

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
        wallParams : WallParams
            Contains a guess of the wall thicknesses and wall offsets.

        Returns
        -------
        wallVelocity : double
            Value of the wall velocity that solves the scalar EOMs.
        wallParams : WallParams
            Array containing the wall thicknesses and wall offsets that
            minimize the action and solve the EOM.

        """
        pressureMax, wallParamsMax = self.pressure(wallVelocityMax, wallParams, True)
        if pressureMax < 0:
            print('Maximum pressure is negative!')
            print(f"{pressureMax=} {wallParamsMax=}")
            return 1
        pressureMin, wallParamsMin = self.pressure(wallVelocityMin, wallParams, True)
        if pressureMin > 0:
            return 0

        def pressureWrapper(vw, flag=False):
            if vw == wallVelocityMin:
                return pressureMin
            if vw == wallVelocityMax:
                return pressureMax

            return self.pressure(vw, wallParamsMin+(wallParamsMax-wallParamsMin)*(vw-wallVelocityMin)/(wallVelocityMax-wallVelocityMin), flag)

        wallVelocity = root_scalar(pressureWrapper, method='brentq', bracket=[wallVelocityMin,wallVelocityMax], xtol=1e-3).root
        _,wallParams = pressureWrapper(wallVelocity, True)
        return wallVelocity, wallParams


    def pressure(self, wallVelocity: float, wallParams: WallParams, returnOptimalWallParams=False):
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action.

        Parameters
        ----------
        wallVelocity : double
            Wall velocity at which the pressure is computed.
        wallParams : WallParams
            Contains a guess of the wall thicknesses and wall offsets.
        returnOptimalWallParams : bool, optional
            If False, only the pressure is returned. If True, both the pressure
            and optimal wall parameters are returned. The default is False.

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

        zeroPoly = Polynomial(np.zeros(self.grid.M-1), self.grid)
        offEquilDeltas = {"00": zeroPoly, "02": zeroPoly, "20": zeroPoly, "11": zeroPoly}

        # TODO: Solve the Boltzmann equation to update offEquilDeltas.

        print(f"{wallVelocity=}")
        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.thermo.freeEnergyLow(Tminus).getFields()
        vevHighT = self.thermo.freeEnergyHigh(Tplus).getFields()

        ## No idea whats going on in this loop!
        i = 0
        # TODO: Implement a better condition
        while i < 2:
    
            fields, dXdz = self.wallProfile(
                self.grid.xiValues, vevLowT, vevHighT, wallParams
            )

            Tprofile, velocityProfile = self.findPlasmaProfile(
                c1, c2, velocityMid, fields, dXdz, offEquilDeltas, Tplus, Tminus
            )

            if self.includeOffEq:
                TWithEndpoints = np.concatenate(([Tminus], Tprofile, [Tplus]))
                XWithEndpoints = np.concatenate((vevLowT[:,None], fields, vevHighT[:,None]), 1)
                vWithEndpoints = np.concatenate(([velocityProfile[0]], velocityProfile, [velocityProfile[-1]]))
                boltzmannBackground = BoltzmannBackground(velocityMid, vWithEndpoints, XWithEndpoints, TWithEndpoints) 
                boltzmannSolver = BoltzmannSolver(self.grid, boltzmannBackground, self.particle)
                offEquilDeltas = boltzmannSolver.getDeltas()  #This gives an error


            ## Next need to solve wallWidth and wallOffset(?). scipy should work with 2D array input so no need for a loop. 
            ## For this we wallParams in a numpy 2D array:
            wallArray = np.concatenate( (wallParams.widths, wallParams.offsets) ) 
            
            ## Where do the bounds here come from!?
            
            ## first width, then offset
            lowerBounds = np.concatenate((self.nbrFields * [0.1 / self.Tnucl] , self.nbrFields * [-10.] ))
            upperBounds = np.concatenate((self.nbrFields * [100. / self.Tnucl] , self.nbrFields * [10.] ))
            bounds = scipy.optimize.Bounds(lb = lowerBounds, ub = upperBounds)

            ## And then a wrapper that puts the inputs back in WallParams (could maybe bypass this somehow...?)
            def actionWrapper(wallArray: np.ndarray, *args) -> float:
                wallParams = WallParams(widths = wallArray[:self.nbrFields], offsets = wallArray[self.nbrFields:])
                return self.action(wallParams, *args)
            
            ## LN: I think the old version somehow manages to vectorize this...? but it was incomprehensible. TODO...?

            sol = minimize(actionWrapper, wallArray, args=(vevLowT, vevHighT, Tprofile, offEquilDeltas['00']), method='Nelder-Mead', bounds=bounds)

            print(sol)
            input()
            wallParams = sol.x
            i += 1

        wallWidths = wallParams[:self.nbrFields]
        wallOffsets = wallParams[self.nbrFields:]
        X, dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        dVdX = self.thermo.effectivePotential.derivField(X, Tprofile)

        # TODO: Add the mass derivative in the Particle class and use it here.
        dVout = 12*X[0]*offEquilDeltas['00'].coefficients/2 
        EOMPoly = Polynomial((dVdX*dXdz)[0]+dVout*dXdz[0], self.grid)
        print(f"{dVdX=}, {dXdz}")
        pressure = EOMPoly.integrate(w=-self.grid.L_xi/(1-self.grid.chiValues**2)**1.5)

        print(f"{pressure=}")

        if returnOptimalWallParams:
            return pressure,wallParams
        else:
            return pressure

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
        #wallWidths = wallParams[:self.nbrFields]
        #wallOffsets = wallParams[self.nbrFields:]
        wallWidths = wallParams.widths

        #fields, dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        fields, dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallParams)


        # TODO had to put here diagonal since fieldsi (2,N) and Tprofile (N) gave (N,N) result
        V = np.diag(self.thermo.effectivePotential.evaluate(fields, Tprofile))
        VOut = 12*self.particle.msqVacuum(fields)*offEquilDelta00.coefficients/2

        VLowT = self.thermo.effectivePotential.evaluate(vevLowT,Tprofile[0])
        VHighT = self.thermo.effectivePotential.evaluate(vevHighT,Tprofile[-1])

        Vref = (VLowT+VHighT)/2
        
        ## Dunno whats going on here. Polynomial.integrate() here should always return a Polynomial object according to usage, 
        ## which definitely should not work with scipy.optimize.minimize(). But why did it work in the old version then?
        
        VPoly = Polynomial(V+VOut-Vref, self.grid)
        U = VPoly.integrate(w=self.grid.L_xi/(1-self.grid.chiValues**2)**1.5)
        K = np.sum((vevHighT-vevLowT)**2/(6*wallWidths))

        res: Polynomial = U + K        

        ## force result to float. Dunno if this is a good way
        res = res.coefficients

        return res


    def momentsOfWallEoM(self, wallParameters, offEquilDeltas):
        wallVelocity = wallParameters[0]
        wallWidths = wallParameters[1:self.nbrFields+1]
        wallOffsets = wallParameters[self.nbrFields+1:]
        c1, c2, Tplus, Tminus, velocityMid = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.thermo.freeEnergyLow(Tminus)[:-1]
        vevHighT = self.thermo.freeEnergyHigh(Tplus)[:-1]

        fields, dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        Tprofile, vprofile = self.findPlasmaProfile(c1, c2, velocityMid, fields, dXdz, offEquilDeltas, Tplus, Tminus)

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
        fields, dXdz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(fields, Tfunc(z), Delta00func(z))
        return -dXdz*EOM

    def pressureMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        return quad_vec(self.pressureLocal, -1, 1, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]

    def stretchLocal(self, z, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        fields, dXdz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(fields, Tfunc(z), Delta00func(z))

        return dXdz*(2*(fields-vevLowT)/(vevHighT-vevLowT)-1)*EOM

    def stretchMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        kinetic = (2/15)*(vevHighT-vevLowT)**2/wallWidths**2
        return kinetic + quad_vec(self.stretchLocal, -np.inf, np.inf, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]

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
        dXdz : array-like
            Derivative with respect to the position of the scalar field profile.

        """
        if np.isscalar(z):
            z_L = z / wallParams.widths
        else:
            ## Broadcast mess needed
            z_L = z[:,None] / wallParams.widths[None,:]

        ## Not needed: wallParams.offsets has already len = nbrFields
        #wallOffsetsCompleted = np.append([0], wallOffsets)

        ## do these need transpose? old version had that
        fields = vevLowT + 0.5*(vevHighT - vevLowT) * (1 + np.tanh( z_L + wallParams.offsets ))
        dXdz = 0.5*(vevHighT-vevLowT) / ( wallParams.widths * np.cosh(z_L + wallParams.offsets)**2 )

        return fields, dXdz

    def findPlasmaProfile(self, c1: float , c2: float, velocityMid: float, fields: Fields, dXdz: Fields, 
                          offEquilDeltas: dict[str, float], Tplus: float, Tminus: float) -> Tuple[np.ndarray, np.ndarray]:
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
            Scalar field profile.
        dXdz : array-like
            Derivative with respect to the position of the scalar field profile.
        offEquilDeltas : dictionary
            Dictionary containing the off-equilibrium Delta functions
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
            T, vPlasma = self.findPlasmaProfilePoint(index, c1, c2, velocityMid, fields.GetFieldPoint(index), dXdz.GetFieldPoint(index), offEquilDeltas, Tplus, Tminus)

            temperatureProfile[index] = T
            velocityProfile[index] = vPlasma

        return temperatureProfile, velocityProfile

    def findPlasmaProfilePoint(self, index: int, c1: float, c2: float, velocityMid: float, fields: Fields, dXdz: Fields, 
                               offEquilDeltas: dict[str, float], Tplus: float, Tminus: float) -> Tuple[float, float]:
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
        dXdz : Fields
            Derivative with respect to the position of the scalar field profile.
        offEquilDeltas : dictionary
            Dictionary containing the off-equilibrium Delta functions
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

        Tout30, Tout33 = self.deltaToTmunu(index, fields, velocityMid, offEquilDeltas)

        s1 = c1 - Tout30
        s2 = c2 - Tout33

        ## LN shouldn't have the bounds hardcoded here
        minRes = minimize_scalar(lambda T: self.temperatureProfileEqLHS(fields, dXdz, T, s1, s2), method='Bounded', bounds=[0,self.thermo.Tc])
        # TODO: A fail safe

        if self.temperatureProfileEqLHS(fields, dXdz, minRes.x, s1, s2) >= 0:
            T = minRes.x
            vPlasma = self.plasmaVelocity(fields, T, s1)
            return T, vPlasma

        TLowerBound = minRes.x
        TStep = np.abs(Tplus - TLowerBound)
        if TStep == 0:
            TStep = np.abs(Tminus - TLowerBound)

        TUpperBound = TLowerBound + TStep
        while self.temperatureProfileEqLHS(fields, dXdz, TUpperBound, s1, s2) < 0:
            TStep *= 2
            TUpperBound = TLowerBound + TStep

        res = brentq(
            lambda T: self.temperatureProfileEqLHS(fields, dXdz, T, s1, s2),
            TLowerBound,
            TUpperBound,
            xtol=1e-9,
            rtol=1e-9,
        )
        # TODO: Can the function have multiple zeros?

        T = res   #is this okay? #Maybe not? Sometimes it returns an array, sometimes a double
        vPlasma = self.plasmaVelocity(fields, T, s1)
        return T, vPlasma

    def plasmaVelocity(self, fields, T, s1):
        r"""
        Computes the plasma velocity as a function of the temperature.

        Parameters
        ----------
        fields : array-like
            Scalar field profile.
        T : double
            Temparature.
        s1 : double
            Value of :math:`T^{30}-T_{\rm out}^{30}`.

        Returns
        -------
        double
            Plasma velocity.

        """
        dVdT = self.thermo.effectivePotential.derivT(fields, T)
        return (T * dVdT  + np.sqrt(4 * s1**2 + (T * dVdT)**2)) / (2 * s1)

    def temperatureProfileEqLHS(self, fields: Fields, dXdz: Fields, T: float, s1: float, s2: float):
        r"""
        The LHS of Eq. (20) of arXiv:2204.13120v1.

        Parameters
        ----------
        fields : array-like
            Scalar field profile.
        dXdz : array-like
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
        result = (
            + 0.5*np.sum(dXdz**2, axis=0)
            - self.thermo.effectivePotential.evaluate(fields, T)
            + 0.5*T*self.thermo.effectivePotential.derivT(fields, T)
            + 0.5*np.sqrt(4*s1**2
            + (T*self.thermo.effectivePotential.derivT(fields, T))**2)
            - s2
        )
        result = np.asarray(result)
        if result.shape == (1,) and len(result) == 1:
            return result[0]
        elif result.shape == ():
            return result
        else:
            raise TypeError(f"LHS has wrong type, {result.shape=}")


    def deltaToTmunu(self, index: int, fields: Fields, velocityMid: float, offEquilDeltas: dict[str, float]) -> Tuple[float, float]:
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
        offEquilDeltas : dictionary
            Dictionary containing the off-equilibrium Delta functions

        Returns
        -------
        T30 : double
            Out-of-equilibrium part of :math:`T^{30}`.
        T33 : double
            Out-of-equilibrium part of :math:`T^{33}`.

        """
        delta00 = offEquilDeltas["00"].coefficients[index]
        delta11 = offEquilDeltas["11"].coefficients[index]
        delta02 = offEquilDeltas["02"].coefficients[index]
        delta20 = offEquilDeltas["20"].coefficients[index]

        u0 = np.sqrt(gammaSq(velocityMid))
        u3 = np.sqrt(gammaSq(velocityMid))*velocityMid
        ubar0 = u3
        ubar3 = u0

        ## Where do these come from?
        T30 = (
            + (3*delta20 - delta02 - self.particle.msqVacuum(fields)*delta00)*u3*u0
            + (3*delta02 - delta20 + self.particle.msqVacuum(fields)*delta00)*ubar3*ubar0
            + 2*delta11*(u3*ubar0 + ubar3*u0))/2.
        T33 = ((
            + (3*delta20 - delta02 - self.particle.msqVacuum(fields)*delta00)*u3*u3
            + (3*delta02 - delta20 + self.particle.msqVacuum(fields)*delta00)*ubar3*ubar3
            + 4*delta11*u3*ubar3)/2. 
            - (self.particle.msqVacuum(fields)*delta00+ delta02-delta20)/2.)

        return T30, T33