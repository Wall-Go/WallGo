import numpy as np

from scipy.optimize import minimize, minimize_scalar, brentq, root, root_scalar
from scipy.integrate import quad_vec,quad
from scipy.interpolate import UnivariateSpline
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .model import Particle, FreeEnergy
from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .helpers import derivative, gammaSq, GCLQuadrature # derivatives for callable functions

class EOM:
    """
    Class that solves the energy-momentum conservation equations and the scalar EOMs to determine the wall velocity.
    """
    def __init__(self, particle, freeEnergy, grid, nbrFields, includeOffEq=False, errTol=1e-6):
        """
        Initialization 

        Parameters
        ----------
        particle : Particle
            Object of the class Particle, which contains the information about 
            the out-of-equilibrium particles for which the Boltzmann equation 
            will be solved.
        freeEnergy : FreeEnergy
            Object of the class FreeEnergy.
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
        self.freeEnergy = freeEnergy
        self.grid = grid
        self.errTol = errTol
        self.nbrFields = nbrFields
        self.Tnucl = freeEnergy.Tnucl
        self.includeOffEq = includeOffEq

        self.thermo = Thermodynamics(freeEnergy)
        self.hydro = Hydro(self.thermo)
        self.wallVelocityLTE = self.hydro.findvwLTE()

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

        c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)

        wallWidthsGuess = (5/self.Tnucl)*np.ones(self.nbrFields)
        wallOffsetsGuess = np.zeros(self.nbrFields-1)
        wallWidths, wallOffsets = wallWidthsGuess, wallOffsetsGuess

        wallParameters = np.concatenate(([wallVelocity], wallWidths, wallOffsets))

        wallParameters = np.array([0.6,0.04,0.04,0.2])

        print(wallParameters)

        offEquilDeltas = {"00": np.zeros(self.grid.M-1), "02": np.zeros(self.grid.M-1), "20": np.zeros(self.grid.M-1), "11": np.zeros(self.grid.M-1)}

        error = self.errTol + 1
        while error > self.errTol:

            oldWallVelocity = wallParameters[0]
            oldWallWidths = wallParameters[1:1+self.nbrFields]
            oldWallOffsets = wallParameters[1+self.nbrFields:]
            oldError = error

            wallVelocity = wallParameters[0]
            wallWidths = wallParameters[1:self.nbrFields+1]
            wallOffsets = wallParameters[self.nbrFields+1:]

            c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)

            vevLowT = self.freeEnergy.findPhases(Tminus)[0]
            vevHighT = self.freeEnergy.findPhases(Tplus)[1]

            wallProfileGrid = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)

            Tprofile, velocityProfile = self.findPlasmaProfile(c1, c2, velocityAtz0, vevLowT, vevHighT, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)

            boltzmannBackground = BoltzmannBackground(wallParameters[0], velocityProfile, wallProfileGrid, Tprofile)

            boltzmannSolver = BoltzmannSolver(self.grid, boltzmannBackground, self.particle)

            # TODO: getDeltas() is not working at the moment (it returns nan), so I turned it off to debug the rest of the loop.
            print('NOTE: offEquilDeltas has been set to 0 to debug the main loop.')
            # offEquilDeltas = boltzmannSolver.getDeltas()
            # print(offEquilDeltas)

            # for i in range(2): # Can run this loop several times to increase the accuracy of the approximation
            #     wallParameters = initialEOMSolution(wallParameters, offEquilDeltas, freeEnergy, hydro, particle, grid)
            #     print(f'Intermediate result: {wallParameters=}')

            intermediateRes = root(self.momentsOfWallEoM, wallParameters, args=(offEquilDeltas,))
            print(intermediateRes)

            wallParameters = intermediateRes.x

            error = 0#np.sqrt((1 - oldWallVelocity/wallVelocity)**2 + np.sum((1 - oldWallWidths/wallWidths)**2) + np.sum((wallOffsets - oldWallOffsets) ** 2))

        return wallParameters

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
        wallWidths = (5/self.Tnucl)*np.ones(self.nbrFields)
        wallOffsets = np.zeros(self.nbrFields-1)
        
        alpha = self.thermo.alpha(self.Tnucl)
        vmin = max(1-(3*alpha)**(-10/13),0.01) #based on eq (103) of 1004.4187

        return self.solvePressure(vmin, self.hydro.vJ-1e-6, np.append(wallWidths, wallOffsets))

    def solvePressure(self, wallVelocityMin, wallVelocityMax, wallParams):
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
        wallParams : array_like
            Array containing a guess of the wall thicknesses and wall offsets.

        Returns
        -------
        wallVelocity : double
            Value of the wall velocity that solves the scalar EOMs.
        wallParams : array-like
            Array containing the wall thicknesses and wall offsets that 
            minimize the action and solve the EOM.

        """
        pressureMax,wallParamsMax = self.pressure(wallVelocityMax, wallParams, True)
        if pressureMax < 0:
            print('Maximum pressure is negative!')
            print(f"{pressureMax=} {wallParamsMax=}")
            return 1
        pressureMin,wallParamsMin = self.pressure(wallVelocityMin, wallParams, True)
        if pressureMin > 0:
            return 0

        def func(vw, flag=False):
            if vw == wallVelocityMin:
                return pressureMin
            if vw == wallVelocityMax:
                return pressureMax

            return self.pressure(vw, wallParamsMin+(wallParamsMax-wallParamsMin)*(vw-wallVelocityMin)/(wallVelocityMax-wallVelocityMin), flag)

        wallVelocity = root_scalar(func, method='brentq', bracket=[wallVelocityMin,wallVelocityMax], xtol=1e-3).root
        _,wallParams = func(wallVelocity, True)
        return wallVelocity, wallParams

    def pressure(self, wallVelocity, wallParams, returnOptimalWallParams=False):
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action.

        Parameters
        ----------
        wallVelocity : double
            Wall velocity at which the pressure is computed.
        wallParams : array-like
            Array containing a guess of the wall thicknesses and wall offsets.
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
        offEquilDeltas = {"00": np.zeros(self.grid.M-1), "02": np.zeros(self.grid.M-1), "20": np.zeros(self.grid.M-1), "11": np.zeros(self.grid.M-1)}

        # TODO: Solve the Boltzmann equation to update offEquilDeltas.

        c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.freeEnergy.findPhases(Tminus)[0]
        vevHighT = self.freeEnergy.findPhases(Tplus)[1]

        i = 0
        # TODO: Implement a better condition
        while i < 1:
            wallWidths = wallParams[:self.nbrFields]
            wallOffsets = wallParams[self.nbrFields:]
            wallProfileGrid = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
            Tprofile, velocityProfile = self.findPlasmaProfile(c1, c2, velocityAtz0, vevLowT, vevHighT, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)

            if self.includeOffEq:
                boltzmannBackground = BoltzmannBackground(0, velocityProfile, wallProfileGrid, Tprofile) #first entry is 0 because that's the wall velocity in the wall frame
                boltzmannSolver = BoltzmannSolver(self.grid, boltzmannBackground, self.particle)
                offEquilDeltas = boltzmannSolver.getDeltas()  #This gives an error

            sol = minimize(self.action, wallParams, args=(vevLowT, vevHighT, Tprofile, offEquilDeltas), method='Nelder-Mead', bounds=self.nbrFields*[(0.1/self.Tnucl,100/self.Tnucl)]+(self.nbrFields-1)*[(-10,10)])
            wallParams = sol.x
            i += 1

        wallWidths = wallParams[:self.nbrFields]
        wallOffsets = wallParams[self.nbrFields:]
        X,dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        dVdX = self.freeEnergy.derivField(X, Tprofile)
        pressure = -GCLQuadrature(np.concatenate(([0], self.grid.L_xi*(dVdX*dXdz)[0]/(1-self.grid.chiValues**2), [0])))

        if returnOptimalWallParams:
            return pressure,wallParams
        else:
            return pressure

    def action(self, wallParams, vevLowT, vevHighT, Tprofile, offEquilDeltas):
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
        offEquilDeltas : dictionary
            Dictionary containing the off-equilibrium Delta functions
            
        Returns
        -------
        action : double
            Action spent by the scalar field configuration.

        """
        wallWidths = wallParams[:self.nbrFields]
        wallOffsets = wallParams[self.nbrFields:]

        X,dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        V = self.freeEnergy(X, Tprofile)
        VOut = self.particle.msqVacuum(X)*offEquilDeltas['00']

        VLowT,VHighT = self.freeEnergy(vevLowT,Tprofile[0]),self.freeEnergy(vevHighT,Tprofile[-1])

        Vref = (VLowT+VHighT)/2 
        
        U = GCLQuadrature(np.concatenate(([0], self.grid.L_xi*(V+VOut-Vref)/(1-self.grid.chiValues**2), [0])))
        K = np.sum((vevHighT-vevLowT)**2/(6*wallWidths))
        return (U+K)
        
        
    def momentsOfWallEoM(self, wallParameters, offEquilDeltas):
        wallVelocity = wallParameters[0]
        wallWidths = wallParameters[1:self.nbrFields+1]
        wallOffsets = wallParameters[self.nbrFields+1:]
        c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.freeEnergy.findPhases(Tminus)[0]
        vevHighT = self.freeEnergy.findPhases(Tplus)[1]
        Tprofile, vprofile = self.findPlasmaProfile(c1, c2, velocityAtz0, vevLowT, vevHighT, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)

        # Define a function returning the local temparature by interpolating through Tprofile.
        Tfunc = UnivariateSpline(self.grid.xiValues, Tprofile, k=3, s=0)

        # Define a function returning the local Delta00 function by interpolating through offEquilDeltas['00'].
        offEquilDelta00 = UnivariateSpline(self.grid.xiValues, offEquilDeltas['00'], k=3, s=0)

        pressures = self.pressureMoment(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, offEquilDelta00)
        stretchs = self.stretchMoment(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, offEquilDelta00)

        return np.append(pressures, stretchs)

    def equationOfMotions(self, X, T, offEquilDelta00):
        dVdX = self.freeEnergy.derivField(X, T)

        # TODO: need to generalize to more than 1 particle.
        def dmtdh(Y):
            Y = np.asanyarray(Y)
            # TODO: Would be nice to compute the mass derivative directly in particle.
            return derivative(lambda x: self.particle.msqVacuum(np.append(x,Y[1:])), Y[0], dx = 1e-3, n=1, order=4)

        dmtdX = np.zeros_like(X)
        dmtdX[0] = dmtdh(X)
        offEquil = 0.5 * 12 * dmtdX * offEquilDelta00

        return dVdX + offEquil

    def pressureLocal(self, z, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        X,dXdz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(X, Tfunc(z), Delta00func(z))
        return -dXdz*EOM

    def pressureMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        return quad_vec(self.pressureLocal, -1, 1, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]

    def stretchLocal(self, z, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        X,dXdz = self.wallProfile(z, vevLowT, vevHighT, wallWidths, wallOffsets)

        EOM = self.equationOfMotions(X, Tfunc(z), Delta00func(z))

        return dXdz*(2*(X-vevLowT)/(vevHighT-vevLowT)-1)*EOM

    def stretchMoment(self, vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func):
        kinetic = (2/15)*(vevHighT-vevLowT)**2/wallWidths**2
        return kinetic + quad_vec(self.stretchLocal, -np.inf, np.inf, args=(vevLowT, vevHighT, wallWidths, wallOffsets, Tfunc, Delta00func))[0]

    def wallProfile(self, z, vevLowT, vevHighT, wallWidths, wallOffsets):
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
        X : array-like
            Scalar field profile.
        dXdz : array-like
            Derivative with respect to the position of the scalar field profile.

        """
        if np.isscalar(z):
            z_L = z/wallWidths
        else:
            z_L = z[:,None]/wallWidths[None,:]
        wallOffsetsCompleted = np.append([0], wallOffsets)

        X = np.transpose(vevLowT + 0.5*(vevHighT-vevLowT)*(1+np.tanh(z_L+wallOffsetsCompleted)))
        dXdz = np.transpose(0.5*(vevHighT-vevLowT)/(wallWidths*np.cosh(z_L+wallOffsetsCompleted)**2))

        return X, dXdz

    def findPlasmaProfile(self, c1, c2, velocityAtz0, vevLowT, vevHighT, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus):
        r"""
        Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of LHS.

        Parameters
        ----------
        c1 : double
            Value of the :math:`T^{30}` component of the energy-momentum tensor.
        c2 : double
            Value of the :math:`T^{33}` component of the energy-momentum tensor.
        velocityAtz0 : double
            Plasma velocity in the wall frame at :math:`z=0`.
        vevLowT : array-like
            Scalar field VEVs in the low-T phase.
        vevHighT : array-like
            Scalar field VEVs in the high-T phase.
        wallWidths : array-like
            Array containing the wall widths.
        wallOffsets : array-like
            Array containing the wall offsets.
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
        temperatureProfile = []
        velocityProfile = []

        X,dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)

        for index in range(len(self.grid.xiValues)):
            T, vPlasma = self.findPlasmaProfilePoint(index, c1, c2, velocityAtz0, X[:,index], dXdz[:,index], offEquilDeltas, Tplus, Tminus)

            temperatureProfile.append(T)
            velocityProfile.append(vPlasma)

        return np.array(temperatureProfile), np.array(velocityProfile)

    def findPlasmaProfilePoint(self, index, c1, c2, velocityAtz0, X, dXdz, offEquilDeltas, Tplus, Tminus):
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
        velocityAtz0 : double
            Plasma velocity in the wall frame at :math:`z=0`.
        X : array-like
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
        T : double
            Temperature at the point grid.xiValues[index].
        vPlasma : double
            Plasma velocity at the point grid.xiValues[index].

        """

        Tout30, Tout33 = self.deltaToTmunu(index,X,velocityAtz0,offEquilDeltas)

        s1 = c1 - Tout30
        s2 = c2 - Tout33

        minRes = minimize_scalar(lambda T: self.temperatureProfileEqLHS(X, dXdz, T, s1, s2), method='Bounded', bounds=[0,self.freeEnergy.Tc])
        # TODO: A fail safe

        if self.temperatureProfileEqLHS(X, dXdz, minRes.x, s1, s2) >= 0:
            T = minRes.x
            vPlasma = self.plasmaVelocity(X, T, s1)
            return T, vPlasma

        TLowerBound = minRes.x
        TStep = np.abs(Tplus - TLowerBound)
        if TStep == 0:
            TStep = np.abs(Tminus - TLowerBound)

        TUpperBound = TLowerBound + TStep
        while self.temperatureProfileEqLHS(X, dXdz, TUpperBound, s1, s2) < 0:
            TStep *= 2
            TUpperBound = TLowerBound + TStep

        res = brentq(
            lambda T: self.temperatureProfileEqLHS(X, dXdz, T, s1, s2),
            TLowerBound,
            TUpperBound,
            xtol=1e-9,
            rtol=1e-9,
        )
        # TODO: Can the function have multiple zeros?

        T = res   #is this okay?
        vPlasma = self.plasmaVelocity(X, T, s1)
        return T, vPlasma

    def plasmaVelocity(self, X, T, s1):
        r"""
        Computes the plasma velocity as a function of the temperature.

        Parameters
        ----------
        X : array-like
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
        dVdT = self.freeEnergy.derivT(X, T)
        return (T * dVdT  + np.sqrt(4 * s1**2 + (T * dVdT)**2)) / (2 * s1)

    def temperatureProfileEqLHS(self, X, dXdz, T, s1, s2):
        r"""
        The LHS of Eq. (20) of arXiv:2204.13120v1.

        Parameters
        ----------
        X : array-like
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
        return 0.5*np.sum(dXdz**2, axis=0) - self.freeEnergy(X, T) + 0.5*T*self.freeEnergy.derivT(X, T) + 0.5*np.sqrt(4*s1**2 + (T*self.freeEnergy.derivT(X, T))**2) - s2

    def deltaToTmunu(self, index, X, velocityAtCenter, offEquilDeltas):
        r"""
        Computes the out-of-equilibrium part of the energy-momentum tensor.

        Parameters
        ----------
        index : int
            Index of the grid point on which to find the plasma profile.
        X : array-like
            Scalar field profile.
        velocityAtCenter : double
            Plasma velocity in the wall frame at :math:`z=0`.
        offEquilDeltas : dictionary
            Dictionary containing the off-equilibrium Delta functions

        Returns
        -------
        T30 : double
            Out-of-equilibrium part of :math:`T^{30}`.
        T33 : double
            Out-of-equilibrium part of :math:`T^{33}`.

        """
        delta00 = offEquilDeltas["00"][index]
        delta11 = offEquilDeltas["11"][index]
        delta02 = offEquilDeltas["02"][index]
        delta20 = offEquilDeltas["20"][index]

        u0 = np.sqrt(gammaSq(velocityAtCenter))
        u3 = np.sqrt(gammaSq(velocityAtCenter))*velocityAtCenter
        ubar0 = u3
        ubar3 = u0

        T30 = ((3*delta20 - delta02 - self.particle.msqVacuum(X)*delta00)*u3*u0+
                (3*delta02 - delta20 + self.particle.msqVacuum(X)*delta00)*ubar3*ubar0+2*delta11*(u3*ubar0 + ubar3*u0))/2.
        T33 = ((3*delta20 - delta02 - self.particle.msqVacuum(X)*delta00)*u3*u3+
                (3*delta02 - delta20 + self.particle.msqVacuum(X)*delta00)*ubar3*ubar3+4*delta11*u3*ubar3)/2.

        return T30, T33
