import numpy as np
from scipy.optimize import minimize,root_scalar
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .EOM import EOM
from .Polynomial import Polynomial
from .helpers import GCLQuadrature
import matplotlib.pyplot as plt

class EOMGeneralShape:
    """
    Class that solves the energy-momentum conservation equations and the scalar 
    EOMs without using a tanh profile to determine the wall velocity.
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
        
        self.eom = EOM(particle, freeEnergy, grid, nbrFields, includeOffEq, errTol)
        
        self.polynomial = Polynomial(grid)
        self.deriv = self.polynomial.deriv('Cardinal', 'z', False)[None,:,:]
        
    def findWallVelocity(self):
        """
        Finds the wall velocity by minimizing the action and solving for the 
        solution with 0 total pressure on the wall.

        Returns
        -------
        wallVelocity : double
            Value of the wall velocity that solves the scalar EOMs.
        shape : array-like
            Deviation from the tanh profile in the scalar field profile.
        wallParams : array-like
            Array containing the wall thicknesses and wall offsets that 
            minimize the action and solve the EOM.

        """
        return self.solvePressure(0.01, self.hydro.vJ)
        
    def solvePressure(self, wallVelocityMin, wallVelocityMax):
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

        Returns
        -------
        wallVelocity : double
            Value of the wall velocity that solves the scalar EOMs.
        shape : array-like
            Deviation from the tanh profile in the scalar field profile.
        wallParams : array-like
            Array containing the wall thicknesses and wall offsets that 
            minimize the action and solve the EOM.

        """
        wallWidths = (5/self.Tnucl)*np.ones(self.nbrFields)
        wallOffsets = np.zeros(self.nbrFields-1)
        wallParams = np.append(wallWidths, wallOffsets)
        
        pressureMax,_,wallParamsMax = self.pressure(wallVelocityMax, wallParams, True)
        if pressureMax < 0:
            return 1,_,wallParamsMax
        pressureMin,_,wallParamsMin = self.pressure(wallVelocityMin, wallParams, True)
        if pressureMin > 0:
            return 0,_,wallParamsMin
        
        def func(vw, flag=False):
            if vw == wallVelocityMin:
                return pressureMin
            if vw == wallVelocityMax:
                return pressureMax
            
            return self.pressure(vw, wallParamsMin+(wallParamsMax-wallParamsMin)*(vw-wallVelocityMin)/(wallVelocityMax-wallVelocityMin), flag)
        
        wallVelocity = root_scalar(func, method='brentq', bracket=[wallVelocityMin,wallVelocityMax], xtol=1e-8).root
        _,shape,wallParams = func(wallVelocity, True)
        return wallVelocity, shape, wallParams
    
    def pressure(self, wallVelocity, wallParamsIni=None, returnOptimalWallParams=False):
        """
        Computes the total pressure on the wall by finding the tanh profile
        that minimizes the action.

        Parameters
        ----------
        wallVelocity : double
            Wall velocity at which the pressure is computed.
        wallParamsIni : array-like, optional
            Array containing a guess of the wall widths and wall offsets.
            If None, sets the initial widths and offsets to 5/Tn and 0, respectively.
            Default is None.
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
        if wallParamsIni is None:
            wallParamsIni = np.append(self.nbrFields*[5/self.Tnucl], (self.nbrFields-1)*[0])
            
        offEquilDeltas = {"00": np.zeros(self.grid.M-1), "02": np.zeros(self.grid.M-1), "20": np.zeros(self.grid.M-1), "11": np.zeros(self.grid.M-1)}
        
        # TODO: Solve the Boltzmann equation to update offEquilDeltas.
        
        c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)

        vevLowT = self.freeEnergy.findPhases(Tminus)[0]
        vevHighT = self.freeEnergy.findPhases(Tplus)[1]
        
        p,wallParams = self.eom.pressure(wallVelocity, wallParamsIni, True)
        wallWidths,wallOffsets = wallParams[:self.nbrFields],wallParams[self.nbrFields:]
        XIni,dXdzIni = self.eom.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        
        trunc = int(XIni.shape[1]) # Adjust this number to filter out high-frequency modes. This can be useful when aliasing becomes a problem.
        def chebToCard(spectralCoeff):
            a = np.zeros_like(XIni)
            a[:,:trunc] = np.concatenate(([0], spectralCoeff)).reshape((a.shape[0],trunc))
            n = np.arange(a.shape[1])+2
            meanPos = np.sum(np.where(n%2==0, -a[0]*2*n**2/(n**2-1), 0))
            a[0,0] = 3*meanPos/8
            cardinalCoeff = (self.polynomial.chebyshevMatrix('z')@a.T).T
            return cardinalCoeff
        
        def action(spectralCoeff, T):
            deltaShape = chebToCard(spectralCoeff)
            X = deltaShape + XIni
            dXdz = dXdzIni + np.sum(self.deriv*deltaShape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
            return self.action(X, dXdz, vevLowT, vevHighT, T, offEquilDeltas['00'])
        
        i = 0
        # TODO: Implement a better condition and update offEquilDeltas
        spectral = np.zeros(trunc*self.nbrFields-1)
        success = False
        while not success and i < 10:
            cardinal = chebToCard(spectral)
            X = XIni + cardinal
            dXdz = dXdzIni + np.sum(self.deriv*cardinal[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
            Tprofile, velocityProfile = self.eom.findPlasmaProfile(c1, c2, velocityAtz0, X, dXdz, offEquilDeltas, Tplus, Tminus)
            sol = minimize(action, spectral, method='Nelder-Mead', args=(Tprofile,))
            success = sol.success
            spectral = sol.x
            i += 1
        
        cardinal = chebToCard(spectral)
        X = XIni + cardinal
        dXdz = dXdzIni + np.sum(self.deriv*cardinal[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
        
        dVdX = self.freeEnergy.derivField(X, Tprofile)
        # TODO: Add out-of-equilibrium contribution
        pressure = -GCLQuadrature(np.concatenate(([0], self.grid.L_xi*np.sum(dVdX*dXdz,axis=0)/(1-self.grid.chiValues**2), [0])))
        
        if returnOptimalWallParams:
            return pressure,cardinal,wallParams
        else:
            return pressure
    
    def action(self, X, dXdz, vevLowT, vevHighT, Tprofile, offEquilDelta00):
        r"""
        Computes the action by using gaussian quadratrure to integrate the Lagrangian.

        Parameters
        ----------
        X : array-like
            Scalar field profile.
        dXdz : array-like
            Derivative with respect to the position of the scalar field profile.
        vevLowT : array-like
            Field values in the low-T phase.
        vevHighT : array-like
            Field values in the high-T phase.
        Tprofile : array-like
            Temperature profile on the grid.
        offEquilDelta00 : array-like
            Array containing the out-of-equilibrium function Delta00.
            
        Returns
        -------
        action : double
            Action spent by the scalar field configuration X.

        """
        
        V = self.freeEnergy(X, Tprofile)
        VOut = self.particle.msqVacuum(X)*offEquilDelta00/2
        
        VLowT,VHighT = self.freeEnergy(vevLowT,Tprofile[0]),self.freeEnergy(vevHighT,Tprofile[-1])
        Vref = VLowT + 0.5*(VHighT-VLowT)*(1+np.tanh(self.grid.xiValues/self.grid.L_xi))
        
        K = 0.5*np.sum(dXdz**2, axis=0)
        
        S = GCLQuadrature(np.concatenate(([0], self.grid.L_xi*(K+V+VOut-Vref)/(1-self.grid.chiValues**2), [0])))
        return S
    