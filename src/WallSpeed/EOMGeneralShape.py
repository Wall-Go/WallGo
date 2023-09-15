import numpy as np
from scipy.optimize import minimize,root_scalar
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .EOM import EOM
from .Polynomial import Polynomial
from .helpers import GCLQuadrature
import matplotlib.pyplot as plt

class EOMGeneralShape:
    def __init__(self, particle, freeEnergy, grid, nbrFields, errTol=1e-6):
        self.particle = particle
        self.freeEnergy = freeEnergy
        self.grid = grid
        self.errTol = errTol
        self.nbrFields = nbrFields
        self.Tnucl = freeEnergy.Tnucl
        
        self.thermo = Thermodynamics(freeEnergy)
        self.hydro = Hydro(self.thermo)
        self.wallVelocityLTE = self.hydro.findvwLTE()
        
        self.eom = EOM(particle, freeEnergy, grid, nbrFields, errTol)
        
        self.polynomial = Polynomial(grid)
        self.deriv = self.polynomial.deriv('Cardinal', 'z', False)[None,:,:]
        
    def findWallVelocity(self):
        return self.solvePressure(0.01, self.hydro.vJ)
        
    def solvePressure(self, wallVelocityMin, wallVelocityMax):
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
        
        wallVelocity = root_scalar(func, method='brentq', bracket=[wallVelocityMin,wallVelocityMax], xtol=1e-3).root
        _,shape,wallParams = func(wallVelocity, True)
        return wallVelocity, shape, wallParams
    
    def pressure(self, wallVelocity, wallParamsIni=None, returnOptimalWallParams=False):
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
        
        Tprofile, velocityProfile = self.eom.findPlasmaProfile(c1, c2, velocityAtz0, vevLowT, vevHighT, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)
        
        def func(deltaShape):
            deltaShape = deltaShape.reshape(XIni.shape)
            X = deltaShape + XIni
            dXdz = dXdzIni + np.sum(self.deriv*deltaShape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
            # TODO: also returns the derivative of the action with respect to deltaShape to help minimize converge.
            return self.action(X, dXdz, vevLowT, vevHighT, Tprofile, offEquilDeltas['00'])
        
        i = 0
        # TODO: Implement a better condition and update Tprofile in the loop with the general shape
        shape = np.zeros(self.grid.xiValues.size*self.nbrFields)
        while i < 1:
            sol = minimize(func, shape, method='BFGS')
            shape = sol.x
            i += 1
        
        shape = shape.reshape(XIni.shape)
        X = XIni + shape
        dXdz = dXdzIni + np.sum(self.deriv*shape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
        # TODO: Change X.T to X when freeEnergy gets the right ordering.
        dVdX = self.freeEnergy.derivField(X.T, Tprofile).T
        pressure = -GCLQuadrature(np.concatenate(([0], self.grid.L_xi*(dVdX*dXdz)[0]/(1-self.grid.chiValues**2), [0])))
        
        if returnOptimalWallParams:
            return pressure,shape,wallParams
        else:
            return pressure
    
    def action(self, X, dXdz, vevLowT, vevHighT, Tprofile, offEquilDelta00):
        r"""
        Computes the action by using gaussian quadratrure to integrate the Lagrangian. 

        Parameters
        ----------
        X : array-like
            Wall shape.
        dXdz : array-like
            Derivative of the wall shape.
        vevHighT : array-like
            Field values in the high-T phase.
        Tprofile : array-like
            Temperature on the grid.
        offEquilDelta00 : array-like
            Off-equilibrium function Delta00.

        """
        
        # TODO: Change X.T to X when freeEnergy gets the right ordering.
        V = self.freeEnergy(X.T, Tprofile)
        VOut = self.particle.msqVacuum(X)*offEquilDelta00
        
        VLowT,VHighT = self.freeEnergy(vevLowT,Tprofile[0]),self.freeEnergy(vevHighT,Tprofile[-1])
        Vref = VLowT + 0.5*(VHighT-VLowT)*(1+np.tanh(self.grid.xiValues/self.grid.L_xi))
        
        K = 0.5*np.sum(dXdz**2, axis=0)
        
        S = GCLQuadrature(np.concatenate(([0], self.grid.L_xi*(K+V+VOut-Vref)/(1-self.grid.chiValues**2), [0])))
        return S
    
    def wallProfile(self, deltaShape, vevLowT, vevHighT):
        z = self.grid.xiValues
        z_L = z[:,None]/self.wallWidthsIni[None,:]
        wallOffsetsCompleted = np.append([0], self.wallOffsetsIni)
        
        deltaShape = deltaShape.reshape(z_L.T.shape)
        X = np.transpose(vevLowT + 0.5*(vevHighT-vevLowT)*(1+np.tanh(z_L+wallOffsetsCompleted)))+deltaShape
        dXdz = np.transpose(0.5*(vevHighT-vevLowT)/(self.wallWidthsIni*np.cosh(z_L+wallOffsetsCompleted)**2))
        dXdz += np.sum(self.polynomial.deriv('Cardinal', 'z', False)[None,:,:]*deltaShape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+z**2)**1.5

        return X, dXdz