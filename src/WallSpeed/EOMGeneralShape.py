import numpy as np
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
        self.vwIni, wallParamsIni = self.eom.findWallVelocityMinimizeAction()
        self.wallWidthsIni = wallParamsIni[:nbrFields]
        self.wallOffsetsIni = wallParamsIni[nbrFields:]
        
        self.polynomial = Polynomial(grid)
        
    def findWallVelocity(self):
        pass
    
    def action(self, deltaShape, vevLowT, vevHighT, Tprofile, offEquilDelta00):
        r"""
        Computes the action by using gaussian quadratrure to integrate the Lagrangian. 

        Parameters
        ----------
        deltaShape : array-like
            Deviation from the tanh ansatz of the scalar field profiles.
        vevLowT : array-like
            Field values in the low-T phase.
        vevHighT : array-like
            Field values in the high-T phase.
        Tprofile : array-like
            Temperature on the grid.
        offEquilDeltas : array-like
            Off-equilibrium function Delta00.

        """
        
        X,dXdz = self.wallProfile(deltaShape, vevLowT, vevHighT)
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