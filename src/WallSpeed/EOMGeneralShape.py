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
        
        # def func(deltaShape):
        #     deltaShape = np.concatenate((deltaShape[:int(self.grid.xiValues.size/2)],[0],deltaShape[int(self.grid.xiValues.size/2):])).reshape(XIni.shape)
        #     X = deltaShape + XIni
        #     dXdz = dXdzIni + np.sum(self.deriv*deltaShape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
        #     # TODO: also returns the derivative of the action with respect to deltaShape to help minimize converge.
        #     return self.action(X, dXdz, vevLowT, vevHighT, Tprofile, offEquilDeltas['00'])
        
        trunc = int(XIni.shape[1]) # Adjust this number to filter out high-frequency modes. This can be useful when aliasing becomes a problem.
        def chebToCard(spectralCoeff):
            a = np.zeros_like(XIni)
            a[:,:trunc] = np.concatenate(([0], spectralCoeff)).reshape((a.shape[0],trunc))
            n = np.arange(a.shape[1])+2
            meanPos = np.sum(np.where(n%2==0, -a[0]*2*n**2/(n**2-1), 0))
            a[0,0] = 3*meanPos/8
            cardinalCoeff = (self.polynomial.chebyshevMatrix('z')@a.T).T
            return cardinalCoeff
        
        def action(spectralCoeff):
            deltaShape = chebToCard(spectralCoeff)
            X = deltaShape + XIni
            dXdz = dXdzIni + np.sum(self.deriv*deltaShape[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
            # TODO: also returns the derivative of the action with respect to deltaShape to help minimize converge.
            return self.action(X, dXdz, vevLowT, vevHighT, Tprofile, offEquilDeltas['00'])
        
        i = 0
        # TODO: Implement a better condition and update Tprofile in the loop with the general shape
        spectral = np.zeros(trunc*self.nbrFields-1)
        success = False
        while not success and i < 10:
            sol = minimize(action, spectral, method='Nelder-Mead')
            success = sol.success
            spectral = sol.x
            i += 1
        
        cardinal = chebToCard(spectral)
        X = XIni + cardinal
        dXdz = dXdzIni + np.sum(self.deriv*cardinal[:,None,:], axis=-1)*self.grid.L_xi**2/(self.grid.L_xi**2+self.grid.xiValues**2)**1.5
        
        # zs = np.linspace(-0.6,0.6,100)
        # chi = zs / np.sqrt(self.grid.L_xi**2 + zs**2)
        # a = np.linalg.inv(self.polynomial.chebyshevMatrix('z'))@cardinal.T
        # plt.plot(zs, self.polynomial.evaluateCardinal(chi.reshape((100,1)), np.concatenate(([0],shape[1],[0])), ('z',)))
        # plt.plot(zs, self.polynomial.evaluateChebyshev(chi.reshape((100,1)), a, ('z',)))
        # plt.plot(self.grid.xiValues,shape[1])
        # plt.grid()
        # plt.show()
        
        # a[-int(a.shape[0]/3):] = 0
        # plt.plot(np.abs(a))
        # plt.yscale('log')
        # plt.show()
        # plt.plot(self.grid.xiValues, cardinal.T)
        # plt.grid()
        # plt.show()
        # plt.plot(self.grid.xiValues, X.T)
        # plt.plot(self.grid.xiValues, XIni.T)
        # plt.grid()
        # plt.show()
        
        
        # plt.plot(self.grid.xiValues,shape.T)
        # plt.grid()
        # plt.show()
        # plt.plot(self.grid.xiValues, X.T, self.grid.xiValues, XIni.T)
        # plt.grid()
        # plt.show()
        # plt.plot(self.grid.xiValues, dXdz.T, self.grid.xiValues, dXdzIni.T)
        # plt.grid()
        # plt.show()
        # TODO: Change X.T to X when freeEnergy gets the right ordering.
        dVdX = self.freeEnergy.derivField(X.T, Tprofile).T
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