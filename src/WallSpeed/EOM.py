import numpy as np

from scipy.optimize import minimize, minimize_scalar, brentq, root, root_scalar
from scipy.integrate import quad_vec,quad
from scipy.interpolate import UnivariateSpline
#import matplotlib.pyplot as plt
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .model import Particle, FreeEnergy
from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .helpers import derivative, gammasq # derivatives for callable functions

class EOM:
    def __init__(self, particle, freeEnergy, grid, nbrFields, errTol=1e-6):
        self.particle = particle
        self.freeEnergy = freeEnergy
        self.grid = grid
        self.errTol = errTol
        self.N = nbrFields
        self.Tnucl = freeEnergy.Tnucl
        
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
        
        wallWidthsGuess = (5/self.Tnucl)*np.ones(self.N)
        wallOffsetsGuess = np.zeros(self.N-1)
        wallWidths, wallOffsets = wallWidthsGuess, wallOffsetsGuess
        # higgsWidth, singletWidth, wallOffSet = initialWallParameters(
        #     higgsWidthGuess,
        #     singletWidthGuess,
        #     wallOffSetGuess,
        #     0.5 * (Tplus + Tminus),
        #     freeEnergy,
        # )

        wallParameters = np.concatenate(([wallVelocity], wallWidths, wallOffsets))
        
        wallParameters = np.array([0.60665297, 0.05, 0.04, 0.2])

        print(wallParameters)

        offEquilDeltas = {"00": np.zeros(self.grid.M-1), "02": np.zeros(self.grid.M-1), "20": np.zeros(self.grid.M-1), "11": np.zeros(self.grid.M-1)}
        
        # print(self.momentsOfWallEoM(np.array([0.5,0.05,0.05,0]), offEquilDeltas))#[-4818415.398740615, 443797.80971509626, 1973501.8795333474, -1058439.7855602442]
        # raise
        
        error = self.errTol + 1
        while error > self.errTol:

            oldWallVelocity = wallParameters[0]
            oldWallWidths = wallParameters[1:1+self.N]
            oldWallOffsets = wallParameters[1+self.N:]
            oldError = error

            c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)


            # wallProfileGrid = wallProfileOnGrid(wallParameters[1:], Tplus, Tminus, grid,freeEnergy)
            
            Tprofile, velocityProfile = self.findPlasmaProfile(c1, c2, velocityAtz0, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)

            # boltzmannBackground = BoltzmannBackground(wallParameters[0], velocityProfile, wallProfileGrid, Tprofile)

            # boltzmannSolver = BoltzmannSolver(grid, boltzmannBackground, particle)
            
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
    
    def momentsOfWallEoM(self, wallParameters, offEquilDeltas):
        wallVelocity = wallParameters[0]
        wallWidths = wallParameters[1:self.N+1]
        wallOffsets = wallParameters[self.N+1:]
        c1, c2, Tplus, Tminus, velocityAtz0 = self.hydro.findHydroBoundaries(wallVelocity)
        Tprofile, vprofile = self.findPlasmaProfile(c1, c2, velocityAtz0, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus)

        vevLowT = self.freeEnergy.findPhases(Tminus)[0]
        vevHighT = self.freeEnergy.findPhases(Tplus)[1]
        
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
        if np.isscalar(z):
            z_L = z/wallWidths
        else:
            z_L = z[:,None]/wallWidths[None,:]
        wallOffsetsCompleted = np.append([0], wallOffsets)
        
        X = np.transpose(vevLowT + 0.5*(vevHighT-vevLowT)*(1+np.tanh(z_L+wallOffsetsCompleted)))
        dXdz = np.transpose(0.5*(vevHighT-vevLowT)/(wallWidths*np.cosh(z_L+wallOffsetsCompleted)**2))
        
        return X, dXdz
    
    def findPlasmaProfile(self, c1, c2, velocityAtz0, wallWidths, wallOffsets, offEquilDeltas, Tplus, Tminus):
        """
        Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of LHS.
        """
        temperatureProfile = []
        velocityProfile = []
        
        vevLowT = self.freeEnergy.findPhases(Tminus)[0]
        vevHighT = self.freeEnergy.findPhases(Tplus)[1]
        X,dXdz = self.wallProfile(self.grid.xiValues, vevLowT, vevHighT, wallWidths, wallOffsets)
        
        for index in range(len(self.grid.xiValues)):
            T, vPlasma = self.findPlasmaProfilePoint(index, c1, c2, velocityAtz0, X[:,index], dXdz[:,index], offEquilDeltas, Tplus, Tminus) 

            temperatureProfile.append(T)
            velocityProfile.append(vPlasma)

        return np.array(temperatureProfile), np.array(velocityProfile)
    
    def findPlasmaProfilePoint(self, index, c1, c2, velocityAtz0, X, dXdz, offEquilDeltas, Tplus, Tminus):
        """
        Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.
        """
        
        Tout30, Tout33 = self.deltaToTmunu(index,X,velocityAtz0,Tminus,offEquilDeltas)

        s1 = c1 - Tout30 
        s2 = c2 - Tout33
        
        minRes = minimize_scalar(lambda T: self.temperatureProfileEqLHS(X, dXdz, T, s1, s2), method='Bounded', bounds=[0,self.freeEnergy.Tc], tol=1e-9)
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
        
        res = brentq(lambda T: self.temperatureProfileEqLHS(X, dXdz, T, s1, s2), TLowerBound, TUpperBound, xtol=1e-9, rtol=1e-9)
        # TODO: Can the function have multiple zeros?

        T = res   #is this okay?
        vPlasma = self.plasmaVelocity(X, T, s1)
        return T, vPlasma
    
    def plasmaVelocity(self, X, T, s1):
        dVdT = self.freeEnergy.derivT(X, T)
        return (T * dVdT  + np.sqrt(4 * s1**2 + (T * dVdT)**2)) / (2 * s1)
    
    def temperatureProfileEqLHS(self, X, dXdz, T, s1, s2):
        """
        The LHS of Eq. (20) of arXiv:2204.13120v1
        """
        return 0.5*np.sum(dXdz**2, axis=0) - self.freeEnergy(X, T) + 0.5*T*self.freeEnergy.derivT(X, T) + 0.5*np.sqrt(4*s1**2 + (T*self.freeEnergy.derivT(X, T))**2) - s2
    
    def deltaToTmunu(self, index, X, velocityAtCenter, Tm, offEquilDeltas):
        delta00 = offEquilDeltas["00"][index]
        delta11 = offEquilDeltas["11"][index]
        delta02 = offEquilDeltas["02"][index]
        delta20 = offEquilDeltas["20"][index]

        u0 = np.sqrt(gammasq(velocityAtCenter))
        u3 = np.sqrt(gammasq(velocityAtCenter))*velocityAtCenter
        ubar0 = u3
        ubar3 = u0


        T30 = ((3*delta20 - delta02 - self.particle.msqVacuum(X)*delta00)*u3*u0+
                (3*delta02 - delta20 + self.particle.msqVacuum(X)*delta00)*ubar3*ubar0+2*delta11*(u3*ubar0 + ubar3*u0))/2.
        T33 = ((3*delta20 - delta02 - self.particle.msqVacuum(X)*delta00)*u3*u3+
                (3*delta02 - delta20 + self.particle.msqVacuum(X)*delta00)*ubar3*ubar3+4*delta11*u3*ubar3)/2.

        return T30, T33
    
    
    
    
    
    
    
        


# def initialEOMSolution(wallParametersIni, offEquilDeltas, freeEnergy, hydro, particle, grid):
#     """
#     Solves Gs=0, Gh=0, Ph-Ps=0 and Ph+Ps=0 one at a time for Ls, Lh, delta and vw, respectively.
#     This returns an approximate solution to the moment equations.

#     """
#     wallVelocity, higgsWidth, singletWidth, wallOffSet = wallParametersIni
#     c1, c2, Tplus, Tminus, velocityAtz0 = hydro.findHydroBoundaries(wallVelocity)
#     Tprofile, vprofile = findPlasmaProfile(c1, c2, velocityAtz0, higgsWidth, singletWidth, wallOffSet, offEquilDeltas, particle, Tplus, Tminus, freeEnergy, grid)
    
#     higgsVEV = freeEnergy.findPhases(Tminus)[0,0]
#     singletVEV = freeEnergy.findPhases(Tplus)[1,1]
    
#     Tfunc = UnivariateSpline(grid.xiValues, Tprofile, k=3, s=0)
#     offEquilDelta00 = UnivariateSpline(grid.xiValues, offEquilDeltas['00'], k=3, s=0)
    
#     # Solving Gs=0 for Ls
#     Ls,Ls1,Ls2 = singletWidth,0.9*singletWidth,1.1*singletWidth
#     Gs = lambda x: singletStretchMoment(higgsVEV, higgsWidth, singletVEV, x, wallOffSet, freeEnergy, offEquilDelta00, Tfunc)
#     Gs1,Gs2 = Gs(Ls1),Gs(Ls2)
#     i = 0
#     while Gs1*Gs2 > 0 and i < 10:
#         i += 1
#         if abs(Gs1) < abs(Gs2):
#             Ls2,Gs2 = Ls1,Gs1
#             Ls1 *= 0.5
#             Gs1 = Gs(Ls1)
#         else:
#             Ls1,Gs1 = Ls2,Gs2
#             Ls2 *= 2
#             Gs2 = Gs(Ls2)
#     if Gs1*Gs2 <= 0:
#         Ls = root_scalar(Gs, bracket=[Ls1,Ls2], method='brentq').root
#     else:
#         Ls = root_scalar(Gs, x0=Ls1, x1=Ls2, method='secant').root
    
#     # Solving Gh=0 for Lh
#     Tprofile, vprofile = findPlasmaProfile(c1, c2, velocityAtz0, higgsWidth, Ls, wallOffSet, offEquilDeltas, particle, Tplus, Tminus, freeEnergy, grid)
#     Tfunc = UnivariateSpline(grid.xiValues, Tprofile, k=3, s=0)
#     Lh,Lh1,Lh2 = higgsWidth,0.9*higgsWidth,1.1*higgsWidth
#     Gh = lambda x: higgsStretchMoment(higgsVEV, x, singletVEV, Ls, wallOffSet, freeEnergy, particle, offEquilDelta00, Tfunc)
#     Gh1,Gh2 = Gh(Lh1),Gh(Lh2)
#     i = 0
#     while Gh1*Gh2 > 0 and i < 10:
#         i += 1
#         if abs(Gh1) < abs(Gh2):
#             Lh2,Gh2 = Lh1,Gh1
#             Lh1 *= 0.5
#             Gh1 = Gh(Lh1)
#         else:
#             Lh1,Gh1 = Lh2,Gh2
#             Lh2 *= 2
#             Gh2 = Gh(Lh2)
#     if Gh1*Gh2 <= 0:
#         Lh = root_scalar(Gh, bracket=[Lh1,Lh2], method='brentq').root
#     else:
#         Lh = root_scalar(Gh, x0=Lh1, x1=Lh2, method='secant').root
        
#     # Solving Ph-Ps=0 for delta
#     Tprofile, vprofile = findPlasmaProfile(c1, c2, velocityAtz0, Lh, Ls, wallOffSet, offEquilDeltas, particle, Tplus, Tminus, freeEnergy, grid)
#     Tfunc = UnivariateSpline(grid.xiValues, Tprofile, k=3, s=0)
#     delta,delta1,delta2 = wallOffSet,wallOffSet-0.1,wallOffSet+0.1
#     Pdiff = lambda x: higgsPressureMoment(higgsVEV, Lh, singletVEV, Ls, x, freeEnergy, particle, offEquilDelta00, Tfunc)-singletPressureMoment(higgsVEV, Lh, singletVEV, Ls, x, freeEnergy, offEquilDelta00, Tfunc)
#     Pdiff1,Pdiff2 = Pdiff(delta1),Pdiff(delta2)
#     i = 0
#     while Pdiff1*Pdiff2 > 0 and i < 10:
#         i += 1
#         if abs(Pdiff1) < abs(Pdiff2):
#             delta2,Pdiff2 = delta1,Pdiff1
#             delta1 -= 0.5
#             Pdiff1 = Pdiff(delta1)
#         else:
#             delta1,Pdiff1 = delta2,Pdiff2
#             delta2 += 0.5
#             Pdiff2 = Pdiff(delta2)
#     if Pdiff1*Pdiff2 <= 0:
#         delta = root_scalar(Pdiff, bracket=[delta1,delta2], method='brentq').root
#     else:
#         delta = root_scalar(Pdiff, x0=delta1, x1=delta2, method='secant').root
    
#     # Solving Ph+Ps=0 for vw
#     def Ptot(x):
#         # TODO: Update offEquilDeltas at each evaluation
#         c1, c2, Tplus, Tminus, velocityAtz0 = hydro.findHydroBoundaries(x)
#         Tprofile, vprofile = findPlasmaProfile(c1, c2, velocityAtz0, Lh, Ls, delta, offEquilDeltas, particle, Tplus, Tminus, freeEnergy, grid)
#         higgsVEV = freeEnergy.findPhases(Tminus)[0,0]
#         singletVEV = freeEnergy.findPhases(Tplus)[1,1]
#         Tfunc = UnivariateSpline(grid.xiValues, Tprofile, k=3, s=0)
#         return higgsPressureMoment(higgsVEV, Lh, singletVEV, Ls, delta, freeEnergy, particle, offEquilDelta00, Tfunc)+singletPressureMoment(higgsVEV, Lh, singletVEV, Ls, delta, freeEnergy, offEquilDelta00, Tfunc)
    
#     vw = hydro.vJ
#     if Ptot(0.01)*Ptot(hydro.vJ) <= 0:
#         vw = root_scalar(Ptot, bracket=[0.01,hydro.vJ-1e-6], method='brentq').root
    
#     return [vw,Lh,Ls,delta]





# def initialWallParameters(
#     higgsWidthGuess,
#     singletWidthGuess,
#     wallOffSetGuess,
#     TGuess,
#     freeEnergy
# ):
#     higgsVEV = freeEnergy.findPhases(TGuess)[0,0]
#     singletVEV = freeEnergy.findPhases(TGuess)[1,1]

#     initRes = minimize(
#         lambda wallParams: oneDimAction(higgsVEV, singletVEV, wallParams, TGuess, freeEnergy),
#         x0=[higgsWidthGuess, singletWidthGuess, wallOffSetGuess],
#         bounds=[(0, None), (0, None), (-10, 10)],
#     )

#     return initRes.x[0], initRes.x[1], initRes.x[2]


# def oneDimAction(higgsVEV, singletVEV, wallParams, T, freeEnergy):
#     [higgsWidth, singletWidth, wallOffSet] = wallParams

#     kinetic = (higgsVEV**2 / higgsWidth + singletVEV**2 / singletWidth) * 3 / 2

#     integrationLength = (20 + np.abs(wallOffSet)) * max(higgsWidth, singletWidth)

#     integral = quad(
#         lambda z: freeEnergy(
#             wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z),
#             T,
#         ),
#         -integrationLength,
#         integrationLength,
#     )

#     potential = integral[0] - integrationLength * (
#         freeEnergy([higgsVEV, 0], T) + freeEnergy([0, singletVEV], T)
#     )

#     # print(higgsWidth, singletWidth, wallOffSet)

#     # print(kinetic + potential)

#     return kinetic + potential


# def wallProfileOnGrid(staticWallParams, Tplus, Tminus, grid,freeEnergy):
#     [higgsWidth, singletWidth, wallOffSet] = staticWallParams

#     higgsVEV = freeEnergy.findPhases(Tminus)[0,0]
#     singletVEV = freeEnergy.findPhases(Tplus)[1,1]

#     wallProfileGrid = []
#     for z in grid.xiValues:
#         wallProfileGrid.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))

#     return np.transpose(wallProfileGrid)


# def wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z):
#     h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
#     s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))

#     return [h, s]


