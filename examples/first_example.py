"""
A first example.
"""
import numpy as np
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver

class BoltzmannBackground():
    def __init__(self, M):
        self.vw = 1 / np.sqrt(3)
        self.velocityProfile = - np.ones(M - 1) / np.sqrt(3)
        self.fieldProfile = np.ones(M - 1)
        self.fieldProfile[M // 2:]  = 0
        self.temperatureProfile = 100 * np.ones(M - 1)
        self.polynomialBasis = "Cardinal"

class BoltzmannParticle():
    def __init__(self):
        self.msqVacuum = lambda x: 0.5 * x**2
        self.msqThermal = lambda T: 0.1 * T**2
        self.statistics = -1
        self.isOutOfEquilibrium = True
        gsq = 0.4
        self.collisionPrefactors = [gsq**2, gsq**2, gsq**2]

#self.polynomialBasis = "Chebyshev"
#self.M = M
#self.N = N

M = 20
N = 20
grid = Grid(M, N, 1, 1)
poly = Polynomial(grid)
background = BoltzmannBackground(M)
mode = BoltzmannParticle()
boltzmann = BoltzmannSolver(grid, background, mode)
print("BoltzmannSolver object =", boltzmann)
operator, source = boltzmann.buildLinearEquations()
print("operator.shape =", operator.shape)
print("source.shape =", source.shape)

deltaF = boltzmann.solveBoltzmannEquations()
print("deltaF.shape =", deltaF.shape)
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])

Deltas = boltzmann.getDeltas(deltaF)
print("Deltas =", Deltas)
