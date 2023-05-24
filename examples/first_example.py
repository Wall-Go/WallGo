"""
A first example.
"""
import numpy as np
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver

class Background():
    def __init__(self, M, N):
        self.vw = 1 / np.sqrt(3)
        self.velocityProfile = - np.ones(M - 1) / np.sqrt(3)
        self.fieldProfile = np.ones(M - 1)
        self.fieldProfile[M // 2:]  = 0
        self.temperatureProfile = 100 * np.ones(M - 1)

class Mode():
    def __init__(self)    :
        self.msq = lambda x: 0.5 * x**2
        self.statistics = -1

M = 10
N = 10
grid = Grid(M, N, 1, 1)
poly = Polynomial(grid)
background = Background(M, N)
mode = Mode()
boltzmann = BoltzmannSolver(grid, background, mode)
print("BoltzmannSolver object =", boltzmann)
operator, source = boltzmann.buildLinearEquations()
print("operator.shape =", operator.shape)
print("source.shape =", source.shape)

deltaF = boltzmann.solveBoltzmannEquations()
print("deltaF.shape =", deltaF.shape)
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])
