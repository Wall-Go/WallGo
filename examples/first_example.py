"""
A first example.
"""
import numpy as np
from WallSpeed.Grid import Grid
from WallSpeed.Boltzmann import BoltzmannSolver

class Background():
    def __init__(self, M, N):
        self.vw = 1 / np.sqrt(3)
        self.velocityProfile = - np.ones(M) / np.sqrt(3)
        self.fieldProfile = np.ones(M)
        self.fieldProfile[:M // 2]
        self.temperatureProfile = 100 * np.ones(M)

class Mode():
    def __init__(self)    :
        self.msq = lambda x: 0.5 * x**2
        self.statistics = -1

M = 10
N = 10
grid = Grid(M, N, 1, 1)
background = Background(M, N)
mode = None
boltzmann = BoltzmannSolver(grid, background, mode)
M, b = boltzmann.buildLinearEquations()

print(boltzmann)
