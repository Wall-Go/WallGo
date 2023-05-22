"""
A first example.
"""
from WallSpeed.Grid import Grid
from WallSpeed.boltzmann import BoltzmannSolver

grid = Grid(10, 10, 1, 1)
background = None
mode = None
collisionFile = "collision_mock.hdf5"
boltzmann = BoltzmannSolver(grid, background, mode, collisionFile)

print(boltzmann)
