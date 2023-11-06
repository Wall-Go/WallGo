"""
Benchmark model: xSM, as implemented earlier by Benoit
defines model, grid and top
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannBackground, BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Hydro import Hydro
from WallSpeed import Particle, FreeEnergy, GenericModel
from WallSpeed.EOM import EOM
from WallSpeed.EOMGeneralShape import EOMGeneralShape
import WallSpeed
from models.xSM import xSM

"""
Grid
"""
M = 20
N = 20
grid = Grid(M, N, 0.05, 100)
poly = Polynomial(grid)

"""
Model definition
"""

mod = xSM(125, 120, 1.0, 0.9, use_EFT=False)

Tc = None
Tn = 100

fxSM = WallSpeed.FreeEnergy(mod, Tc, Tn)
Tc = fxSM.Tc
print(f"{Tc=}, {Tn=}")
print("\nFree energy:", fxSM)
print(f"{fxSM([[0],[1]], 100)=}")
print(f"{fxSM([0,1], 100)=}")
print(f"{fxSM([[0,0],[1,10]], 100)=}")
print(f"{fxSM.derivT([[0],[1]], 100)=}")
print(f"{fxSM.derivField([[0],[1]], 100)=}")


# defining particles which are out of equilibrium for WallGo
offEqParticles = mod.outOfEquilibriumParticles()

"""
Define thermodynamics, hydrodynamics and equation of motion
"""
fxSM.interpolateMinima(0,1.2*fxSM.Tc,1)

thermo = Thermodynamics(fxSM)
hydro = Hydro(thermo)

#Without out-of-equilibrium contributions
eom = EOM(offEqParticles[0], fxSM, grid, 2)
eomGeneral = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2)

#With out-of-equilibrium contributions
eomOffEq = EOM(offEqParticles[0], fxSM, grid, 2, True)
eomGeneralOffEq = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2, True)
