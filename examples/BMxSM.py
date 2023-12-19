"""
Benchmark model: xSM, as implemented earlier by Benoit
defines model, grid and top
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallGo.Grid import Grid
from WallGo.Polynomial import Polynomial
from WallGo.Boltzmann import BoltzmannBackground, BoltzmannSolver
from WallGo.Thermodynamics import Thermodynamics
from WallGo.Hydro import Hydro
from WallGo import Particle, FreeEnergy, Model
from WallGo.EOM import EOM
from WallGo.EOMGeneralShape import EOMGeneralShape
import WallGo

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
mod = WallGo.Model(125, 120, 1.0, 0.9, use_EFT=False)
params = mod.params

Tc = None
Tn = 100

fxSM = WallGo.FreeEnergy(mod.Vtot, Tc, Tn, params=params)
Tc = fxSM.Tc
pprint(params)
print(f"{Tc=}, {Tn=}")
print("\nFree energy:", fxSM)
print(f"{fxSM([[0],[1]], 100)=}")
print(f"{fxSM([0,1], 100)=}")
print(f"{fxSM([[0,0],[1,10]], 100)=}")
print(f"{fxSM.derivT([[0],[1]], 100)=}")
print(f"{fxSM.derivField([[0],[1]], 100)=}")


# defining particles which are out of equilibrium for WallGo
top = WallGo.Particle(
    "top",
    msqVacuum=lambda X: params["yt"]**2 * np.asanyarray(X)[0,...]**2,
    msqThermal=lambda T: 0.251327 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=False,
    collisionPrefactors=[params["g3"]**4, params["g3"]**4, params["g3"]**4],
)

offEqParticles = [top]

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
