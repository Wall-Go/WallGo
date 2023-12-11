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
from WallSpeed.HydroTemplateModel import HydroTemplateModel
from WallSpeed import Particle, FreeEnergy, Model
from WallSpeed.EOM import EOM
from WallSpeed.EOMGeneralShape import EOMGeneralShape
import WallSpeed

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
mod = WallSpeed.Model(125, 120, 1.0, 0.9, use_EFT=False)
params = mod.params

Tc = None
Tn = 100

fxSM = WallSpeed.FreeEnergy(mod.Vtot, Tc, Tn, params=params)
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
top = WallSpeed.Particle(
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

hydrotemplate = HydroTemplateModel(thermo)
_,_,_,TminGuess = hydrotemplate.findMatching(0.01)
_,_,TmaxGuess,_=hydrotemplate.findMatching(0.99*hydrotemplate.vJ)
print(f"{TminGuess=}, {TmaxGuess=}")

# print("Test the minimum temperature")
# print(hydro.findHydroBoundaries(0.01))


# print(hydrotemplate.findMatching(0.01))

# print("Find the maximum temperature")
# print(hydro.findHydroBoundaries(0.1))
# print(hydro.findHydroBoundaries(0.3))
print(hydro.findHydroBoundaries(0.4))
print(hydro.findHydroBoundaries(1.01*hydro.vJ))
print(hydro.findHydroBoundaries(0.6))
print(hydro.findHydroBoundaries(0.8))
# print(hydro.findHydroBoundaries(0.99*hydro.vJ))

# print(hydrotemplate.findMatching(0.99*hydrotemplate.vJ))

#Without out-of-equilibrium contributions
# eom = EOM(offEqParticles[0], fxSM, grid, 2)
# print(eom.findWallVelocityMinimizeAction())
eomGeneral = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2)

#With out-of-equilibrium contributions
eomOffEq = EOM(offEqParticles[0], fxSM, grid, 2, True)
eomGeneralOffEq = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2, True)
