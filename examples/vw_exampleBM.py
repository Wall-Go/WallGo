"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Hydro import Hydro
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy, Model
from WallSpeed.EOM import EOM

"""
Grid
"""
M = 20
N = 20
grid = Grid(M, N, 0.05, 100)
poly = Polynomial(grid)


print("--------------")
print("xSM model")
print("Testing the hydrodynamics against Benoit's earlier results")
mod = Model(125,120,1.0,0.9)
params = mod.params
pprint(params)

Tc = 108.22
Tn = 100
print(f"{Tc=}, {Tn=}")

fxSM = FreeEnergy(mod.Vtot, Tc, Tn, params=params)
print("\nFree energy:", fxSM)
print(f"{fxSM([0, 1], 100)=}")
print(f"{fxSM.derivT([0, 1], 100)=}")
print(f"{fxSM.derivField([0, 1], 100)=}")


# defining particles which are out of equilibrium for WallGo
top = Particle(
    "top",
    msqVacuum=lambda X: params["yt"]**2 * np.asanyarray(X)[0]**2,
    msqThermal=lambda T: params["yt"]**2 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=False,
    collisionPrefactors=[params["g2"]**4, params["g2"]**4, params["g2"]**4],
)
particles = [top]
print("\ntop quark:", top)


"""
Compute the wall velocity in local thermal equilibrium
"""
fxSM.interpolateMinima(0,1.2*fxSM.Tc,1)
thermo = Thermodynamics(fxSM)
hydro = Hydro(thermo)
vwLTE = hydro.findvwLTE()
print("The wall velocity in local thermal equilibrium is")
print(vwLTE)

"""
Compute the wall velocity with out-of-equilibrium effects
"""
eom = EOM(top, fxSM, grid, 2)
#print(eom.findWallVelocityLoop())
print(eom.findWallVelocityMinimizeAction())

# now compute the bubble wall speed
# findWallVelocityLoop
