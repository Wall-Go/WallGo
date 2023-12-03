"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
import matplotlib.pyplot as plt
from scipy import integrate
from WallSpeed.Boltzmann2 import BoltzmannBackground, BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Polynomial2 import Polynomial2
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy, Grid, Polynomial

"""
Grid
"""
M = 40
N = 19
T = 100
L = 5/T
grid = Grid(M, N, 0.35, T)
poly = Polynomial(grid)

"""
Background
"""
vw = 0
v = - np.ones(M - 1) / np.sqrt(3)
vev = 90
field = np.array([vev*(1-np.tanh(grid.xiValues/L))/2, vev*(1+np.tanh(grid.xiValues/L))/2])
T = 100 * np.ones(M - 1)
Tmid = (T[0]+T[-1])/2
basis = "Cardinal"
velocityMid = 0.5 * (v[0] + v[-1])

background = BoltzmannBackground(
    velocityMid=velocityMid,
    velocityProfile=np.concatenate(([v[0]],v,[v[-1]])),
    fieldProfile=np.concatenate((field[:,0,None],field,field[:,-1,None]),1),
    temperatureProfile=np.concatenate(([T[0]],T,[T[-1]])),
    polynomialBasis=basis,
)

#test boost
# background.vw=0
# print(background.velocityProfile)
# background.boostToPlasmaFrame()
# print(background.velocityProfile)
# background.boostToWallFrame()
# print(background.velocityProfile)

"""
Particle
"""
particle = Particle(
    name="top",
    msqVacuum=lambda phi: 0.5 * phi[0]**2,
    msqThermal=lambda T: 0.1 * T**2,
    statistics="Fermion",
    inEquilibrium=False,
    ultrarelativistic=False,
    collisionPrefactors=[1, 1, 1],
)

"""
Boltzmann solver
"""
boltzmannCheb = BoltzmannSolver(grid, background, particle, 'Cardinal', 'Chebyshev')
boltzmannCard = BoltzmannSolver(grid, background, particle, 'Cardinal', 'Cardinal')

DeltasCheb = boltzmannCheb.getDeltas()
DeltasCard = boltzmannCard.getDeltas()

# plotting
chi = grid.getCompactCoordinates()[0]
xi = np.linspace(-200*L,200*L,1000)
chi2 = xi/np.sqrt(xi**2+grid.L_xi**2)

plt.plot(xi, 12*DeltasCheb['00'].evaluate(chi2[None,:]))
plt.plot(xi, 12*DeltasCard['00'].evaluate(chi2[None,:]))
plt.legend(('Chebyshev basis','Cardinal basis'))
plt.xlabel(r"$\xi$")
plt.ylabel(r"$\Delta_{00}\ \mathrm{(Poly)}$")
plt.xlim((-20*L,20*L))
plt.grid()
plt.show()

# now making a deltaF by hand
# deltaF = np.zeros(deltaF.shape)
# # coordinates
# chi, rz, rp = grid.getCompactCoordinates() # compact
# xi, pz, pp = grid.getCoordinates() # non-compact
# xi = xi[:, np.newaxis, np.newaxis]
# pz = pz[np.newaxis, :, np.newaxis]
# pp = pp[np.newaxis, np.newaxis, :]

# # background
# vFixed = background.velocityProfile[0]
# T0 = background.temperatureProfile[0]

# # fluctuation mode
# msq = particle.msqVacuum(background.fieldProfile)
# msq = msq[:, np.newaxis, np.newaxis]
# E = np.sqrt(msq + pz**2 + pp**2)

# # integrand with known result
# integrand_analytic = (
#     E
#     * np.sqrt(1 - rz[np.newaxis, :, np.newaxis]**2)
#     * np.sqrt(1 - rp[np.newaxis, np.newaxis, :])
# )

# # test with integrand_analytic
# Deltas = boltzmann.getDeltas(integrand_analytic)
# Deltas_analytic = 2 * np.sqrt(2) * background.temperatureProfile**3 / np.pi
# print("Ratio = 1 =", Deltas["00"] / Deltas_analytic)
# print("T =", background.temperatureProfile)
