"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
import matplotlib.pyplot as plt
from scipy import integrate
from WallSpeed.Boltzmann import BoltzmannBackground, BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
from WallSpeed.Polynomial2 import Polynomial2
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy, Grid, Polynomial

"""
Grid
"""
M = 20
N = 20
T = 100
L = 5/T
grid = Grid(M, N, L, T)
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
boltzmann = BoltzmannSolver(grid, background, particle)
print("BoltzmannSolver object =", boltzmann)
operator, source = boltzmann.buildLinearEquations()
print("operator.shape =", operator.shape)
print("source.shape =", source.shape)

deltaF = boltzmann.solveBoltzmannEquations()
print("deltaF.shape =", deltaF.shape)
# print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])

# Integrate to compute Delta using the integrate method of Polynomial2
deltaFPoly = Polynomial2(deltaF, grid, ('Cardinal','Chebyshev','Chebyshev'), ('z','pz','pp'), False)
E = np.sqrt(particle.msqVacuum(field)[:,None,None]+grid.pzValues[None,:,None]**2+grid.ppValues[None,None,:]**2)
dpzdrz = 2*Tmid/(1-grid.rzValues**2)[None,:,None]
dppdrp = Tmid/(1-grid.rpValues)[None,None,:]
print(deltaFPoly.basis)
Delta00Poly = deltaFPoly.integrate((1,2), dpzdrz*dppdrp*grid.ppValues[None,None,:]/(4*np.pi**2*E))
print(Delta00Poly.coefficients.shape,deltaFPoly.basis)

print((dpzdrz*dppdrp*grid.ppValues[None,None,:]/(4*np.pi**2*E)).shape,deltaFPoly.coefficients.shape)
measure = (dpzdrz*dppdrp*grid.ppValues[None,None,:]/(4*np.pi**2*E))
integrandPoly = deltaFPoly*measure
func = lambda rz,rp,chi: integrandPoly.evaluate([chi,rz,rp])
Delta00_dblquad = [integrate.dblquad(func, -1, 1, -1, 1, args=(chi,))[0] for chi in grid.chiValues]

Deltas = boltzmann.getDeltas(deltaF)
# print("Deltas =", Deltas)

# plotting
chi = boltzmann.grid.getCompactCoordinates()[0]
fig, axs = plt.subplots(4)
axs[0].plot(chi, Deltas["00"], label=r"$\Delta_{00}$")
axs[0].set_xlabel(r"$\chi$")
axs[0].set_ylabel(r"$\Delta_{00}\ \mathrm{(Boltzmann)}$")
axs[0].grid()
axs[1].plot(chi, Delta00Poly.coefficients, label=r"$\Delta_{00}\ \mathrm{(Poly)}$")
axs[1].set_xlabel(r"$\chi$")
axs[1].set_ylabel(r"$\Delta_{00}\ \mathrm{(Poly)}$")
axs[1].grid()
axs[2].plot(chi, Delta00_dblquad, label=r"$\Delta_{00}\ \mathrm{(dblquad)}$")
axs[2].set_xlabel(r"$\chi$")
axs[2].set_ylabel(r"$\Delta_{00}\ \mathrm{(dblquad)}$")
axs[2].grid()
axs[3].plot(chi, Deltas["02"], label=r"$\Delta_{02}$")
axs[3].plot(chi, Deltas["20"], label=r"$\Delta_{20}$")
axs[3].plot(chi, Deltas["11"], label=r"$\Delta_{11}$")
axs[3].set_xlabel(r"$\chi$")
axs[3].set_ylabel(r"$\Delta\mathrm{s}$")
axs[3].legend(loc="upper left")
plt.tight_layout()
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
