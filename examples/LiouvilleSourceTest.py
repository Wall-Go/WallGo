"""
A first example.
"""
import numpy as np # arrays, maths and stuff
from pprint import pprint # pretty printing of dicts
import matplotlib.pyplot as plt
from WallSpeed.Boltzmann import BoltzmannBackground, BoltzmannSolver
from WallSpeed.Thermodynamics import Thermodynamics
#from WallSpeed.eomHydro import findWallVelocityLoop
from WallSpeed import Particle, FreeEnergy, Grid, Polynomial
import warnings


"""
Test objects
"""
def __feq(x, statistics):
    if np.isclose(statistics, 1, atol=1e-14):
        return 1 / np.expm1(x)
    else:
        return 1 / (np.exp(x) + 1)
        
def __dfeq(x, statistics):
    x = np.asarray(x)
    if np.isclose(statistics, 1, atol=1e-14):
        return np.where(x > 100, -np.exp(-x), -np.exp(x) / np.expm1(x) ** 2)
    else:
        return np.where(
            x > 100, -np.exp(-x), -1 / (np.exp(x) + 2 + np.exp(-x))
        )

def fromCardinalToChebyshev(matrix, polynomial):
    zChange = np.transpose(polynomial.chebyshevMatrix('z'))
    pzChange = np.transpose(polynomial.chebyshevMatrix('pz'))
    ppChange = np.transpose(polynomial.chebyshevMatrix('pp'))

    return np.einsum('ia,jb,kc,abc->ijk', zChange, pzChange, ppChange, matrix)
        
def buildLiouvilleOperatorSourceAndEquilibriumDistribution(boltzmannSolver):
    """
    Constructs matrix and source for Boltzmann equation.

    Note, we make extensive use of numpy's broadcasting rules.
    """
    # coordinates
    xi, pz, pp = boltzmannSolver.grid.getCoordinates()  # non-compact
    xi = xi[:, np.newaxis, np.newaxis]
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # intertwiner matrices
    TChiMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisM, "z")
    TRzMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pz")
    TRpMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pp")

    # derivative matrices
    derivChi = boltzmannSolver.poly.deriv(boltzmannSolver.basisM, "z")
    derivRz = boltzmannSolver.poly.deriv(boltzmannSolver.basisN, "pz")

    # background profiles
    T = boltzmannSolver.background.temperatureProfile[:, np.newaxis, np.newaxis]
    field = boltzmannSolver.background.fieldProfile[..., np.newaxis, np.newaxis]
    v = boltzmannSolver.background.velocityProfile[:, np.newaxis, np.newaxis]
    vw = boltzmannSolver.background.vw

    # fluctuation mode
    statistics = -1 if boltzmannSolver.particle.statistics == "Fermion" else 1
    # TODO: indices order not consistent across different functions.
    msq = boltzmannSolver.particle.msqVacuum(field)
    E = np.sqrt(msq + pz**2 + pp**2)

    # dot products with wall velocity
    gammaWall = 1 / np.sqrt(1 - vw**2)
    EWall = gammaWall * (E - vw * pz)
    PWall = gammaWall * (pz - vw * E)

    # dot products with plasma profile velocity
    gammaPlasma = 1 / np.sqrt(1 - v**2)
    EPlasma = gammaPlasma * (E - v * pz)
    PPlasma = gammaPlasma * (pz - v * E)

    # dot product of velocities
    uwBaruPl = gammaWall * gammaPlasma * (vw - v)

    # spatial derivatives of profiles
    dTdChi = np.einsum("ij,jbc->ibc", derivChi, T, optimize=True)
    dvdChi = np.einsum("ij,jbc->ibc", derivChi, v, optimize=True)
    dmsqdChi = np.einsum("ij,jbc->ibc", derivChi, msq, optimize=True)

    # derivatives of compactified coordinates
    dchidxi, drzdpz, drpdpp = boltzmannSolver.grid.getCompactificationDerivatives()
    dchidxi = dchidxi[:, np.newaxis, np.newaxis]
    drzdpz = drzdpz[np.newaxis, :, np.newaxis]

    # equilibrium distribution, and its derivative
    warnings.filterwarnings("ignore", message="overflow encountered in exp")
    fEq = __feq(EPlasma / T, statistics)
    dfEq = __dfeq(EPlasma / T, statistics)
    warnings.filterwarnings(
        "default", message="overflow encountered in exp"
    )

    ##### source term #####
    source = (dfEq / T) * dchidxi * (
        PWall * PPlasma * gammaPlasma**2 * dvdChi
        + PWall * EPlasma * dTdChi / T
        + 1 / 2 * dmsqdChi * uwBaruPl
    )

    ##### liouville operator #####
    liouville = (
        dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * PWall[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * derivChi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
            * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        - dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * drzdpz[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * gammaWall / 2
            * dmsqdChi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * derivRz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
            * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    )

    """
    No reshaping!

    # doing matrix-like multiplication
    N_new = (boltzmannSolver.grid.M - 1) * (boltzmannSolver.grid.N - 1) * (boltzmannSolver.grid.N - 1)

    # reshaping indices
    N_new = (boltzmannSolver.grid.M - 1) * (boltzmannSolver.grid.N - 1) * (boltzmannSolver.grid.N - 1)
    source = np.reshape(source, N_new, order="C")
    operator = np.reshape(operator, (N_new, N_new), order="C")
    """

    # returning results
    return liouville, source, fEq


"""
Grid
"""
M = 10
N = 10
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
basis = "Cardinal"
velocityMid = 0.5 * (v[0] + v[-1])

background = BoltzmannBackground(
    velocityMid=velocityMid,
    velocityProfile=v,
    fieldProfile=field,
    temperatureProfile=T,
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
liouvilleOperator, source, eqDistribution = buildLiouvilleOperatorSourceAndEquilibriumDistribution(boltzmann)
print("liouvilleOperator.shape =", liouvilleOperator.shape)
print("source.shape =", source.shape)
print("eqDistribution.shape =", eqDistribution.shape)

eqDistribution = fromCardinalToChebyshev(eqDistribution, poly)

shouldBeSource = -np.einsum("ijkabc,abc->ijk", liouvilleOperator, eqDistribution)

difference = np.abs(source - shouldBeSource) / source

print(difference)

exit()

deltaF = boltzmann.solveBoltzmannEquations()
print("deltaF.shape =", deltaF.shape)
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])

Deltas = boltzmann.getDeltas(deltaF)
print("Deltas =", Deltas)

# plotting
chi = boltzmann.grid.getCompactCoordinates()[0]
fig, axs = plt.subplots(2)
axs[0].plot(chi, Deltas["00"], label=r"$\Delta_{00}$")
axs[0].set_xlabel(r"$\chi$")
axs[0].set_ylabel(r"$\Delta_{00}$")
axs[1].plot(chi, Deltas["02"], label=r"$\Delta_{02}$")
axs[1].plot(chi, Deltas["20"], label=r"$\Delta_{20}$")
axs[1].plot(chi, Deltas["11"], label=r"$\Delta_{11}$")
axs[1].set_xlabel(r"$\chi$")
axs[1].set_ylabel(r"$\Delta\mathrm{s}$")
axs[1].legend(loc="upper left")
plt.tight_layout()
plt.show()

# now making a deltaF by hand
deltaF = np.zeros(deltaF.shape)
# coordinates
chi, rz, rp = grid.getCompactCoordinates() # compact
xi, pz, pp = grid.getCoordinates() # non-compact
xi = xi[:, np.newaxis, np.newaxis]
pz = pz[np.newaxis, :, np.newaxis]
pp = pp[np.newaxis, np.newaxis, :]

# background
vFixed = background.velocityProfile[0]
T0 = background.temperatureProfile[0]

# fluctuation mode
msq = particle.msqVacuum(background.fieldProfile)
msq = msq[:, np.newaxis, np.newaxis]
E = np.sqrt(msq + pz**2 + pp**2)

# integrand with known result
integrand_analytic = (
    E
    * np.sqrt(1 - rz[np.newaxis, :, np.newaxis]**2)
    * np.sqrt(1 - rp[np.newaxis, np.newaxis, :])
)

# test with integrand_analytic
Deltas = boltzmann.getDeltas(integrand_analytic)
Deltas_analytic = 2 * np.sqrt(2) * background.temperatureProfile**3 / np.pi
print("Ratio = 1 =", Deltas["00"] / Deltas_analytic)
print("T =", background.temperatureProfile)
