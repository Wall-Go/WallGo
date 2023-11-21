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
        
def buildSourceAndShouldBeSource(boltzmannSolver):
    """
    Constructs matrix and source for Boltzmann equation.

    Note, we make extensive use of numpy's broadcasting rules.
    """
    # coordinates
    xi, pz, pp = boltzmannSolver.grid.getCoordinates()  # non-compact
    xi = xi[:, np.newaxis, np.newaxis]
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # compactified coordinates
    chi, rz, rp = boltzmannSolver.grid.getCompactCoordinates() # compact

    # intertwiner matrices
    TChiMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisM, "z")
    TRzMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pz")
    TRpMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pp")

    # derivative matrices
    derivChi = np.delete(boltzmannSolver.poly.deriv(boltzmannSolver.basisM, "z", endpoints=True), [0, -1], axis=0)
    derivRz = boltzmannSolver.poly.deriv(boltzmannSolver.basisN, "pz")

    # background profiles
    T = boltzmannSolver.background.temperatureProfile[1:-1, np.newaxis, np.newaxis]
    field = boltzmannSolver.background.fieldProfile[:, 1:-1, np.newaxis, np.newaxis]
    v = boltzmannSolver.background.velocityProfile[1:-1, np.newaxis, np.newaxis]
    vw = boltzmannSolver.background.vw

    # fluctuation mode
    statistics = -1 if boltzmannSolver.particle.statistics == "Fermion" else 1
    # TODO: indices order not consistent across different functions.
    msq = boltzmannSolver.particle.msqVacuum(field)
    E = np.sqrt(msq + pz**2 + pp**2)

    # fit the background profiles to polynomial
#        print(numpy.polynomial.chebyshev.chebfit(chi, boltzmannSolver.background.temperatureProfile, boltzmannSolver.grid.M))
#        print(np.shape(numpy.polynomial.chebyshev.chebfit(chi, boltzmannSolver.background.temperatureProfile, boltzmannSolver.grid.M)))
    Tpoly = Polynomial2(boltzmannSolver.background.temperatureProfile, boltzmannSolver.grid,  'Cardinal','z', True)
    msqpoly = Polynomial2(boltzmannSolver.particle.msqVacuum(boltzmannSolver.background.fieldProfile) ,boltzmannSolver.grid,  'Cardinal','z', True)
    vpoly = Polynomial2(boltzmannSolver.background.velocityProfile, boltzmannSolver.grid,  'Cardinal','z', True)

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
    dTdChi = Tpoly.derivative(0).coefficients[1:-1, None, None]
    dvdChi = vpoly.derivative(0).coefficients[1:-1, None, None]
    dmsqdChi = msqpoly.derivative(0).coefficients[1:-1, None, None]
    
    # derivatives of compactified coordinates
    dchidxi, drzdpz, drpdpp = boltzmannSolver.grid.getCompactificationDerivatives()
    dchidxi = dchidxi[:, np.newaxis, np.newaxis]
    drzdpz = drzdpz[np.newaxis, :, np.newaxis]

    # equilibrium distribution, and its derivative
    warnings.filterwarnings("ignore", message="overflow encountered in exp")
    fEq2 = __feq(EPlasma / T, statistics)
    #print(fEq2.shape)
    fEq1 = np.insert(fEq2, [0, -1], [fEq2[0], fEq2[-1]], axis=0)
    #print(fEq1[0]-fEq1[1])
    #print(fEq1[-1]-fEq1[-2])

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
    liouville1 = (
        dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * PWall[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * derivChi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
            * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    )
    liouville2 = (
        - dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * drzdpz[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * gammaWall / 2
            * dmsqdChi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * derivRz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
            * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
    )

    sbSource1 = -np.einsum("abcijk,ijk->abc", liouville1, fEq1)
    sbSource2 = -np.einsum("abcijk,ijk->abc", liouville2, fEq2)

    """

    # doing matrix-like multiplication
    N_new = (boltzmannSolver.grid.M - 1) * (boltzmannSolver.grid.N - 1) * (boltzmannSolver.grid.N - 1)

    # reshaping indices
    N_new = (boltzmannSolver.grid.M - 1) * (boltzmannSolver.grid.N - 1) * (boltzmannSolver.grid.N - 1)
    source = np.reshape(source, N_new, order="C")
    liouville = np.reshape(liouville, (N_new, N_new), order="C")
    """

    # returning results
    return source, sbSource1+sbSource2


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
boltzmann = BoltzmannSolver(grid, background, particle, basisN="Cardinal")

"""
Testing the Boltzmann solver
"""
source, sbSource = buildSourceAndShouldBeSource(boltzmann)

plt.plot(boltzmann.grid.xiValues, source[:,10,0], label=r"$S$")
plt.plot(boltzmann.grid.xiValues, sbSource[:,10,0], label=r"$\tilde{S}=-L f_{eq}$")
plt.plot(boltzmann.grid.xiValues, (sbSource-source)[:,10,0], label=r"$-(L f_{eq}+S)$")
plt.xlabel(r"$\xi$")
plt.legend()
plt.show()

plt.plot(source.flatten(), label=r"$S$")
plt.plot(sbSource.flatten(), label=r"$\tilde{S}=-L f_{eq}$", alpha=0.5)
plt.plot((sbSource-source).flatten(), label=r"$-(L f_{eq}+S)$")
plt.legend()
plt.show()

