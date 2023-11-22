"""
Testing the construction of the Liouville operator and the source terms in the Boltzmann equation
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

"""
Test objects
"""    
def buildSourceAndShouldBeSource(boltzmannSolver):
    """
    Constructs the source for Boltzmann equation,
    and a should-be source using the Liouville operator.
    """
    # coordinates
    xi, pz, pp = boltzmannSolver.grid.getCoordinates()  # non-compact
    xi = xi[:, np.newaxis, np.newaxis]
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # intertwiner matrices
    TChiMat = np.delete(boltzmannSolver.poly.matrix(boltzmannSolver.basisM, "z", endpoints=True), [0, -1], axis=0)
    TRzMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pz")
    TRpMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pp")

    # derivative matrices
    derivChi = np.delete(boltzmannSolver.poly.deriv(boltzmannSolver.basisM, "z", endpoints=True), [0, -1], axis=0)
    derivRz = boltzmannSolver.poly.deriv(boltzmannSolver.basisN, "pz")

    # background profiles
    # The ones with 1 are for the extended profile
    T = boltzmannSolver.background.temperatureProfile[1:-1, np.newaxis, np.newaxis]
    T1 = boltzmannSolver.background.temperatureProfile[:, np.newaxis, np.newaxis]
    field = boltzmannSolver.background.fieldProfile[:, 1:-1, np.newaxis, np.newaxis]
    field1 = boltzmannSolver.background.fieldProfile[:, :, np.newaxis, np.newaxis]
    v = boltzmannSolver.background.velocityProfile[1:-1, np.newaxis, np.newaxis]
    v1 = boltzmannSolver.background.velocityProfile[:, np.newaxis, np.newaxis]
    vw = boltzmannSolver.background.vw

    # fluctuation mode
    statistics = -1 if boltzmannSolver.particle.statistics == "Fermion" else 1
    # TODO: indices order not consistent across different functions.
    msq = boltzmannSolver.particle.msqVacuum(field)
    E = np.sqrt(msq + pz**2 + pp**2)
    msq1 = boltzmannSolver.particle.msqVacuum(field1)
    E1 = np.sqrt(msq1 + pz**2 + pp**2)

    Tpoly = Polynomial2(boltzmannSolver.background.temperatureProfile, boltzmannSolver.grid,  'Cardinal','z', True)
    msqpoly = Polynomial2(boltzmannSolver.particle.msqVacuum(boltzmannSolver.background.fieldProfile) ,boltzmannSolver.grid,  'Cardinal','z', True)
    vpoly = Polynomial2(boltzmannSolver.background.velocityProfile, boltzmannSolver.grid,  'Cardinal','z', True)

    # dot products with wall velocity
    gammaWall = 1 / np.sqrt(1 - vw**2)
    PWall = gammaWall * (pz - vw * E)

    # dot products with plasma profile velocity
    gammaPlasma = 1 / np.sqrt(1 - v**2)
    EPlasma = gammaPlasma * (E - v * pz)
    PPlasma = gammaPlasma * (pz - v * E)
    gammaPlasma1 = 1 / np.sqrt(1 - v1**2)
    EPlasma1 = gammaPlasma1 * (E1 - v1 * pz)

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
    fEq1 = __feq(EPlasma1 / T1, statistics)

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

    sbSource = -np.einsum("abcijk,ijk->abc", liouville, fEq1)

    # returning results
    return source, sbSource


def buildEqDistrAndShouldBeEqDistr(boltzmannSolver):
    """
    Constructs the source for Boltzmann equation,
    and a should-be source using the Liouville operator.
    """
    # coordinates
    xi, pz, pp = boltzmannSolver.grid.getCoordinates()  # non-compact
    xi = xi[:, np.newaxis, np.newaxis]
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # intertwiner matrices
    TChiMat = np.delete(boltzmannSolver.poly.matrix(boltzmannSolver.basisM, "z", endpoints=True), [0, -1], axis=0)
    TRzMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pz")
    TRpMat = boltzmannSolver.poly.matrix(boltzmannSolver.basisN, "pp")

    # derivative matrices
    derivChi = np.delete(boltzmannSolver.poly.deriv(boltzmannSolver.basisM, "z", endpoints=True), [0, -1], axis=0)
    derivRz = boltzmannSolver.poly.deriv(boltzmannSolver.basisN, "pz")

    # background profiles
    # The ones with 1 are for the extended profile
    T = boltzmannSolver.background.temperatureProfile[1:-1, np.newaxis, np.newaxis]
    T1 = boltzmannSolver.background.temperatureProfile[:, np.newaxis, np.newaxis]
    field = boltzmannSolver.background.fieldProfile[:, 1:-1, np.newaxis, np.newaxis]
    field1 = boltzmannSolver.background.fieldProfile[:, :, np.newaxis, np.newaxis]
    v = boltzmannSolver.background.velocityProfile[1:-1, np.newaxis, np.newaxis]
    v1 = boltzmannSolver.background.velocityProfile[:, np.newaxis, np.newaxis]
    vw = boltzmannSolver.background.vw

    # fluctuation mode
    statistics = -1 if boltzmannSolver.particle.statistics == "Fermion" else 1
    # TODO: indices order not consistent across different functions.
    msq = boltzmannSolver.particle.msqVacuum(field)
    E = np.sqrt(msq + pz**2 + pp**2)
    msq1 = boltzmannSolver.particle.msqVacuum(field1)
    E1 = np.sqrt(msq1 + pz**2 + pp**2)

    Tpoly = Polynomial2(boltzmannSolver.background.temperatureProfile, boltzmannSolver.grid,  'Cardinal','z', True)
    msqpoly = Polynomial2(boltzmannSolver.particle.msqVacuum(boltzmannSolver.background.fieldProfile) ,boltzmannSolver.grid,  'Cardinal','z', True)
    vpoly = Polynomial2(boltzmannSolver.background.velocityProfile, boltzmannSolver.grid,  'Cardinal','z', True)

    # dot products with wall velocity
    gammaWall = 1 / np.sqrt(1 - vw**2)
    PWall = gammaWall * (pz - vw * E)
    PWall1 = gammaWall * (pz - vw * E1)

    # dot products with plasma profile velocity
    gammaPlasma = 1 / np.sqrt(1 - v**2)
    EPlasma = gammaPlasma * (E - v * pz)
    PPlasma = gammaPlasma * (pz - v * E)
    gammaPlasma1 = 1 / np.sqrt(1 - v1**2)
    EPlasma1 = gammaPlasma1 * (E1 - v1 * pz)

    # dot product of velocities
    uwBaruPl = gammaWall * gammaPlasma * (vw - v)

    # spatial derivatives of profiles
    dTdChi = Tpoly.derivative(0).coefficients[1:-1, None, None]
    dvdChi = vpoly.derivative(0).coefficients[1:-1, None, None]
    dmsqdChi = msqpoly.derivative(0).coefficients[1:-1, None, None]
    dmsqdChi1 = msqpoly.derivative(0).coefficients[:, None, None]
    
    # derivatives of compactified coordinates
    dchidxi, drzdpz, drpdpp = boltzmannSolver.grid.getCompactificationDerivatives()
    dchidxi = dchidxi[:, np.newaxis, np.newaxis]
    drzdpz = drzdpz[np.newaxis, :, np.newaxis]
    dchidxi1, drzdpz1, drpdpp1 = boltzmannSolver.grid.getCompactificationDerivatives(endpoints=True)
    dchidxi1 = dchidxi1[:, np.newaxis, np.newaxis]

    # equilibrium distribution, and its derivative
    warnings.filterwarnings("ignore", message="overflow encountered in exp")
    fEq2 = __feq(EPlasma / T, statistics)
    fEq1 = __feq(EPlasma1 / T1, statistics)

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

    """
    sMomShape = source[0].shape
    source = np.insert(source, 0, np.zeros(sMomShape), axis=0)
    source = np.append(source, [np.zeros(sMomShape)], axis=0)
    """

    print(dchidxi1.shape)
    print(PWall1.shape)
    print(derivChi.shape)
    print(dmsqdChi1.shape)
    print(TChiMat.shape)

    ##### liouville operator #####
    liouvilleTooLarge = (
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

    #sbSourceNoBc = -np.einsum("abcijk,ijk->abc", liouvilleTooLarge, fEq1)

    liouville = liouvilleTooLarge[:,:,:,1:-1,:,:]
    liouville[:,:,:,0,:,:] = liouville[:,:,:,0,:,:] + liouvilleTooLarge[:,:,:,0,:,:]
    liouville[:,:,:,-1,:,:] = liouville[:,:,:,-1,:,:] + liouvilleTooLarge[:,:,:,-1,:,:]

    sbSource = -np.einsum("abcijk,ijk->abc", liouville, fEq2)

    # returning results
    return source, sbSource

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
source, sbSource = buildEqDistrAndShouldBeEqDistr(boltzmann)

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

exit()

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

