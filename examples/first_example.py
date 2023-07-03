"""
A first example.
"""
import numpy as np
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver

class BoltzmannBackground():
    def __init__(self, M):
        self.vw = 1 / np.sqrt(3)
        self.velocityProfile = - np.ones(M - 1) / np.sqrt(3)
        self.fieldProfile = np.ones(M - 1)
        self.fieldProfile[M // 2:]  = 0
        self.temperatureProfile = 100 * np.ones(M - 1)
        self.polynomialBasis = "Cardinal"

class BoltzmannParticle():
    def __init__(self):
        self.msqVacuum = lambda x: 0.5 * x**2
        self.msqThermal = lambda T: 0.1 * T**2
        self.statistics = -1
        self.isOutOfEquilibrium = True
        gsq = 0.4
        self.collisionPrefactors = [gsq**2, gsq**2, gsq**2]

#self.polynomialBasis = "Chebyshev"
#self.M = M
#self.N = N

M = 20
N = 20
grid = Grid(M, N, 1, 1)
poly = Polynomial(grid)
background = BoltzmannBackground(M)
particle = BoltzmannParticle()
boltzmann = BoltzmannSolver(grid, background, particle)
print("BoltzmannSolver object =", boltzmann)
operator, source = boltzmann.buildLinearEquations()
print("operator.shape =", operator.shape)
print("source.shape =", source.shape)

deltaF = boltzmann.solveBoltzmannEquations()
print("deltaF.shape =", deltaF.shape)
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])
print("deltaF[:, 0, 0] =", deltaF[:, 0, 0])

Deltas = boltzmann.getDeltas(deltaF)
print("Deltas =", Deltas)

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
