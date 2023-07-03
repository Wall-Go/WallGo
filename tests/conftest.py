import pytest
import numpy as np

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


@pytest.fixture
def background(M):
    return BoltzmannBackground(M)


@pytest.fixture
def particle():
    return BoltzmannParticle()
