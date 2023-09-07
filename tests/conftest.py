import pytest
import numpy as np
from WallSpeed.Boltzmann import BoltzmannBackground
from WallSpeed.model import Particle


@pytest.fixture
def background(M):
    vw = 1 / np.sqrt(3)
    v = - np.ones(M - 1) / np.sqrt(3)
    field = np.ones((M - 1,))
    field[M // 2:]  = 0
    T = 100 * np.ones(M - 1)
    return BoltzmannBackground(
        vw=vw,
        velocityProfile=v,
        fieldProfile=field,
        temperatureProfile=T,
        polynomialBasis="Cardinal",
    )


@pytest.fixture
def particle():
    return Particle(
        name="top",
        msqVacuum=lambda phi: 0.5 * phi**2,
        msqThermal=lambda T: 0.1 * T**2,
        statistics="Fermion",
        inEquilibrium=False,
        ultrarelativistic=False,
        collisionPrefactors=[1, 1, 1],
    )
