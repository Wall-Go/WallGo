import pytest
import numpy as np
import WallSpeed


def background(M):
    vw = 0#1 / np.sqrt(3)
    v = - np.ones(M - 1) / np.sqrt(3)
    v += 0.01 * np.sin(10 * 2 * np.pi * np.arange(M - 1))
    field = np.ones((M - 1,))
    field[M // 2:]  = 0
    field += 0.1 * np.sin(7 * 2 * np.pi * np.arange(M - 1) + 6)
    T = 100 * np.ones(M - 1)
    T += 1 * np.sin(11 * 2 * np.pi * np.arange(M - 1) + 6)
    return WallSpeed.BoltzmannBackground(
        vw=vw,
        velocityProfile=v,
        fieldProfile=field,
        temperatureProfile=T,
        polynomialBasis="Cardinal",
    )


@pytest.fixture
def particle():
    return WallSpeed.Particle(
        name="top",
        msqVacuum=lambda phi: 0.5 * phi**2,
        msqThermal=lambda T: 0.1 * T**2,
        statistics="Fermion",
        inEquilibrium=False,
        ultrarelativistic=False,
        collisionPrefactors=[1, 1, 1],
    )
