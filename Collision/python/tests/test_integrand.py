import pytest

import WallGoCollision

@pytest.mark.parametrize(
    "particle1, particle2, basisSize, p2, phi2, phi3, cosTheta2, cosTheta3, gridPoint, expected",
    [
        ("top", "top", 11, 1.0, 1.0, 1.0, 0.5, 0.5, WallGoCollision.GridPoint(2, 1, 1, 1), 0.0877125),
    ],
)
def test_integrandValue(
    collisionTensorQCD: WallGoCollision.CollisionTensor,
    particle1: str,
    particle2: str,
    basisSize: int,
    gridPoint: WallGoCollision.GridPoint,
    p2: float,
    phi2: float,
    phi3: float,
    cosTheta2: float,
    cosTheta3: float,
    expected: float) -> None:
    """"""

    collisionTensorQCD.changePolynomialBasisSize(basisSize)
    integral = collisionTensorQCD.getIntegralForPair(particle1, particle2)

    res = integral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, gridPoint)
    assert res == expected