import pytest

import WallGoCollision

@pytest.mark.parametrize(
    "particle1, particle2, basisSize, p2, phi2, phi3, cosTheta2, cosTheta3, gridPoint, expected",
    [
        ("top", "top", 11, 1.5, 0.7, 1.1, 0.5, -0.23, WallGoCollision.GridPoint(2, 1, 1, 1), 0.000347083678371972),
        ("top", "top", 11, 0.2, 0.7, 1.0, 0.2, 0.7, WallGoCollision.GridPoint(3, 6, 4, 5), 1.5657791721961197e-05)
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
    assert res == pytest.approx(expected, rel=1e-3)