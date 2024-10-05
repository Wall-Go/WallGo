import pytest

import WallGoCollision

def test_getIntegralForPair(collisionTensorQCD: WallGoCollision.CollisionTensor) -> None:
    """"""
    integral = collisionTensorQCD.getIntegralForPair("top", "top")
    assert integral is not None
    assert isinstance(integral, WallGoCollision.CollisionIntegral4)
    
    nonExistentIntegral = collisionTensorQCD.getIntegralForPair("dumb1", "dumb2")
    assert nonExistentIntegral is None

@pytest.mark.slow
@pytest.mark.parametrize(
    "particle1, particle2, gridPoint, expected",
    [
        ("top", "top", (2, 1, 2, 2), 0.0878282),
        ("gluon", "gluon", (2, 1, 2, 2), 0.0626843),
    ],
)
def test_collisionTensorFull(
    resultsQCD_N3: WallGoCollision.CollisionTensorResult,
    particle1: str,
    particle2: str,
    gridPoint: tuple,
    expected: float,
) -> None:
    """Checks a few elements of the QCD collision tensor."""

    resultsForPair = resultsQCD_N3.getResultsForParticlePair(particle1, particle2)

    assert resultsForPair is not None

    point = WallGoCollision.GridPoint(*gridPoint)

    err = resultsForPair.errorAt(point)

    assert resultsForPair.valueAt(
        WallGoCollision.GridPoint(*gridPoint)
    ) == pytest.approx(expected, abs=err)
