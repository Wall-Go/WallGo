import pytest

import WallGoCollision


@pytest.mark.slow
@pytest.mark.parametrize(
    "particle1, particle2, gridPoint, expected",
    [
        ("top", "top", (2, 1, 1, 1), 0.0877125),
        ("gluon", "gluon", (2, 1, 1, 1), 0.0611469),
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
