import pytest
from typing import Tuple

import WallGo
from tests.BenchmarkPoint import BenchmarkPoint


@pytest.mark.slow
def test_equilibriumEOM_singlet(
    singletBenchmarkEOM_equilibrium: Tuple[WallGo.EOM, BenchmarkPoint]
) -> None:
    """
    Compute the wall velocity from the loop without out-of-equilibrium effects.
    This should match the LTE wall velocity
    """

    eom, BM = singletBenchmarkEOM_equilibrium

    results = eom.findWallVelocityDeflagrationHybrid()
    vwEOM = results.wallVelocity
    vwLTE = BM.expectedResults["vwLTE"]

    # Currently the wall velocity solver in EOM has hardcoded absolute
    # tolerance of 1e-3. So no point testing for more precision than that

    assert vwEOM == pytest.approx(vwLTE, abs=1e-3)


@pytest.mark.slow
def test_detonationEOM_singlet(
    singletBenchmarkEOM: Tuple[WallGo.EOM, BenchmarkPoint]
) -> None:
    """
    Compute a detonation, and check output as expected.
    """
    return None

    eom, BM = singletBenchmarkEOM

    # code copied from WallGoManager
    vmin = max(eom.hydrodynamics.vJ + 1e-3, eom.hydrodynamics.slowestDeton())
    vmax = 0.99

    results = eom.findWallVelocityDeflagrationHybrid(
        vmin=vmin,
        vmax=vmax,
    )

    assert results.success

    print(results.solutionType)

    assert results.solutionType in [
        WallGo.ESolutionType.DETONATION,
        WallGo.ESolutionType.RUNAWAY,
    ]

    assert 0 < results.wallVelocity <= 1
    assert eom.hydrodynamics.vJ < results.wallVelocity <= 1
    assert vmin <= results.wallVelocity <= vmax
