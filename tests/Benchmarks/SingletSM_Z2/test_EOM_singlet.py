import pytest
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo


@pytest.mark.slow
def test_equilibriumEOM_singlet(
    singletBenchmarkEOM_equilibrium: Tuple[WallGo.EOM, BenchmarkPoint]
):
    """
    Compute the wall velocity from the loop without out-of-equilibrium effects.
    This should match the LTE wall velocity
    """

    eom, BM = singletBenchmarkEOM_equilibrium

    vwEOM, _ = eom.findWallVelocityMinimizeAction()
    vwLTE = BM.expectedResults["vwLTE"]

    # Currently the wall velocity solver in EOM has hardcoded absolute
    # tolerance of 1e-3. So no point testing for more precision than that

    print(f"{vwEOM=}")

    assert vwEOM == pytest.approx(vwLTE, abs=1e-3)
