import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallSpeed


@pytest.mark.slow
def test_equilibriumEOM_singlet(singletBenchmarkEOM_equilibrium: Tuple[WallSpeed.EOM, BenchmarkPoint]):
    """
    Compute the wall velocity from the loop without out-of-equilibrium effects.
    This should match the LTE wall velocity
    """
    
    eom, BM = singletBenchmarkEOM_equilibrium

    vwLoop = eom.findWallVelocityMinimizeAction()[0]
    vwLTE = BM.expectedResults["vwLTE"]

    assert(vwLoop == pytest.approx(vwLTE, rel = 1e-2))
