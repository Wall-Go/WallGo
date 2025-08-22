import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo

@pytest.mark.slow
def test_standardModelThermodynamicsFindCriticalTemperature(
    standardModelBenchmarkThermo_interpolate: Tuple[WallGo.Thermodynamics, BenchmarkPoint],
):

    thermodynamics, BM = standardModelBenchmarkThermo_interpolate

    Tc = thermodynamics.findCriticalTemperature(
        dT=0.01,
        rTol=1e-8,
        paranoid=True,
    )

    expectedTc = BM.expectedResults["Tc"]

    assert Tc == pytest.approx(expectedTc, rel=1e-4)
