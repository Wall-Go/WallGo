import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo

@pytest.mark.slow
def test_standardModelPhaseTrace(
    standardModelBenchmarkFreeEnergy: Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint],
):

    
    _, freeEnergyLowT, BM = standardModelBenchmarkFreeEnergy

    _,_,dT = BM.config["interpolateTemperatureRangeHighTPhase"]

    maxT = freeEnergyLowT.maxPossibleTemperature[0] + 2*dT
    expectedMaxT = BM.expectedResults["maximumTemperaturePhase2"]


    assert maxT == pytest.approx(expectedMaxT, rel=1e-3)