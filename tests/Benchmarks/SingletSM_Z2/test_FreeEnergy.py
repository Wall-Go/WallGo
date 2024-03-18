import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo


@pytest.mark.parametrize("T", [90, 110])
def test_freeEnergy_singletSimple(
    singletSimpleBenchmarkFreeEnergy: Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint],
    T: float,
):
    """
    Testing numerics of FreeEnergy
    """
    freeEnergy1, freeEnergy2, BM = singletSimpleBenchmarkFreeEnergy

    # exact results
    thermalParameters = freeEnergy1.effectivePotential.getThermalParameters(T)
    f0 = -107.75 * np.pi ** 2 / 90 * T ** 4
    vExact = np.sqrt(-thermalParameters["msq"] / thermalParameters["lambda"])
    VvExact = -0.25 * thermalParameters["msq"] ** 2 / thermalParameters["lambda"]
    xExact = np.sqrt(-thermalParameters["b2"] / thermalParameters["b4"])
    VxExact = -0.25 * thermalParameters["b2"] ** 2 / thermalParameters["b4"]

    # tolerance
    rTol = 1e-5
    aTol = rTol * T

    # results from freeEnergy1
    v, x, V = freeEnergy1(T)
    assert 0 == pytest.approx(v, abs=aTol)
    assert xExact == pytest.approx(x, rel=rTol)
    assert f0 + VxExact == pytest.approx(V, rel=rTol)

    # results from freeEnergy2
    v, x, V = freeEnergy2(T)
    assert vExact == pytest.approx(v, rel=rTol)
    assert 0 == pytest.approx(x, abs=aTol)
    assert f0 + VvExact == pytest.approx(V, rel=rTol)
