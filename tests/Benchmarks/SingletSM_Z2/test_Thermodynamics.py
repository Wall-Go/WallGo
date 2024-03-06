import pytest
import numpy as np
from typing import Tuple

import WallGo
from tests.BenchmarkPoint import BenchmarkPoint


@pytest.mark.slow
@pytest.mark.parametrize("expectedTc", [108.22])
def test_singletThermodynamicsFindCriticalTemperature(
    singletBenchmarkThermo: Tuple[WallGo.Thermodynamics, BenchmarkPoint],
    expectedTc: float,
):

    thermodynamics, BM = singletBenchmarkThermo

    Tc = thermodynamics.findCriticalTemperature(
        dT=0.1, rTol=1e-6, paranoid=True,
    )

    assert Tc == pytest.approx(expectedTc, rel=1e-3)


"""
def test_freeEnergy_Tc_singletSimple(
    singletSimpleBenchmarkFreeEnergy: Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint],
):
    # Testing numerics of FreeEnergy
    freeEnergy1, freeEnergy2, BM = singletSimpleBenchmarkFreeEnergy

    # exact results
    p = Veff.modelParameters
    A = p["a2"] / 24 + (p["g1"]**2 + 3 * p["g2"]**2 + 8 * p["lambda"] + 4 * p["yt"]**2) / 16
    B = p["a2"] / 6 + p["b4"] / 4
    Tc = np.sqrt(
        -((p["b2"] * np.sqrt(p["lambda"]) + p["msq"] * np.sqrt(p["b4"]))
        /(A*np.sqrt(p["b4"]) + B*np.sqrt(p["lambda"])))
    )

    # tolerance
    rTol = 1e-5

    # compute Tc numerically
    freeEnergy1.findCriticalTemperature(...)

    # results from freeEnergy1
    assert Tc == ...
"""