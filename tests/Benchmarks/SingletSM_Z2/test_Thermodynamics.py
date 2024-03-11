import pytest
import numpy as np
from typing import Tuple

import WallGo
from tests.BenchmarkPoint import BenchmarkPoint


@pytest.mark.slow
@pytest.mark.parametrize("expectedTc", [108.22])
def test_singletThermodynamicsFindCriticalTemperature(
    singletBenchmarkThermo_interpolate: Tuple[WallGo.Thermodynamics, BenchmarkPoint],
    expectedTc: float,
):

    thermodynamics, BM = singletBenchmarkThermo_interpolate

    Tc = thermodynamics.findCriticalTemperature(
        dT=0.1,
        rTol=1e-6,
        paranoid=False,
    )

    assert Tc == pytest.approx(expectedTc, rel=1e-4)


def test_thermodynamics_Tc_singletSimple(
    singletSimpleBenchmarkThermodynamics: Tuple[WallGo.Thermodynamics, BenchmarkPoint],
):
    # Testing numerics of Thermodynaimcs
    thermodynamics, BM = singletSimpleBenchmarkThermodynamics

    # exact results
    Veff = thermodynamics.freeEnergyLow.effectivePotential
    p = Veff.modelParameters
    A = (
        p["a2"] / 24
        + (p["g1"] ** 2 + 3 * p["g2"] ** 2 + 8 * p["lambda"] + 4 * p["yt"] ** 2) / 16
    )
    B = p["a2"] / 6 + p["b4"] / 4
    expectedTc = np.sqrt(
        (
            (p["b2"] * np.sqrt(p["lambda"]) - p["msq"] * np.sqrt(p["b4"]))
            / (A * np.sqrt(p["b4"]) - B * np.sqrt(p["lambda"]))
        )
    )

    # compute Tc numerically
    Tc = thermodynamics.findCriticalTemperature(
        dT=0.1,
        rTol=1e-12,
        paranoid=False,
    )

    # results from freeEnergy1
    assert Tc == pytest.approx(expectedTc, rel=1e-11)
