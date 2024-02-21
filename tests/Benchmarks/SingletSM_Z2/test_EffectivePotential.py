import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo


@pytest.mark.parametrize("T", [100])
def test_effectivePotential_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of FreeEnergy
    """
    Veff, BM = singletSimpleBenchmarkEffectivePotential

    # parameters
    thermalParameters = Veff.getThermalParameters(T)
    msq = thermalParameters["msq"]
    b2 = thermalParameters["b2"]
    lam = thermalParameters["lambda"]
    a2 = thermalParameters["a2"]
    b4 = thermalParameters["b4"]
    vacuumParameters = Veff.modelParameters
    msq0 = vacuumParameters["msq"]
    b20 = vacuumParameters["b2"]

    # checking T is appropriate
    vsq = 2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam)
    xsq = 2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam)
    assert vsq >= 0 and xsq >= 0, f"Saddlepoint doesn't exist at {T=}"

    # fields
    v = np.sqrt(vsq)
    x = np.sqrt(xsq)
    fields = WallGo.Fields(([v, x]))

    # exact results
    f0 = -107.75 * np.pi**2 / 90 * T**4
    VExact = (b4 * msq**2 - a2 * msq * b2 + lam * b2**2) / (a2**2 - 4 * b4 * lam)
    dVdFieldExact = np.array([0, 0])
    dVdTExact = (
        -107.75 * np.pi**2 / 90 * 4 * T**3
        + (msq - msq0) / T * v**2
        + (b2 - b20) / T * x**2
    )
    a = 4 * lam * (-(a2 * b2) + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam)
    b = (
        (2 * a2)
        * np.sqrt((2 * b2 * lam - a2 * msq) * (-(a2 * b2) + 2 * b4 * msq))
        / (a2**2 - 4 * b4 * lam)
    )
    d = b4 * (8 * b2 * lam - 4 * a2 * msq) / (a2**2 - 4 * b4 * lam)
    d2VdField2 = np.array([[a, b], [b, d]])
    d2VdFielddTExact = np.array(
        [
            2 * (msq - msq0) / T * v,
            2 * (b2 - b20) / T * x,
        ]
    )

    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    V = Veff.evaluate(fields, T)[0]
    assert f0 + VExact == pytest.approx(V, rel=1e-13)
    dVdField = Veff.derivField(fields, T)
    assert dVdFieldExact == pytest.approx(dVdField[0], abs=abs(V / v * 1e-12))
    dVdT = Veff.derivT(fields, T)
    assert dVdTExact == pytest.approx(dVdT[0], rel=1e-12)
    d2VdField2 = Veff.deriv2Field2(fields, T)
    assert d2VdField2 == pytest.approx(d2VdField2, rel=1e-12)
    d2VdFielddT = Veff.deriv2FieldT(fields, T)
    assert d2VdFielddTExact == pytest.approx(d2VdFielddT, rel=1e-7)  # HACK! This should be more accurate
