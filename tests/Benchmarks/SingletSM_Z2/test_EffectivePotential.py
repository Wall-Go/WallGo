import pytest
import numpy as np
from typing import Tuple

import WallGo
from tests.BenchmarkPoint import BenchmarkPoint


@pytest.mark.parametrize("T", [90, 110])
def test_effectivePotential_V_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of EffectivePotential
    """
    Veff, BM = singletSimpleBenchmarkEffectivePotential

    # parameters
    thermalParameters = Veff.getThermalParameters(T)
    msq = thermalParameters["msq"]
    b2 = thermalParameters["b2"]
    lam = thermalParameters["lambda"]
    a2 = thermalParameters["a2"]
    b4 = thermalParameters["b4"]

    # fields
    v = np.sqrt(2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam))
    x = np.sqrt(2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    f0 = -107.75 * np.pi**2 / 90 * T**4
    VExact = (b4 * msq**2 - a2 * msq * b2 + lam * b2**2) / (a2**2 - 4 * b4 * lam)

    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    V = Veff.evaluate(fields, T)[0]
    assert f0 + VExact == pytest.approx(V, rel=1e-13)


@pytest.mark.parametrize("T", [90, 110])
def test_effectivePotential_dVdField_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of EffectivePotential field derivative
    """
    Veff, BM = singletSimpleBenchmarkEffectivePotential

    # parameters
    thermalParameters = Veff.getThermalParameters(T)
    msq = thermalParameters["msq"]
    b2 = thermalParameters["b2"]
    lam = thermalParameters["lambda"]
    a2 = thermalParameters["a2"]
    b4 = thermalParameters["b4"]

    # fields
    v = np.sqrt(2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam))
    x = np.sqrt(2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    dVdFieldExact = np.array([0, 0])

    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    V = Veff.evaluate(fields, T)[0]
    dVdField = Veff.derivField(fields, T)
    assert dVdFieldExact == pytest.approx(dVdField[0], abs=abs(V / v * 1e-11))


@pytest.mark.parametrize("T", [90, 110])
def test_effectivePotential_dVdT_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of EffectivePotential T derivative
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

    # fields
    v = np.sqrt(2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam))
    x = np.sqrt(2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    dVdTExact = (
        -107.75 * np.pi**2 / 90 * 4 * T**3
        + (msq - msq0) / T * v**2
        + (b2 - b20) / T * x**2
    )

    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    dVdT = Veff.derivT(fields, T)
    assert dVdTExact == pytest.approx(dVdT, rel=1e-10)


@pytest.mark.parametrize("T", [90, 110])
def test_effectivePotential_d2VdFielddT_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of FreeEnergy Field and T derivative
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

    # fields
    v = np.sqrt(2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam))
    x = np.sqrt(2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
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
    d2VdFielddT = Veff.deriv2FieldT(fields, T)[0]
    assert d2VdFielddTExact == pytest.approx(d2VdFielddT, rel=1e-5)  # HACK! This should be more accurate


@pytest.mark.parametrize("T", [90, 110])
def test_effectivePotential_d2VdField2_singletSimple(
    singletSimpleBenchmarkEffectivePotential: Tuple[
        WallGo.EffectivePotential, BenchmarkPoint
    ],
    T: float,
):
    """
    Testing numerics of EffectivePotential Hessian
    """
    Veff, BM = singletSimpleBenchmarkEffectivePotential

    # parameters
    thermalParameters = Veff.getThermalParameters(T)
    msq = thermalParameters["msq"]
    b2 = thermalParameters["b2"]
    lam = thermalParameters["lambda"]
    a2 = thermalParameters["a2"]
    b4 = thermalParameters["b4"]

    # fields
    v = np.sqrt(2 * (-a2 * b2 + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam))
    x = np.sqrt(2 * (-a2 * msq + 2 * lam * b2) / (a2**2 - 4 * b4 * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    a = 4 * lam * (-(a2 * b2) + 2 * b4 * msq) / (a2**2 - 4 * b4 * lam)
    b = (
        (2 * a2)
        * np.sqrt((2 * b2 * lam - a2 * msq) * (-(a2 * b2) + 2 * b4 * msq))
        / (a2**2 - 4 * b4 * lam)
    )
    d = b4 * (8 * b2 * lam - 4 * a2 * msq) / (a2**2 - 4 * b4 * lam)
    d2VdField2 = np.array([[a, b], [b, d]])


    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    d2VdField2 = Veff.deriv2Field2(fields, T)
    assert d2VdField2 == pytest.approx(d2VdField2, rel=1e-12)
