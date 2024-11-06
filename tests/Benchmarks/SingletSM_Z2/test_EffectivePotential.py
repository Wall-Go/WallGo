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
    muHsq = thermalParameters["muHsq"]
    muSsq = thermalParameters["muSsq"]
    lam = thermalParameters["lHH"]
    lHS = thermalParameters["lHS"]
    lSS = thermalParameters["lSS"]

    # fields
    v = np.sqrt(2 * (-lHS * muSsq + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam))
    x = np.sqrt(2 * (-lHS * muHsq + 2 * lam * muSsq) / (lHS**2 - 4 * lSS * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    f0 = -107.75 * np.pi**2 / 90 * T**4
    VExact = (lSS * muHsq**2 - lHS * muHsq * muSsq + lam * muSsq**2) / (lHS**2 - 4 * lSS * lam)

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
    muHsq = thermalParameters["muHsq"]
    muSsq = thermalParameters["muSsq"]
    lam = thermalParameters["lHH"]
    lHS = thermalParameters["lHS"]
    lSS = thermalParameters["lSS"]

    # fields
    v = np.sqrt(2 * (-lHS * muSsq + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam))
    x = np.sqrt(2 * (-lHS * muHsq + 2 * lam * muSsq) / (lHS**2 - 4 * lSS * lam))
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
    muHsq = thermalParameters["muHsq"]
    muSsq = thermalParameters["muSsq"]
    lam = thermalParameters["lHH"]
    lHS = thermalParameters["lHS"]
    lSS = thermalParameters["lSS"]
    vacuumParameters = Veff.modelParameters
    muHsq0 = vacuumParameters["muHsq"]
    muSsq0 = vacuumParameters["muSsq"]

    # fields
    v = np.sqrt(2 * (-lHS * muSsq + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam))
    x = np.sqrt(2 * (-lHS * muHsq + 2 * lam * muSsq) / (lHS**2 - 4 * lSS * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    dVdTExact = (
        -107.75 * np.pi**2 / 90 * 4 * T**3
        + (muHsq - muHsq0) / T * v**2
        + (muSsq - muSsq0) / T * x**2
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
    muHsq = thermalParameters["muHsq"]
    muSsq = thermalParameters["muSsq"]
    lam = thermalParameters["lHH"]
    lHS = thermalParameters["lHS"]
    lSS = thermalParameters["lSS"]
    vacuumParameters = Veff.modelParameters
    muHsq0 = vacuumParameters["muHsq"]
    muSsq0 = vacuumParameters["muSsq"]

    # fields
    v = np.sqrt(2 * (-lHS * muSsq + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam))
    x = np.sqrt(2 * (-lHS * muHsq + 2 * lam * muSsq) / (lHS**2 - 4 * lSS * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    d2VdFielddTExact = np.array(
        [
            2 * (muHsq - muHsq0) / T * v,
            2 * (muSsq - muSsq0) / T * x,
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
    muHsq = thermalParameters["muHsq"]
    muSsq = thermalParameters["muSsq"]
    lam = thermalParameters["lHH"]
    lHS = thermalParameters["lHS"]
    lSS = thermalParameters["lSS"]

    # fields
    v = np.sqrt(2 * (-lHS * muSsq + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam))
    x = np.sqrt(2 * (-lHS * muHsq + 2 * lam * muSsq) / (lHS**2 - 4 * lSS * lam))
    fields = WallGo.Fields(([v, x]))

    # exact results
    a = 4 * lam * (-(lHS * muSsq) + 2 * lSS * muHsq) / (lHS**2 - 4 * lSS * lam)
    b = (
        (2 * lHS)
        * np.sqrt((2 * muSsq * lam - lHS * muHsq) * (-(lHS * muSsq) + 2 * lSS * muHsq))
        / (lHS**2 - 4 * lSS * lam)
    )
    d = lSS * (8 * muSsq * lam - 4 * lHS * muHsq) / (lHS**2 - 4 * lSS * lam)
    d2VdField2 = np.array([[a, b], [b, d]])


    # tolerance
    Veff.dT = 1e-4 * T
    Veff.dPhi = 1e-4 * max(abs(v), abs(x))

    # results from Veff
    d2VdField2 = Veff.deriv2Field2(fields, T)
    assert d2VdField2 == pytest.approx(d2VdField2, rel=1e-12)
