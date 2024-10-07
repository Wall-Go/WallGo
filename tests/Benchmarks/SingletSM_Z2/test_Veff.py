import pytest
import numpy as np

# WallGo imports
import WallGo
from tests.BenchmarkPoint import BenchmarkModel


@pytest.mark.parametrize(
    "fields, temperature, expectedVeffValue",
    [
        (WallGo.Fields([110, 130]), 100, -1.19018205e09),
        (WallGo.Fields([130, 130]), 100, -1.17839699e09),
    ],
)
def test_singletModelVeffValue(
    singletBenchmarkModel: BenchmarkModel,
    fields: WallGo.Fields,
    temperature: float,
    expectedVeffValue: WallGo.Fields,
):

    # Could also take model objects as inputs instead of BM.
    # But doesn't really matter as long as the model is fast to construct

    model = singletBenchmarkModel.model

    # This tests real part only!!
    res = model.getEffectivePotential().evaluate(fields, temperature)
    assert res == pytest.approx(expectedVeffValue, rel=1e-6)


# Same as test_singletModelVeffValue but gives the Veff list of
# field-space points
@pytest.mark.parametrize(
    "fields, temperature, expectedVeffValue",
    [
        (
            WallGo.Fields([[110, 130], [130, 130]]),
            100,
            [-1.19018205e09, -1.17839699e09],
        ),
    ],
)
def test_singletModelVeffValue_manyFieldPoints(
    singletBenchmarkModel: BenchmarkModel,
    fields: WallGo.Fields,
    temperature: float,
    expectedVeffValue: WallGo.Fields,
):

    model = singletBenchmarkModel.model

    # This tests real part only!!
    res = model.getEffectivePotential().evaluate(fields, temperature)
    np.testing.assert_allclose(res, expectedVeffValue, rtol=1e-6)


@pytest.mark.parametrize(
    "initialGuess, temperature, expectedMinimum, expectedVeffValue",
    [
        (
            WallGo.Fields([0.0, 200.0]),
            100,
            WallGo.Fields([0.0, 104.86914171]),
            -1.223482e09,
        ),
        (
            WallGo.Fields([246.0, 0.0]),
            100,
            WallGo.Fields([195.03215146, 0.0]),
            -1.231926e09,
        ),
    ],
)
def test_singletModelVeffMinimization(
    singletBenchmarkModel: BenchmarkModel,
    initialGuess: WallGo.Fields,
    temperature: float,
    expectedMinimum: WallGo.Fields,
    expectedVeffValue: float,
):

    model = singletBenchmarkModel.model

    resMinimum, resValue = model.getEffectivePotential().findLocalMinimum(
        initialGuess, temperature
    )

    # The expected value is for full V(phi) + constants(T) so include that
    resValue = model.getEffectivePotential().evaluate(resMinimum, temperature)

    np.testing.assert_allclose(resMinimum, expectedMinimum, rtol=1e-3)
    np.testing.assert_allclose(resValue, expectedVeffValue, rtol=1e-3)


# ---- Derivative tests


@pytest.mark.parametrize(
    "fields, temperature, expectedVeffValue",
    [
        (
            WallGo.Fields([110, 130]),
            100,
            WallGo.Fields([512754.5552253, 1437167.06776619]),
        ),
        (
            WallGo.Fields([130, 130]),
            100,
            WallGo.Fields([670916.4147377, 1712203.95803452]),
        ),
    ],
)
def test_singletModelDerivField(
    singletBenchmarkModel: BenchmarkModel,
    fields: WallGo.Fields,
    temperature: float,
    expectedVeffValue: WallGo.Fields,
):
    model = singletBenchmarkModel.model

    res = model.getEffectivePotential().derivField(fields, temperature)

    assert res == pytest.approx(expectedVeffValue, rel=1e-4)


@pytest.mark.parametrize(
    "fields, temperature, expectedVeffValue",
    [
        (WallGo.Fields([110, 130]), 100, -46660927.93128967),
        (WallGo.Fields([130, 130]), 100, -46494985.30003357),
    ],
)
def test_singletModelDerivT(
    singletBenchmarkModel: BenchmarkModel,
    fields: WallGo.Fields,
    temperature: float,
    expectedVeffValue: float,
):

    model = singletBenchmarkModel.model

    res = model.getEffectivePotential().derivT(fields, temperature)

    assert res == pytest.approx(expectedVeffValue, rel=1e-4)
