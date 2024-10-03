from typing import Union
import numpy as np
import numpy.typing as npt
import pytest

from WallGo import InterpolatableFunction, EExtrapolationType

from WallGo import PotentialTools
from WallGo.PotentialTools import JbIntegral, JfIntegral

### Test real parts of Jb, Jf integrals


@pytest.mark.parametrize(
    "x, expectedResult",
    [
        (800, [-1.0470380039436925e-10, 0]),
        (-20.5, [8.964742569922326, 7.3412855807505775]),
    ],
)
def test_directJb(x: float, expectedResult: np.array) -> None:

    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb(x) == pytest.approx(expectedResult, rel=1e-6)


@pytest.mark.parametrize(
    "x, expectedResult",
    [(5, [0.11952474943876305, 0]), (-1, [0.5167030334522096, -0.5890486225480862])],
)
def test_directJb_derivative(x: float, expectedResult: float) -> None:

    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb.derivative(x, 1, False) == pytest.approx(expectedResult, rel=1e-6)


@pytest.mark.parametrize(
    "x, expectedResult",
    [(800, [-1.047038003943492e-10, 0]), (-20.5, [11.913574189883063, 4.962444471425764])],
)
def test_directJf(x: float, expectedResult: float) -> None:

    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf(x) == pytest.approx(expectedResult, rel=1e-6)


@pytest.mark.parametrize(
    "x, expectedResult",
    [(5, [0.1113267810730111, 0]), (-1, [0.5376680405566582, -0.19634954084936207])],
)
def test_directJf_derivative(x: float, expectedResult: float) -> None:

    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf.derivative(x, 1, False) == pytest.approx(expectedResult, rel=1e-6)


## Interpolated Jb integral fixture, no extrapolation. The interpolation here is very rough to make this run fast
@pytest.fixture()
def Jb_interpolated() -> InterpolatableFunction:
    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    Jb.newInterpolationTable(1.0, 10.0, 100)
    return Jb


@pytest.fixture()
def Jf_interpolated() -> InterpolatableFunction:
    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    Jf.newInterpolationTable(1.0, 10.0, 100)
    return Jf


@pytest.mark.parametrize(
    "x, expectedResult",
    [
        (2.0, [-1.4088678478127121, 0]),
        (
            np.array([-1.0, 0.5]),
            np.array(
                [[-2.8184452672778013, 0.4254240051736174], [-1.8908313667001768, 0]]
            ),
        ),
        (
            np.array([[-1.0, 0.5], [7.0, 12.0]]),
            np.array(
                [
                    [
                        [-2.8184452672778013, 0.4254240051736174],
                        [-1.8908313667001768, 0],
                    ],
                    [[-0.700785264789041, 0], [-0.4074674546874202, 0]],
                ]
            ),
        ),
    ],
)
def test_Jb_interpolated(
    Jb_interpolated: JbIntegral,
    x: Union[float, np.array],
    expectedResult: Union[float, np.array],
) -> None:
    ## This also tests array input
    np.testing.assert_allclose(Jb_interpolated(x), expectedResult, rtol=1e-6)


@pytest.mark.parametrize("x", [-5, -1, 0, 0.5, 1, 5, 10])
def test_Jb_derivative_interpolated(Jb_interpolated: JbIntegral, x: float) -> None:
    np.testing.assert_allclose(
        Jb_interpolated.derivative(x, 1, True),
        Jb_interpolated.derivative(x, 1, False),
        rtol=1e-4,
    )


@pytest.mark.parametrize("x", [-5, -1, 0, 0.5, 1, 5, 10])
def test_Jb_second_derivative_interpolated(
    Jb_interpolated: JbIntegral, x: float
) -> None:
    np.testing.assert_allclose(
        Jb_interpolated.derivative(x, 2, True),
        Jb_interpolated.derivative(x, 2, False),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("x", [-5, -1, 0, 0.5, 1, 5, 10])
def test_Jf_derivative_interpolated(Jf_interpolated: JfIntegral, x: float) -> None:
    np.testing.assert_allclose(
        Jf_interpolated.derivative(x, 1, True),
        Jf_interpolated.derivative(x, 1, False),
        rtol=1e-4,
    )


@pytest.mark.parametrize("x", [-5, -1, 0, 0.5, 1, 5, 10])
def test_Jf_second_derivative_interpolated(
    Jf_interpolated: JfIntegral, x: float
) -> None:
    np.testing.assert_allclose(
        Jf_interpolated.derivative(x, 2, True),
        Jf_interpolated.derivative(x, 2, False),
        rtol=1e-3,
        atol=1e-3,
    )


### Test out-of-bounds behavior with extrapolations


## Got lazy with parametrization here, so this is just one big function now
def test_Jb_extrapolation_constant(Jb_interpolated: JbIntegral) -> None:

    Jb = Jb_interpolated
    Jb.setExtrapolationType(
        extrapolationTypeLower=EExtrapolationType.CONSTANT,
        extrapolationTypeUpper=EExtrapolationType.NONE,
    )

    relativeTolerance = 1e-6

    x = -100.0
    np.testing.assert_allclose(Jb(x), Jb(1.0), rtol=relativeTolerance)

    ## Check that we didn't modify the input for whatever reason
    assert isinstance(x, float)

    x = np.asarray(-100.0)
    np.testing.assert_allclose(Jb(x), Jb(1.0), rtol=relativeTolerance)
    assert isinstance(x, np.ndarray)

    x = np.array([-100.0])
    np.testing.assert_allclose(Jb(x), Jb([1.0]), rtol=relativeTolerance)
    assert isinstance(x, np.ndarray)

    x = np.array([-20.0, 7.0, 12.0])
    np.testing.assert_allclose(
        Jb(x), np.array([Jb(1.0), [-0.700785264789041, 0], [-0.4074674546874202, 0]]), rtol=relativeTolerance
    )

    Jb.setExtrapolationType(
        extrapolationTypeLower=EExtrapolationType.CONSTANT,
        extrapolationTypeUpper=EExtrapolationType.CONSTANT,
    )

    np.testing.assert_allclose(
        Jb(x), np.array([Jb(1.0), [-0.700785264789041, 0], Jb(10.0)]), rtol=relativeTolerance
    )

    Jb.setExtrapolationType(
        extrapolationTypeLower=EExtrapolationType.NONE,
        extrapolationTypeUpper=EExtrapolationType.CONSTANT,
    )

    np.testing.assert_allclose(
        Jb(x), np.array([[8.433656189032257, 7.56219003706576], [-0.700785264789041, 0], Jb(10.0)]), rtol=relativeTolerance
    )


##
def test_Jb_extend_range(Jb_interpolated: JbIntegral) -> None:

    Jb = Jb_interpolated
    relativeTolerance = 1e-6

    newMin = Jb.interpolationRangeMin() - 2.0
    newMax = Jb.interpolationRangeMax() + 3

    ## evaluate these directly for later comparison
    JbNewMin_direct = Jb(newMin)
    JbNewMax_direct = Jb(newMax)

    Jb.extendInterpolationTable(newMin, Jb.interpolationRangeMax(), 2, 0)

    assert Jb.interpolationRangeMin() == pytest.approx(newMin, rel=relativeTolerance)
    assert Jb(newMin) == pytest.approx(JbNewMin_direct, rel=relativeTolerance)

    Jb.extendInterpolationTable(Jb.interpolationRangeMin(), newMax, 0, 2)

    assert Jb.interpolationRangeMax() == pytest.approx(newMax, rel=relativeTolerance)
    assert Jb(newMax) == pytest.approx(JbNewMax_direct, rel=relativeTolerance)

    ## This shouldn't do anything:
    fakeNewMax = newMax - 2.0
    Jb.extendInterpolationTable(Jb.interpolationRangeMin(), fakeNewMax, 0, 2)

    assert Jb.interpolationRangeMax() == pytest.approx(newMax, rel=relativeTolerance)
    assert Jb(newMax) == pytest.approx(JbNewMax_direct, rel=relativeTolerance)
