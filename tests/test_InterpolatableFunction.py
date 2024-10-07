import numpy as np
import pytest

from WallGo import InterpolatableFunction, EExtrapolationType

class DumbVectorFunction(InterpolatableFunction):
    """Vector valued function for testing interpolations."""

    def __init__(self) -> None:
        """"""
        super().__init__(bUseAdaptiveInterpolation=False,
                         initialInterpolationPointCount=1000,
                         returnValueCount=4)

    #~ Begin InterpolatableFunction interface
    def _functionImplementation(self, xIn: float | list[float] | np.ndarray) -> np.ndarray:
        """"""
        x = np.asanyarray(xIn)
        # apply element wise as required by InterpolatableFunction
        return np.stack([42.0 * np.ones_like(x), x, np.sqrt(x), x**2], axis=-1)
    #~

def test_vectorFunctionShape() -> None:
    """"""
    
    f = DumbVectorFunction()

    res = f(2.0)
    assert len(res) == 4

    res = f([1.0, 2.0, 3.0])
    assert res.shape == (3, 4)

@pytest.mark.parametrize("x",
                        # choose random inputs that are not directly on the interpolation table
                        [
                            2.354512,
                            [2.354512, 5.354, 1.1992]
                        ]
)
def test_vectorFunctionInterpolation(x: float | list[float]) -> None:
    """"""

    f = DumbVectorFunction()
    f.newInterpolationTable(0.0, 10.0, 100)
    
    assert f.hasInterpolation()
    
    fxExact = f(x, bUseInterpolatedValues=False)
    fxInterpolated = f(x)

    # Should not be exactly equal if interpolation is used
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, fxExact, fxInterpolated)

    # but should be close enough
    np.testing.assert_allclose(fxExact, fxInterpolated, rtol=0.05, atol=1e-6)

@pytest.mark.parametrize("x",
                        [
                            2.354512,
                            [0.0, 5.354, 11.1992],
                            [0.0, 0.5, 11.0], # only extrapolated values
                        ]
)
def test_outOfBoundsShape(x: float | list[float]) -> None:
    """"""
    f = DumbVectorFunction()
    f.setExtrapolationType(EExtrapolationType.CONSTANT, EExtrapolationType.FUNCTION)

    fxNoInterp = f(x)
    f.newInterpolationTable(1.0, 10.0, 10)

    assert f(x).shape == fxNoInterp.shape


def test_outOfBoundsMixed() -> None:
    """Different out-of-bounds behavior on lower and upper ends"""
    f = DumbVectorFunction()
    f.setExtrapolationType(EExtrapolationType.CONSTANT, EExtrapolationType.ERROR)
    f.newInterpolationTable(1.0, 10.0, 10)

    resInterp = f(0.0, bUseInterpolatedValues=True)
    resBoundary = f(1.0)

    # CONSTANT extrapolation means we use the boundary value
    np.testing.assert_allclose(resInterp, resBoundary, rtol=1e-3, atol=1e-6)

    # these should be fine:
    f(5.42)
    f([0.0, 2., 5.3])

    # but these should raise:
    with pytest.raises(ValueError):
        f(50.0)

    with pytest.raises(ValueError):
        f([0.0, 2., 55.0])
    
def test_outOfBoundsNoExtrapolation() -> None:
    """"""
    f = DumbVectorFunction()
    f.setExtrapolationType(EExtrapolationType.NONE, EExtrapolationType.NONE)
    f.newInterpolationTable(1.0, 10.0, 10)

    # No extrapolation, so should default to direct evaluate even if using interpolations
    np.testing.assert_array_equal(f(0.232, bUseInterpolatedValues=True), f(0.232))

    x = [0.232, 4.3, 8.2, 11.5]
    np.testing.assert_array_equal(f(x, bUseInterpolatedValues=True), f(x))


@pytest.mark.parametrize("x",
                        [
                            # use inputs that are not directly on the interpolation table
                            2.354512,
                            [0.0, 5.354, 11.1992],
                            [0.0, 0.5, 11.0], # only extrapolated values
                        ]
)
def test_outOfBoundsExtrapolateInterpolation(x: float | list[float]) -> None:
    """"""
    f = DumbVectorFunction()
    f.setExtrapolationType(EExtrapolationType.FUNCTION, EExtrapolationType.FUNCTION)
    f.newInterpolationTable(1.0, 10.0, 10)

    # Shouldn't be exactly equal to directly evaluated values
    np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, f(x, bUseInterpolatedValues=False), f(x))