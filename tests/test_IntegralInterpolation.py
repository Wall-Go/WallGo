import numpy as np
import pytest

from WallSpeed.Integrals import JbIntegral, JfIntegral
from WallSpeed.InterpolatableFunction import EExtrapolationType

### Test real parts of Jb, Jf integrals

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.054003365177269e-10),
    (-20.5, 8.964742570241336)
])
def test_directJb(x: float, expectedResult: float):

    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb(x) == pytest.approx(expectedResult, rel=1e-6)
    

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.0557025829810107e-10),
    (-20.5, 11.9135695475507)
])
def test_directJf(x: float, expectedResult: float):

    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf(x) == pytest.approx(expectedResult, rel=1e-6)
    

## Interpolated Jb integral fixture, no extrapolation. The interpolation here is very rough to make this run fast
@pytest.fixture()
def Jb_interpolated():
    
    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    Jb.newInterpolationTable(1.0, 10.0, 10)
    return Jb


@pytest.mark.parametrize("x, expectedResult", [
    (2.0, -1.408868),
    (np.array([[-1.0, 0.5], [7.0, 12.0]]), np.array([[-2.81844527, -1.89083137], [-0.70078526, -0.40746745]]))
])
def test_Jb_interpolated(Jb_interpolated: JbIntegral, x, expectedResult):
    ## This also tests array input

    np.testing.assert_allclose( Jb_interpolated(x), expectedResult, rtol=1e-6)
    

### Test out-of-bounds behavior with extrapolations

## Got lazy with parametrization here, so this is just one big function now
def test_Jb_extrapolation_constant(Jb_interpolated: JbIntegral):

    Jb = Jb_interpolated
    Jb.setExtrapolationType(extrapolationTypeLower=EExtrapolationType.CONSTANT, extrapolationTypeUpper=EExtrapolationType.NONE)
    
    relativeTolerance = 1e-6

    x = -100.0 
    np.testing.assert_allclose( Jb(x), Jb(1.0), rtol=relativeTolerance)

    ## Check that we didn't modify the input for whatever reason
    assert type(x) == float
    
    x = np.array(-100.0)
    np.testing.assert_allclose( Jb(x), Jb(1.0), rtol=relativeTolerance)
    assert type(x) == np.ndarray

    x = np.array([-100.0])
    np.testing.assert_allclose( Jb(x), Jb(1.0), rtol=relativeTolerance)
    assert type(x) == np.ndarray

    x = np.array([ -20.0, 7.0, 12.0])
    np.testing.assert_allclose( Jb(x), np.array([Jb(1.0), -0.70078526, -0.40746745]) , rtol=relativeTolerance)

    Jb.setExtrapolationType(extrapolationTypeLower=EExtrapolationType.CONSTANT, extrapolationTypeUpper=EExtrapolationType.CONSTANT)

    np.testing.assert_allclose( Jb(x), np.array([Jb(1.0), -0.70078526, Jb(10.0)]) , rtol=relativeTolerance)

    Jb.setExtrapolationType(extrapolationTypeLower=EExtrapolationType.NONE, extrapolationTypeUpper=EExtrapolationType.CONSTANT)

    np.testing.assert_allclose( Jb(x), np.array([8.433656, -0.70078526, Jb(10.0)]) , rtol=relativeTolerance)
