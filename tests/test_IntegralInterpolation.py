import numpy as np
import numpy.typing as npt
import pytest

from WallGo import InterpolatableFunction
from WallGo.integrals import JbIntegral, JfIntegral
from WallGo.interpolatableFunction import EExtrapolationType

### Test real parts of Jb, Jf integrals

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.054003365177269e-10),
    (-20.5, 8.964742570241336)
])
def test_directJb(x: float, expectedResult: float):

    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb(x) == pytest.approx(expectedResult, rel=1e-6)

@pytest.mark.parametrize("x, expectedResult", [
    (5, 0.1195247494387632),
    (-1, 0.516703124731066)
])
def test_directJb_derivative(x: float, expectedResult: float):

    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb.derivative(x,1,False) == pytest.approx(expectedResult, rel=1e-6)
    

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.0557025829810107e-10),
    (-20.5, 11.9135695475507)
])
def test_directJf(x: float, expectedResult: float):

    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf(x) == pytest.approx(expectedResult, rel=1e-6)
    
@pytest.mark.parametrize("x, expectedResult", [
    (5, 0.1113267810730111),
    (-1, 0.5376680405566582)
])
def test_directJf_derivative(x: float, expectedResult: float):

    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf.derivative(x,1,False) == pytest.approx(expectedResult, rel=1e-6)
    

## Interpolated Jb integral fixture, no extrapolation. The interpolation here is very rough to make this run fast
@pytest.fixture()
def Jb_interpolated():
    
    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    Jb.newInterpolationTable(1.0, 10.0, 100)
    return Jb

@pytest.fixture()
def Jf_interpolated():
    
    Jf = JfIntegral(bUseAdaptiveInterpolation=False)
    Jf.newInterpolationTable(1.0, 10.0, 100)
    return Jf


@pytest.mark.parametrize("x, expectedResult", [
    (2.0, -1.408868),
    (np.array([-1.0, 0.5]), np.array([-2.81844527, -1.89083137])),
    (np.array([[-1.0, 0.5], [7.0, 12.0]]), np.array([[-2.81844527, -1.89083137], [-0.70078526, -0.40746745]])),
])
def test_Jb_interpolated(Jb_interpolated: JbIntegral, x, expectedResult):
    ## This also tests array input

    np.testing.assert_allclose( Jb_interpolated(x), expectedResult, rtol=1e-6)
    
@pytest.mark.parametrize("x", [-5,-1,0,0.5,1,5,10])
def test_Jb_derivative_interpolated(Jb_interpolated: JbIntegral, x):
    np.testing.assert_allclose( Jb_interpolated.derivative(x, 1, True), Jb_interpolated.derivative(x, 1, False), rtol=1e-4)
    
@pytest.mark.parametrize("x", [-5,-1,0,0.5,1,5,10])
def test_Jb_second_derivative_interpolated(Jb_interpolated: JbIntegral, x):
    np.testing.assert_allclose( Jb_interpolated.derivative(x, 2, True), Jb_interpolated.derivative(x, 2, False), rtol=1e-3, atol=1e-3)
    
@pytest.mark.parametrize("x", [-5,-1,0,0.5,1,5,10])
def test_Jf_derivative_interpolated(Jf_interpolated: JfIntegral, x):
    np.testing.assert_allclose( Jf_interpolated.derivative(x, 1, True), Jf_interpolated.derivative(x, 1, False), rtol=1e-4)
    
@pytest.mark.parametrize("x", [-5,-1,0,0.5,1,5,10])
def test_Jf_second_derivative_interpolated(Jf_interpolated: JfIntegral, x):
    np.testing.assert_allclose( Jf_interpolated.derivative(x, 2, True), Jf_interpolated.derivative(x, 2, False), rtol=1e-3, atol=1e-3)
    

class JbJf(InterpolatableFunction):
    """Function that returns Jb and Jf simultaneously in array format. This emulates a vector-valued function.
    """
    Jb = JbIntegral(bUseAdaptiveInterpolation=False)
    Jf = JfIntegral(bUseAdaptiveInterpolation=False)

    def _functionImplementation(self, x: npt.ArrayLike) -> npt.ArrayLike:
        return np.column_stack((self.Jb(x), self.Jf(x)))    
    
@pytest.fixture()
def JbJf_interpolated():
    """Interpolate Jb and Jf simultaneously as a "vector valued" function
    """

    JbJf_interpolated = JbJf(bUseAdaptiveInterpolation=False, returnValueCount=2)

    JbJf_interpolated.newInterpolationTable(1.0, 10.0, 10)
    return JbJf_interpolated

    
## just shorthands because I'm lazy
def jb(x: npt.ArrayLike) -> npt.ArrayLike:
    return JbIntegral()(x)

def jf(x: npt.ArrayLike) -> npt.ArrayLike:
    return JfIntegral()(x)

@pytest.mark.parametrize("x, fx", [
( 2.0, np.array([-1.408868, -1.330923]) ),
( np.array([2.0, 0.5]), np.array([[jb(2.0), jf(2.0)], [jb(0.5), jf(0.5)]]) ),
( np.array([[-1.0, 0.5], [7.0, 9.0]]), np.array([ [[jb(-1.0), jf(-1.0)], [jb(0.5), jf(0.5)]], [[jb(7.0), jf(7.0)], [jb(9.0), jf(9.0)]] ]) ),
( np.array([[-1.0, 11.5], [15.0, 12.0]]), np.array([ [[jb(-1.0), jf(-1.0)], [jb(11.5), jf(11.5)]], [[jb(15.0), jf(15.0)], [jb(12.0), jf(12.0)]] ]) ),
])
def test_vectorValuedInterpolation(JbJf_interpolated: InterpolatableFunction, x: npt.ArrayLike, fx: npt.ArrayLike):
    # the last one here checks that things work even when we combine interpolation and direct evaluation
    x = np.asanyarray(x)
    fx = np.asanyarray(fx)

    assert fx.shape == x.shape + (2,)
    np.testing.assert_allclose(JbJf_interpolated(x), fx, rtol=1e-6, atol=1e-8)


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

##
def test_Jb_extend_range(Jb_interpolated: JbIntegral):

    Jb = Jb_interpolated
    relativeTolerance = 1e-6

    newMin = Jb.interpolationRangeMin() - 2.
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
    fakeNewMax = newMax - 2.
    Jb.extendInterpolationTable(Jb.interpolationRangeMin(), fakeNewMax, 0, 2)

    assert Jb.interpolationRangeMax() == pytest.approx(newMax, rel=relativeTolerance)
    assert Jb(newMax) == pytest.approx(JbNewMax_direct, rel=relativeTolerance)