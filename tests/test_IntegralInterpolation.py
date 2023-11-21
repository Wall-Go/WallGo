import numpy as np
import pytest

import WallSpeed.Integrals

### Test real parts of Jb, Jf integrals

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.054003365177269e-10),
    (-20.5, 8.964742570241336)
])
def test_directJb(x: float, expectedResult: float):

    Jb = WallSpeed.Integrals.JbIntegral(bUseAdaptiveInterpolation=False)
    assert Jb(x) == pytest.approx(expectedResult, abs=1e-9)
    

@pytest.mark.parametrize("x, expectedResult", [
    (800, -1.0557025829810107e-10),
    (-20.5, 11.9135695475507)
])
def test_directJf(x: float, expectedResult: float):

    Jf = WallSpeed.Integrals.JfIntegral(bUseAdaptiveInterpolation=False)
    assert Jf(x) == pytest.approx(expectedResult, abs=1e-9)
    