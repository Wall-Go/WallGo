import pytest
import numpy as np
import WallGo

def test_derivative():
    xs = np.linspace(-10,10,100000)
    
    f = lambda x: x*np.sin(x)
    dfdx = lambda x: np.sin(x)+x*np.cos(x)
    d2fdx2 = lambda x: 2*np.cos(x)-x*np.sin(x)
    
    np.testing.assert_allclose(dfdx(xs), WallGo.helpers.derivative(f, xs, 1, 2, (-10,10)),atol=1e-5)
    np.testing.assert_allclose(dfdx(xs), WallGo.helpers.derivative(f, xs, 1, 4, (-10,10)),atol=1e-40)
    np.testing.assert_allclose(d2fdx2(xs), WallGo.helpers.derivative(f, xs, 2, 2, (-20,20)),atol=1e-6)
    np.testing.assert_allclose(d2fdx2(xs), WallGo.helpers.derivative(f, xs, 2, 4, (-10,10)),atol=1e-7)