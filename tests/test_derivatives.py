import pytest
import numpy as np
import WallGo

def test_derivative():
    dx = 1e-4
    xs = np.linspace(-10,10,100000)
    
    f = lambda x: x*np.sin(x)
    dfdx = lambda x: np.sin(x)+x*np.cos(x)
    d2fdx2 = lambda x: 2*np.cos(x)-x*np.sin(x)
    
    # np.testing.assert_allclose(dfdx(xs), WallGo.helpers.derivative(f, xs, dx, 1, 2, (-10,10)),atol=1e-2)
    # np.testing.assert_allclose(dfdx(xs), WallGo.helpers.derivative(f, xs, dx, 1, 4, (-10,10)),atol=1e-5)
    # np.testing.assert_allclose(d2fdx2(xs), WallGo.helpers.derivative(f, xs, dx, 2, 2, (-10,10)),atol=0.1)
    np.testing.assert_allclose(d2fdx2(xs), WallGo.helpers.derivative(f, xs, dx, 2, 4, (-10,10)),atol=1e-10)