import numpy as np
import pytest
from WallGo.grid import Grid
from WallGo.polynomial import Polynomial

grid = Grid(4,4,1,1)

# Test on the polynomial 1/2+x-x^2/2-x^3

def test_evaluate():
    polyCard = Polynomial([(1-np.sqrt(2))/4, 0.5, (1+np.sqrt(2))/4],grid,'Cardinal','z',False)
    polyCheb = Polynomial([-0.25,-0.25,0],grid,'Chebyshev','z',False)
    x = [[-1,-0.3,0.2,0.7,1]]
    np.testing.assert_allclose(polyCard.evaluate(x),[0., 0.182, 0.672, 0.612, 0.],rtol=1e-15,atol=1e-15)
    np.testing.assert_allclose(polyCheb.evaluate(x),[0., 0.182, 0.672, 0.612, 0.],rtol=1e-15,atol=1e-15)
    
def test_changeBasis():
    polyCard = Polynomial([(1-np.sqrt(2))/4, 0.5, (1+np.sqrt(2))/4],grid,'Cardinal','z',False)
    polyCheb = Polynomial([-0.25,-0.25,0],grid,'Chebyshev','z',False)
    polyCard.changeBasis('Chebyshev')
    polyCheb.changeBasis('Cardinal')
    np.testing.assert_allclose(polyCard.coefficients, [-0.25,-0.25,0],rtol=1e-15,atol=1e-15)
    np.testing.assert_allclose(polyCheb.coefficients, [(1-np.sqrt(2))/4, 0.5, (1+np.sqrt(2))/4],rtol=1e-15,atol=1e-15)
    
def test_deriv():
    polyCard = Polynomial([(1-np.sqrt(2))/4, 0.5, (1+np.sqrt(2))/4],grid,'Cardinal','z',False)
    polyCheb = Polynomial([-0.25,-0.25,0],grid,'Chebyshev','z',False)
    
    derivCard = polyCard.derivative(axis=0)
    derivCheb = polyCheb.derivative(axis=0)
    
    np.testing.assert_allclose(derivCard.coefficients, [-1., -0.5+1/np.sqrt(2), 1., -0.5-1/np.sqrt(2), -3.],rtol=1e-15,atol=1e-15)
    np.testing.assert_allclose(derivCheb.coefficients, [-1., -0.5+1/np.sqrt(2), 1., -0.5-1/np.sqrt(2), -3.],rtol=1e-15,atol=1e-15)
    
    derivCheb.changeBasis('Chebyshev')
    np.testing.assert_allclose(derivCheb.coefficients, [-0.5,-1,-1.5,0,0],rtol=1e-15,atol=1e-15)
    
def test_integrate():
    polyCard = Polynomial([(1-np.sqrt(2))/4, 0.5, (1+np.sqrt(2))/4],grid,'Cardinal','z',False)
    polyCheb = Polynomial([-0.25,-0.25,0],grid,'Chebyshev','z',False)
    
    assert np.isclose(polyCard.integrate(weight=1/np.sqrt(1-grid.chiValues**2)), np.pi/4,rtol=1e-15,atol=1e-15)
    assert np.isclose(polyCheb.integrate(weight=1/np.sqrt(1-grid.chiValues**2)), np.pi/4,rtol=1e-15,atol=1e-15)
