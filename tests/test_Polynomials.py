import numpy as np
import pytest
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Grid import Grid

def test_cardinal():
    r"""
    Tests if the cardinal basis evaluated on the grid returns a Kronecker delta

    """
    
    grid = Grid(16,21,1,1)
    chiValues,rzValues,rpValues = grid.getCompactCoordinates(True)
    
    pol = Polynomial(grid)
    
    cardChi,cardRz,cardRp = np.zeros((chiValues.size,chiValues.size)),np.zeros((rzValues.size,rzValues.size)),np.zeros((rpValues.size,rpValues.size))
    
    for i,chi in enumerate(chiValues):
        cardChi[i] = pol.cardinal(chi,chiValues)
    for i,rz in enumerate(rzValues):
        cardRz[i] = pol.cardinal(rz,rzValues)
    for i,rp in enumerate(rpValues):
        cardRp[i] = pol.cardinal(rp,rpValues)
        
    maxDiffChi = np.amax(np.abs(cardChi-np.identity(chiValues.size)))
    maxDiffRz = np.amax(np.abs(cardRz-np.identity(rzValues.size)))
    maxDiffRp = np.amax(np.abs(cardRp-np.identity(rpValues.size)))
    
    assert maxDiffChi == pytest.approx(0,abs=1e-10) and maxDiffRz == pytest.approx(0,abs=1e-10) and maxDiffRp == pytest.approx(0,abs=1e-10)
    
def test_evaluateCardinal():
    r"""
    Interpolates a function f by a cardinal series and tests if this is a good approximation on and between the grid points.

    """
    
    M,N = 20,20
    grid = Grid(M,N,1,1)
    #grid.chiValues = np.linspace(-1,1,M+1)[1:-1]
    #grid.rzValues = np.linspace(-1,1,N+1)[1:-1]
    #grid.rpValues = np.linspace(-1,1,N)[:-1]
    pol = Polynomial(grid)
    chiValues,rzValues,rpValues = grid.getCompactCoordinates(True)
    
    f = lambda x,y,z: np.exp(-(2*x)**2-(3*y)**2-(1.5*z)**2)*(1+x+y**2+0.5*z)
    
    fGrid = f(chiValues[:,None,None],rzValues[None,:,None],rpValues[None,None,:])
    fOffGrid = f((chiValues[:-1,None,None]+chiValues[1:,None,None])/2,(rzValues[None,:-1,None]+rzValues[None,1:,None])/2,(rpValues[None,None,:-1]+rpValues[None,None,1:])/2)
    
    fOnGridSeries = np.zeros(fGrid.shape)
    fOffGridSeries = np.zeros((M,N,N-1))
    
    for i in range(M+1):
        for j in range(N+1):
            for k in range(N):
                fOnGridSeries[i,j,k] = pol.evaluateCardinal([chiValues[i],rzValues[j],rpValues[k]],fGrid)
                if i<M and j<N and k<N-1:
                    fOffGridSeries[i,j,k] = pol.evaluateCardinal([(chiValues[i]+chiValues[i+1])/2,(rzValues[j]+rzValues[j+1])/2,(rpValues[k]+rpValues[k+1])/2],fGrid)
    
    maxDiffOnGrid = np.amax(np.abs(fOnGridSeries-fGrid))
    maxDiffOffGrid = np.amax(np.abs(fOffGridSeries-fOffGrid))
    
    assert maxDiffOnGrid == pytest.approx(0,abs=1e-10) and maxDiffOffGrid == pytest.approx(0,abs=1e-5)
    
def test_derivatives():
    """
    Compares the derivative matrices to finite difference derivatives.

    """
    h = 1e-6
    
    M,N = 20,21
    grid = Grid(M,N,1,1)
    pol = Polynomial(grid)
    chiValues,rzValues,rpValues = grid.getCompactCoordinates(True)
    
    finiteDerivChi = np.zeros(pol.derivChi.shape)
    finiteDerivRz = np.zeros(pol.derivRz.shape)
    
    for i in range(M+1):
        finiteDerivChi[i] = (pol.cardinal(chiValues[i]+h,chiValues)-pol.cardinal(chiValues[i]-h,chiValues))/(2*h)
    for i in range(N+1):
        finiteDerivRz[i] = (pol.cardinal(rzValues[i]+h,rzValues)-pol.cardinal(rzValues[i]-h,rzValues))/(2*h)
        
    maxDiffChi = np.amax(np.abs(finiteDerivChi-np.transpose(pol.derivChi)))
    maxDiffRz = np.amax(np.abs(finiteDerivRz-np.transpose(pol.derivRz)))
    assert maxDiffChi == pytest.approx(0,abs=1e-6) and maxDiffRz == pytest.approx(0,abs=1e-6)
    




































    