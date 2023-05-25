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
    
    cardChi = pol.cardinal(chiValues,np.arange(chiValues.size),'z')    
    cardRz = pol.cardinal(rzValues,np.arange(rzValues.size),'pz')   
    cardRp = pol.cardinal(rpValues,np.arange(rpValues.size),'pp')   
        
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
    
    # fOnGridSeries = np.zeros(fGrid.shape)
    # fOffGridSeries = np.zeros((M,N,N-1))
    
    completeGrid = np.transpose(np.meshgrid(chiValues,rzValues,rpValues,indexing='ij'), axes=(1,2,3,0))
    completeOffGrid = np.transpose(np.meshgrid((chiValues[:-1]+chiValues[1:])/2,(rzValues[:-1]+rzValues[1:])/2,(rpValues[:-1]+rpValues[1:])/2
                                                ,indexing='ij'), axes=(1,2,3,0))
    
    fOnGridSeries = pol.evaluateCardinal(completeGrid,fGrid)
    fOffGridSeries = pol.evaluateCardinal(completeOffGrid,fGrid)
    
    # for i in range(M+1):
    #     for j in range(N+1):
    #         for k in range(N):
    #             fOnGridSeries[i,j,k] = pol.evaluateCardinal([chiValues[i],rzValues[j],rpValues[k]],fGrid)
    #             if i<M and j<N and k<N-1:
    #                 fOffGridSeries[i,j,k] = pol.evaluateCardinal([(chiValues[i]+chiValues[i+1])/2,(rzValues[j]+rzValues[j+1])/2,(rpValues[k]+rpValues[k+1])/2],fGrid)
    
    maxDiffOnGrid = np.amax(np.abs(fOnGridSeries-fGrid))
    maxDiffOffGrid = np.amax(np.abs(fOffGridSeries-fOffGrid))
    
    assert maxDiffOnGrid == pytest.approx(0,abs=1e-10) and maxDiffOffGrid == pytest.approx(0,abs=1e-5)
    
def test_cardinalDeriv():
    """
    Compares the cardinal derivative matrices to finite difference derivatives.

    """
    h = 1e-6
    
    M,N = 20,21
    grid = Grid(M,N,1,1)
    pol = Polynomial(grid)
    chiValues,rzValues,rpValues = grid.getCompactCoordinates(True)
    
    finiteDerivChi = (pol.cardinal(chiValues+h,np.arange(M+1),'z')-pol.cardinal(chiValues-h,np.arange(M+1),'z'))/(2*h)
    finiteDerivRz = (pol.cardinal(rzValues+h,np.arange(N+1),'pz')-pol.cardinal(rzValues-h,np.arange(N+1),'pz'))/(2*h)
        
    maxDiffChi = np.amax(np.abs(finiteDerivChi-pol.cardinalDeriv('z',True)))
    maxDiffRz = np.amax(np.abs(finiteDerivRz-pol.cardinalDeriv('pz',True)))
    assert maxDiffChi == pytest.approx(0,abs=1e-6) and maxDiffRz == pytest.approx(0,abs=1e-6)
    
def test_chebyshevDeriv():
    """
    Compares the Chebyshev derivative matrices to finite difference derivatives.

    """
    h = 1e-6
    
    M,N = 20,21
    grid = Grid(M,N,1,1)
    pol = Polynomial(grid)
    chiValues,rzValues,rpValues = grid.getCompactCoordinates(False)
    
    finiteDerivChi = (pol.chebyshev(chiValues+h,np.arange(2,M+1),'full')-pol.chebyshev(chiValues-h,np.arange(2,M+1),'full'))/(2*h)
    finiteDerivRz = (pol.chebyshev(rzValues+h,np.arange(2,N+1),'full')-pol.chebyshev(rzValues-h,np.arange(2,N+1),'full'))/(2*h)
        
    maxDiffChi = np.amax(np.abs(finiteDerivChi-pol.chebyshevDeriv('z',False)))
    maxDiffRz = np.amax(np.abs(finiteDerivRz-pol.chebyshevDeriv('pz',False)))
    assert maxDiffChi == pytest.approx(0,abs=1e-6) and maxDiffRz == pytest.approx(0,abs=1e-6)
    




































    