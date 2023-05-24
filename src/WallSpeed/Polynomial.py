#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:41:50 2023

@author: benoitlaurent
"""

import numpy as np
from Grid import Grid################
from scipy.special import eval_chebyt

class Polynomial:
    r"""
    Evaluates the cardinal basis and polynomial series.

    Parameters
    ----------
    grid : Grid
        An object of the class Grid
    
    Attributes
    ----------
    derivChi : array_like
        Derivative matrix in the chi direction
    derivRz : array_like 
        Derivative matrix in the rz direction
    """
    def __init__(self,grid):
        self.directions = {'z':0,'pz':1,'pp':2}
        
        self.grid = grid
        
        #Computing the chi and rz derivative matrices
        self.gridValues = self.grid.getCompactCoordinates(True)
        self.derivChi = self.derivatives(self.gridValues[0])
        self.derivRz = self.derivatives(self.gridValues[1])
        
    def cardinal(self,x,n,direction):
        r"""
        Computes the whole basis of cardinal functions :math:`C_n(x)` defined by grid.

        Parameters
        ----------
        x : array_like
            Coordinate at which to evaluate the cardinal function.
        n : array_like
            Order of the cardinal functions to evaluate
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'.

        Returns
        -------
        cn : array_like
            Values of the cardinal functions.
        """
        
        x = np.asarray(x)
        n = np.asarray(n)
        
        xShapeSize = len(x.shape)
        nShapeSize = len(n.shape)
        
        #Resizing the inputs in preparation for the calculation
        x = np.expand_dims(x, tuple(-np.arange(nShapeSize+1)))
        n = np.expand_dims(n, tuple(np.arange(xShapeSize+1))).astype(int)
        
        #Selecting the appropriate grid and resizing it
        grid = None
        match direction:
            case 'z': grid = self.gridValues[0]
            case 'pz': grid = self.gridValues[1]
            case 'pp': grid = self.gridValues[2]
        completeGrid = np.expand_dims(grid, tuple(np.arange(1,nShapeSize+xShapeSize+1)))
        nGrid = grid[n]
        
        #Computing all the factor in the product defining the cardinal functions
        cn_partial = np.divide(x-completeGrid, nGrid-completeGrid, where=nGrid-completeGrid!=0)
        
        #Multiplying all the factors to get the cardinal functions
        cn = np.prod(np.where(nGrid-completeGrid==0, 1, cn_partial),axis=0)
        
        return cn
    
    def chebyshev(self,x,n,restriction=None):
        r"""
        Computes the Chebyshev polynomial :math:`T_n(x)`

        Parameters
        ----------
        x : array_like
            Coordinate at which to evaluate the Chebyshev polynomial.
        n : array_like
            Order of the Chebyshev polynomial to evaluate
        restriction : None or string, optional
            Select the restriction on the Chebyshev basis. 
            If None, evaluates the unrestricted basis.
            If 'full', the polynomials are 0 at :math:`x=\pm 1`.
            If 'partial', the polynomials are 0 at :math:`x=+1`.

        Returns
        -------
        tn : array_like 
            Values of the polynomial

        """
        
        x = np.asarray(x)
        n = np.asarray(n)
        
        xShapeSize = len(x.shape)
        nShapeSize = len(n.shape)
        
        #Resizing the inputs in preparation for the calculation
        x = np.expand_dims(x, tuple(-np.arange(nShapeSize)-1))
        n = np.expand_dims(n, tuple(np.arange(xShapeSize))).astype(int)
        
        #Computing the unrestricted basis
        #tn = np.cos(n*np.arccos(x))
        tn = eval_chebyt(n, x)
        
        #Applying the restriction
        if restriction == 'partial':
            tn -= 1
        elif restriction == 'full':
            tn -= np.where(n%2==0,1,x)
        
        return tn
    
    def evaluateCardinal(self,x,f):
        """
        Evaluates the cardinal series with coefficients f at the point x.

        Parameters
        ----------
        x : array_like, shape (3,)
            Coordinate at which to evaluate the polynomial series.
        f : array_like, shape (M,N,N)
            Coefficients of the series, which are the values of the function evaluated on the grid.

        Returns
        -------
        series : float
            Value of the series at the point x.
        """
        
        #Getting the grid coordinates
        chiValues,rzValues,rpValues = self.grid.getCompactCoordinates(True)
        
        #Computing the cardinal functions for the chi, rz and rp directions
        cardinal_chi = self.cardinal(x[0], chiValues)
        cardinal_rz = self.cardinal(x[1], rzValues)
        cardinal_rp = self.cardinal(x[2], rpValues)
        
        #Summing over all the terms
        series = np.sum(f*cardinal_chi[:,None,None]*cardinal_rz[None,:,None]*cardinal_rp[None,None,:],axis=(0,1,2))
        return series
    
    def evaluateChebyshev(self,x,f):
        """
        Evaluates the Chebyshev series with coefficients f at the point x.

        Parameters
        ----------
        x : array_like, shape (3,)
            Coordinate at which to evaluate the Chebyshev series.
        f : array_like, shape (M,N,N)
            Spectral coefficients of the series.

        Returns
        -------
        series : float
            Value of the series at the point x.
        """
        
        #Computing the Chebyshev polynomials for the chi, rz and rp directions
        cheb_chi = self.chebyshev(x[0], np.arange(2,self.grid.M+1), 'full')
        cheb_rz = self.chebyshev(x[1], np.arange(2,self.grid.N+1), 'full')
        cheb_rp = self.chebyshev(x[2], np.arange(1,self.grid.N), 'partial')
        
        #Summing over all the terms
        series = np.sum(f*cheb_chi[:,None,None]*cheb_rz[None,:,None]*cheb_rp[None,None,:],axis=(0,1,2))
        return series
    
    def cardinalMatrix(self, direction, endpoints=False):
        r"""
        Returns the matrix :math:`M_{ij}=C_j(x_i)` computed in a specific direction.
        
        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """
        
        if direction == 'z':
            return np.identity(self.grid.M-1+2*endpoints)
        if direction == 'pz':
            return np.identity(self.grid.N-1+2*endpoints)
        if direction == 'pp':
            return np.identity(self.grid.N-1+endpoints)
        
    def chebyshevMatrix(self, direction, endpoints=False):
        r"""
        Returns the matrix :math:`M_{ij}=T_j(x_i)` computed in a specific direction.
        
        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """
        
        grid,n,restriction = None,None,None
        match direction:
            case 'z':
                grid = self.grid.getCompactCoordinates(endpoints)[0]
                n = np.arange(grid.size)+2-2*endpoints
                restriction = 'full'
            case 'pz':
                grid = self.grid.getCompactCoordinates(endpoints)[1]
                n = np.arange(grid.size)+2-2*endpoints
                restriction = 'full'
            case 'pp':
                grid = self.grid.getCompactCoordinates(endpoints)[2]
                n = np.arange(grid.size)+1-endpoints
                restriction = 'partial'
        if endpoints:
            restriction = None
        
        return self.chebyshev(grid, n, restriction)
        
    
    def derivatives(self,grid):
        """
        Computes the derivative matrix defined by grid.
        
        Parameters
        ----------
        grid : array_like
            Array of the grid points defining the cardinal basis.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """
        #Computing the diagonal part
        diagonal = np.sum(np.where(grid[:,None]-grid[None,:] == 0, 0, np.divide(1, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0)),axis=1)
        
        #Computing the off-diagonal part
        offDiagonal = np.prod(np.where((grid[:,None,None]-grid[None,None,:])*(grid[None,:,None]-grid[None,None,:]) == 0, 1, np.divide(grid[None,:,None]-grid[None,None,:], grid[:,None,None]-grid[None,None,:], where=grid[:,None,None]-grid[None,None,:]!=0)),axis=-1)
        
        #Putting all together
        deriv = np.where(grid[:,None]-grid[None,:] == 0,diagonal[:,None], np.divide(offDiagonal, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0))
        
        return deriv
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        