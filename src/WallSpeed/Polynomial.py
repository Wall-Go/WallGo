#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:41:50 2023

@author: benoitlaurent
"""

import numpy as np
from .Grid import Grid

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
        self.grid = grid
        
        #Computing the chi and rz derivative matrices
        chiValues,rzValues,rpValues = self.grid.getCompactCoordinates(True)
        self.derivChi = self.derivatives(chiValues)
        self.derivRz = self.derivatives(rzValues)
        
    def cardinal(self,x,grid):
        r"""
        Computes the whole basis of cardinal functions :math:`C_n(x)` defined by grid.

        Parameters
        ----------
        x : float
            Coordinate at which to evaluate the cardinal function.
        grid : array_like
            Array of the grid points defining the cardinal basis.

        Returns
        -------
        cn : array_like
            Values of the cardinal functions.
        """
        
        #Computing all the factor in the product defining the cardinal functions
        cn_partial = np.divide(x-grid[:,None], grid[None,:]-grid[:,None], where=grid[None,:]-grid[:,None]!=0)
        
        #Multiplying all the factors to get the cardinal functions
        cn = np.prod(np.where(grid[None,:]-grid[:,None] == 0, 1, cn_partial),axis=0)
        
        return cn
    
    def chebyshev(self,x,n,restriction=None):
        r"""
        Computes the Chebyshev polynomial :math:`T_n(x)`

        Parameters
        ----------
        x : float
            Coordinate at which to evaluate the polynomial.
        n : int or array_like 
            Order of the Chebyshev polynomial.
        restriction : None or string, optional
            Select the restriction on the Chebyshev basis. 
            If None, evaluates the unrestricted basis.
            If 'full', the polynomials are 0 at :math:`x=\pm 1`.
            If 'partial', the polynomials are 0 at :math:`x=+1`.

        Returns
        -------
        tn : float or array_like 
            Values of the polynomial

        """
        
        #Computing the unrestricted basis
        tn = np.cos(n*np.arccos(x))
        
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        