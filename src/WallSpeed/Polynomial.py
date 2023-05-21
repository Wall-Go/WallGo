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
        self.derivChi = self.derivatives(grid.chiValues)
        self.derivRz = self.derivatives(grid.rzValues)
        
    def cardinal(self,x,n,grid):
        r"""
        Computes the cardinal basis function :math:`C_n(x)`.

        Parameters
        ----------
        x : float
            Coordinate at which to evaluate the cardinal function.
        n : array_like
            Order of the cardinal function.
        grid : array_like
            Array of the grid points defining the cardinal basis.

        Returns
        -------
        cn : array_like
            Values of the cardinal functions.
        """
        
        #Computing all the factor in the product defining the cardinal function
        cn_partial = np.where(grid[None,:]-grid[:,None] != 0,(x-grid[:,None])/(grid[None,:]-grid[:,None]))
        
        #Multiplying all the factors to get the cardinal function
        cn = np.prod(cn_partial,axis=0)
        
        return cn
    
    def evaluate(self,x,f):
        """
        Evaluates the polynomial series with coefficients f at the points x.

        Parameters
        ----------
        x : array_like, shape (3,)
            Coordinate at which to evaluate the polynomial series.
        f : array_like, shape (M,N,N)
            Coefficients of the series, which are the values of the function evaluated on the grid.

        Returns
        -------
        series : float
            Values of the series at the points x.
        """
        
        #Computing the cardinal functions for the chi, rz and rp directions
        cardinal_chi = self.cardinal(x[0], np.arange(self.grid.M), self.grid.chiValues)
        cardinal_rz = self.cardinal(x[1], np.arange(self.grid.N), self.grid.rzValues)
        cardinal_rp = self.cardinal(x[2], np.arange(self.grid.N), self.grid.rpValues)
        
        #Summing over all the terms
        series = np.sum(f*cardinal_chi[:,None,None]*cardinal_rz[None,:,None]*cardinal_rp[None,None,:],axis=(0,1,2))
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
        diagonal = np.sum(np.where(grid[:,None]-grid[None,:] != 0, 1/(grid[:,None]-grid[None,:])),axis=1)
        
        #Computing the off-diagonal part
        offDiagonal = np.prod(np.where((grid[:,None,None]-grid[None,None,:])*(grid[None,:,None]-grid[None,None,:]) != 0, (grid[None,:,None]-grid[None,None,:])/(grid[:,None,None]-grid[None,None,:])),axis=-1)
        
        #Putting all together
        deriv = np.where(grid[:,None]-grid[None,:] == 0,diagonal[:,None],offDiagonal/(grid[:,None]-grid[None,:]))
        
        return deriv
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        