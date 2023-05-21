#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:02:29 2023

@author: benoitlaurent
"""

import numpy as np
from .coordinates import ...

class Grid:
    """
    Computes the grid on which the Boltzmann equation is solved.
    
    Parameters
    ----------
    M : int
        Number of basis functions in the chi direction.
    N : int
        Number of basis functions in the rhoz and rhoPar directions.
        
    Attributes
    ----------
    chiValues : array_like
        Grid of the chi direction.
    rzValues : array_like
        Grid of the rz direction.
    rpValues : array_like
        Grid of the rp direction.
    xiValues : array_like
        Grid of the xi direction.
    pzValues : array_like
        Grid of the pz direction.
    ppValues : array_like
        Grid of the pp direction.
    """
    def __init__(self,M,N,L_xi,T):
        self.M = M
        self.N = N
        self.L_xi = L_xi
        self.T = T
        
        #Computing the grids in the chi, rz and rp directions
        self.chiValues = -np.cos(np.arange(1,self.M)*np.pi/self.M)
        self.rzValues = -np.cos(np.arange(1,self.N)*np.pi/self.N)
        self.rpValues = np.flip(np.cos(np.arange(1,self.N)*np.pi/(self.N-1)))
        
        #Computing the grids in physical coordinates
        self.xiValues,self.pzValues,self.ppValues = decompactifyCoordinates(self.chiValues, self.rzValues, self.rpValues, L_xi, T)
        
    