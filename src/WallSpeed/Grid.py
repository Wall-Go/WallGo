#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:02:29 2023

@author: benoitlaurent
"""

import numpy as np
from .coordinates import decompactifyCoordinates

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
        #See equation (34) in arXiv:2204.13120.
        #Additional signs are so that each coordinate starts from -1.
        self.chiValues = -np.cos(np.arange(1,self.M)*np.pi/self.M)
        self.rzValues = -np.cos(np.arange(1,self.N)*np.pi/self.N)
        self.rpValues = np.flip(np.cos(np.arange(1,self.N)*np.pi/(self.N-1)))

        #Computing the grids in physical coordinates
        self.xiValues,self.pzValues,self.ppValues = decompactifyCoordinates(self.chiValues, self.rzValues, self.rpValues, L_xi, T)

    def getCompactCoordinates(self, endpoints=False):
        """
        Return compact coordinates of grid.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        ----------
        chiValues : array_like
            Grid of the chi direction.
        rzValues : array_like
            Grid of the rz direction.
        rpValues : array_like
            Grid of the rp direction.
        """
        if endpoints:
            chi = np.array([-1] + list(self.chiValues) + [1])
            rz = np.array([-1] + list(self.rzValues) + [1])
            rp = np.array(list(self.rpValues) + [1])
            return chi, rz, rp
        else:
            return self.chiValues, self.rzValues, self.rpValues

    def getCoordinates(self, endpoints=False):
        """
        Return coordinates of grid, not compactified.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        ----------
        xiValues : array_like
            Grid of the chi direction.
        pzValues : array_like
            Grid of the rz direction.
        ppValues : array_like
            Grid of the rp direction.
        """
        if endpoints:
            xi = np.array([-np.inf] + list(self.chiValues) + [np.inf])
            pz = np.array([-np.inf] + list(self.rzValues) + [np.inf])
            pp = np.array(list(self.rpValues) + [np.inf])
            return xi, pz, pp
        else:
            return self.xiValues, self.pzValues, self.ppValues
