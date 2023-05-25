#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class Grid:
    r"""
    Computes the grid on which the Boltzmann equation is solved.

    Grid is 3d, and consists of the physical coordinates:

        - :math:`\xi`, position perpendicular to the wall,
        - :math:`p_z`, momentum perpendicular to the wall,
        - :math:`p_\Vert`, momentum magnitude parallel to the wall.

    In addition there are the corresponding compactified coordinates on the
    interval [-1, 1],

    .. math::
        \chi \equiv \frac{\xi}{\sqrt{\xi^2 + L^2}}, \qquad
        \rho_{z} \equiv \tanh\left(\frac{p_z}{2 T_0}\right), \qquad
        \rho_{\Vert} \equiv 1 - 2 e^{-p_\Vert/T_0}.

    All coordinates are in the wall frame.

    Attributes
    ----------
    chiValues : array_like
        Grid of the :math:`\chi` direction.
    rzValues : array_like
        Grid of the :math:`\rho_z` direction.
    rpValues : array_like
        Grid of the :math:`\rho_\Vert` direction.
    xiValues : array_like
        Grid of the :math:`\xi` direction.
    pzValues : array_like
        Grid of the :math:`p_z` direction.
    ppValues : array_like
        Grid of the :math:`p_\Vert` direction.
    """

    def __init__(self, M, N, L_xi, T):
        r"""
        Initialises Grid object.

        Parameters
        ----------
        M : int
            Number of basis functions in the :math:`\xi` (and :math:`\chi`)
            direction.
        N : int
            Number of basis functions in the :math:`p_z` and :math:`p_\Vert`
            (and :math:`\rho_z` and :math:`\rho_\Vert`) directions.
        L_xi : float
            Length scale determining transform in :math:`\xi` direction.
        T : float
            Temperature scale determining transform in momentum directions.

        """
        self.M = M
        self.N = N
        self.L_xi = L_xi
        self.T = T

        # Computing the grids in the chi, rz and rp directions
        # See equation (34) in arXiv:2204.13120.
        # Additional signs are so that each coordinate starts from -1.
        self.chiValues = -np.cos(np.arange(1, self.M) * np.pi / self.M)
        self.rzValues = -np.cos(np.arange(1, self.N) * np.pi / self.N)
        self.rpValues = -np.cos(np.arange(0, self.N - 1) * np.pi / (self.N - 1))

        # Computing the grids in physical coordinates
        (self.xiValues, self.pzValues, self.ppValues,) = Grid.decompactify(
            self.chiValues, self.rzValues, self.rpValues, L_xi, T
        )

    def getCompactCoordinates(self, endpoints=False):
        r"""
        Return compact coordinates of grid.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        ----------
        chiValues : array_like
            Grid of the :math:`\chi` direction.
        rzValues : array_like
            Grid of the :math:`\rho_z` direction.
        rpValues : array_like
            Grid of the :math:`\rho_\Vert` direction.
        """
        if endpoints:
            chi = np.array([-1] + list(self.chiValues) + [1])
            rz = np.array([-1] + list(self.rzValues) + [1])
            rp = np.array(list(self.rpValues) + [1])
            return chi, rz, rp
        else:
            return self.chiValues, self.rzValues, self.rpValues

    def getCoordinates(self, endpoints=False):
        r"""
        Return coordinates of grid, not compactified.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        ----------
        xiValues : array_like
            Grid of the :math:`\xi` direction.
        pzValues : array_like
            Grid of the :math:`p_z` direction.
        ppValues : array_like
            Grid of the :math:`p_\Vert` direction.
        """
        if endpoints:
            xi = np.array([-np.inf] + list(self.xiValues) + [np.inf])
            pz = np.array([-np.inf] + list(self.pzValues) + [np.inf])
            pp = np.array(list(self.ppValues) + [np.inf])
            return xi, pz, pp
        else:
            return self.xiValues, self.pzValues, self.ppValues

    def compactify(z, pz, pp, L, T0):
        r"""
        Transforms coordinates to [-1, 1] interval
        """
        z_compact = z / np.sqrt(L**2 + z**2)
        pz_compact = np.tanh(pz / 2 / T0)
        pp_compact = 1 - 2 * np.exp(-pp / T0)
        return z_compact, pz_compact, pp_compact

    def decompactify(z_compact, pz_compact, pp_compact, L, T0):
        r"""
        Transforms coordinates from [-1, 1] interval (inverse of compactify).
        """
        z = L * z_compact / np.sqrt(1 - z_compact**2)
        pz = 2 * T0 * np.arctanh(pz_compact)
        pp = -T0 * np.log((1 - pp_compact) / 2)
        return z, pz, pp
