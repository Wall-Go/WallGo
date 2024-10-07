"""
Class for computing and storing the coordinates on the grid and other related 
quantities.
"""

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
        \chi \equiv \frac{\xi}{\sqrt{\xi^2 + L_{\xi}^2}}, \qquad
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

    def __init__(
        self,
        M: int, # pylint: disable=invalid-name
        N: int, # pylint: disable=invalid-name
        positionFalloff: float,
        momentumFalloffT: float,
        spacing: str = "Spectral",
    ):
        r"""
        Initialises Grid object.

        Compactified coordinates are chosen according to

        .. math::
            \chi = -\cos\left(\frac{\pi i}{M}\right), \qquad
            \rho_{z} = -\cos\left(\frac{\pi j}{N}\right), \qquad
            \rho_{\Vert} = -\cos\left(\frac{\pi k}{N-1}\right),

        with integers :math:`i, j, k` taken over

        .. math::
            i = 0, 1, \dots, M, \qquad
            j = 0, 1, \dots, N, \qquad
            k = 0, 1, \dots, N-1.

        These are the Gauss-Lobatto collocation points, here with all
        boundary points included.

        The boundary points :math:`\chi=\pm 1`, :math:`\rho_z=\pm 1` and
        :math:`\rho_{\Vert}=1` correspond to points at infinity. The
        deviation from equilibrium is assumed to equal zero at infinity, so
        these points are dropped when solving the Boltzmann equations. The
        resulting grid is

        .. math::
            i = 1, 2, \dots, M-1, \qquad
            j = 1, 2, \dots, N-1, \qquad
            k = 0, 1, \dots, N-2.


        Parameters
        ----------
        M : int
            Number of basis functions in the :math:`\xi` (and :math:`\chi`)
            direction.
        N : int
            Number of basis functions in the :math:`p_z` and :math:`p_\Vert`
            (and :math:`\rho_z` and :math:`\rho_\Vert`) directions.
        positionFalloff : float
            Length scale determining transform in :math:`\xi` direction. Should be
            expressed in physical units (the units used in EffectivePotential).
        momentumFalloffT : float
            Temperature scale determining transform in momentum directions. 
            Should be close to the plasma temperature.
        spacing : {'Spectral', 'Uniform'}
            Choose 'Spectral' for the Gauss-Lobatto collocation points, as
            required for WallGo's spectral representation, or 'Uniform' for
            a uniform grid. Default is 'Spectral'.

        """
        self.M = M # pylint: disable=invalid-name
        # This number has to be odd
        self.N = N # pylint: disable=invalid-name
        self.positionFalloff = positionFalloff
        assert spacing in [
            "Spectral",
            "Uniform",
        ], f"Unknown spacing {spacing}, not 'Spectral' or 'Uniform'"
        self.spacing = spacing
        self.momentumFalloffT = momentumFalloffT

        # Computing the grids in the chi, rz and rp directions
        if self.spacing == "Spectral":
            # See equation (34) in arXiv:2204.13120.
            # Additional signs are so that each coordinate starts from -1.
            self.chiValues = -np.cos(np.arange(1, self.M) * np.pi / self.M)
            self.rzValues = -np.cos(np.arange(1, self.N) * np.pi / self.N)
            self.rpValues = -np.cos(np.arange(0, self.N - 1) * np.pi / (self.N - 1))
        elif self.spacing == "Uniform":
            dchi = 2 / self.M
            drz = 2 / self.N
            self.chiValues = np.linspace(
                -1 + dchi,
                1,
                num=self.M - 1,
                endpoint=False,
            )
            self.rzValues = np.linspace(
                -1.0 + drz,
                1.0,
                num=self.N - 1,
                endpoint=False,
            )
            self.rpValues = np.linspace(-1, 1, num=self.N - 1, endpoint=False)

        self._cacheCoordinates()

    def _cacheCoordinates(self) -> None:
        """
        Compute physical coordinates and store them internally.
        """
        (self.xiValues, self.pzValues, self.ppValues) = self.decompactify(
            self.chiValues, self.rzValues, self.rpValues
        )
        (self.dxidchi, self.dpzdrz, self.dppdrp) = self.compactificationDerivatives(
            self.chiValues, self.rzValues, self.rpValues
        )

    def changeMomentumFalloffScale(self, newScale: float) -> None:
        """
        Change the momentum falloff scale.

        Parameters
        ----------
        newScale : float
            New momentum falloff scale.
        """
        self.momentumFalloffT = newScale
        self._cacheCoordinates()

    def changePositionFalloffScale(self, newScale: float) -> None:
        """
        Change the position falloff scale.

        Parameters
        ----------
        newScale : float
            New position falloff scale.
        """
        self.positionFalloff = newScale
        self._cacheCoordinates()

    def getCompactCoordinates(
            self,
            endpoints: bool=False,
            direction: str | None=None,
            ) -> tuple[np.ndarray, ...] | np.ndarray:
        r"""
        Return compact coordinates of grid.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.
        direction : string or None, optional
            Specifies which coordinates to return. Can either be 'z', 'pz',
            'pp' or None. If None, returns a tuple containing the 3 directions.
            Default is None.

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
            rz = np.array([-1]+list(self.rzValues)+[1]) # pylint: disable=invalid-name
            rp = np.array(list(self.rpValues) + [1]) # pylint: disable=invalid-name
        else:
            chi, rz, rp = ( # pylint: disable=invalid-name
                self.chiValues, self.rzValues, self.rpValues)

        if direction == "z":
            return chi
        if direction == "pz":
            return rz
        if direction == "pp":
            return rp
        return chi, rz, rp

    def getCoordinates(self, endpoints: bool=False) -> tuple[np.ndarray, ...]:
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
            xi = np.array( # pylint: disable=invalid-name
                          [-np.inf] + list(self.xiValues) + [np.inf])
            pz = np.array( # pylint: disable=invalid-name
                          [-np.inf] + list(self.pzValues) + [np.inf])
            pp = np.array( # pylint: disable=invalid-name
                          list(self.ppValues) + [np.inf])
            return xi, pz, pp
        return self.xiValues, self.pzValues, self.ppValues

    def getCompactificationDerivatives(
            self,
            endpoints: bool=False,
            ) -> tuple[np.ndarray, ...]:
        r"""
        Return derivatives of compactified coordinates of grid, with respect to
        uncompactified derivatives.

        Parameters
        ----------
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        ----------
        dchiValues : array_like
            Grid of the :math:`\partial_\xi\chi` direction.
        drzValues : array_like
            Grid of the :math:`\partial_{p_z}\rho_z` direction.
        drpValues : array_like
            Grid of the :math:`\partial_{p_\Vert}\rho_\Vert` direction.
        """
        if endpoints:
            dxidchi = np.array([np.inf] + list(self.dxidchi) + [np.inf])
            dpzdrz = np.array([np.inf] + list(self.dpzdrz) + [np.inf])
            dppdrp = np.array(list(self.dppdrp) + [np.inf])
            return dxidchi, dpzdrz, dppdrp
        return self.dxidchi, self.dpzdrz, self.dppdrp

    def compactify(
            self,
            z: np.ndarray, # pylint: disable=invalid-name
            pz: np.ndarray, # pylint: disable=invalid-name
            pp: np.ndarray, # pylint: disable=invalid-name
            ) -> tuple[np.ndarray, ...]:
        """
        Transforms coordinates to [-1, 1] interval

        Parameters
        ----------
        z : array-like
            Physical z (or xi) coordinate.
        pz : array-like
            Physical pz coordinate.
        pp : array-like
            Physical p_par coordinate.

        Returns
        -------
        z_compact : array-like
            Compact z coordinate (chi).
        pz_compact : array-like
            Compact pz coordinate (rho_z).
        pp_compact : array-like
            Compact p_par coordinate (rho_par).

        """

        zCompact = z / np.sqrt(self.positionFalloff**2 + z**2)
        pzCompact = np.tanh(pz / 2 / self.momentumFalloffT)
        ppCompact = 1 - 2 * np.exp(-pp / self.momentumFalloffT)
        return zCompact, pzCompact, ppCompact

    def decompactify(
            self,
            zCompact: np.ndarray,
            pzCompact: np.ndarray,
            ppCompact: np.ndarray,
            ) -> tuple[np.ndarray, ...]:
        """
        Transforms coordinates to [-1, 1] interval

        Parameters
        ----------
        z_compact : array-like
            Compact z coordinate (chi).
        pz_compact : array-like
            Compact pz coordinate (rho_z).
        pp_compact : array-like
            Compact p_par coordinate (rho_par).

        Returns
        -------
        z : array-like
            Physical z (or xi) coordinate.
        pz : array-like
            Physical pz coordinate.
        pp : array-like
            Physical p_par coordinate.

        """

        z = ( # pylint: disable=invalid-name
            self.positionFalloff * zCompact / np.sqrt(1 - zCompact**2))
        pz = ( # pylint: disable=invalid-name
            2 * self.momentumFalloffT * np.arctanh(pzCompact))
        pp = ( # pylint: disable=invalid-name
              -self.momentumFalloffT * np.log((1 - ppCompact) / 2))
        return z, pz, pp

    def compactificationDerivatives(
            self,
            zCompact: np.ndarray,
            pzCompact: np.ndarray,
            ppCompact: np.ndarray,
            ) -> tuple[np.ndarray, ...]:
        r"""
        Derivative :math:`d(X)/d(X_\text{compact})` of coordinate transforms to [-1, 1] interval.

        Parameters
        ----------
        z_compact : array-like
            Compact z coordinate (chi).
        pz_compact : array-like
            Compact pz coordinate (rho_z).
        pp_compact : array-like
            Compact p_par coordinate (rho_par).

        Returns
        -------
        dzdzCompact : array-like
            Derivative d(z)/d(chi).
        dpzdpzCompact : array-like
            PDerivative d(p_z)/d(rho_z).
        dppdppCompact : array-like
            Derivative d(p_par)/d(rho_par).
        """

        dzdzCompact = self.positionFalloff / (1 - zCompact**2) ** 1.5
        dpzdpzCompact = 2 * self.momentumFalloffT / (1 - pzCompact**2)
        dppdppCompact = self.momentumFalloffT / (1 - ppCompact)
        return dzdzCompact, dpzdpzCompact, dppdppCompact
