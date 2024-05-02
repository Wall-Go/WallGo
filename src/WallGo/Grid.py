import numpy as np

## TODO documentation about logic and limitations behind L_xi and the momentum falloff scale T


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
        \chi \equiv \frac{\xi}{\sqrt{\xi^2 + L_xi^2}}, \qquad
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

    def __init__(self, M: int, N: int, L_xi: float, momentumFalloffT: float, spacing: str="Spectral"):
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
        L_xi : float
            Length scale determining transform in :math:`\xi` direction.
        momentumFalloffT : float
            Temperature scale determining transform in momentum directions. Should be close to the plasma temperature.
        spacing : {'Spectral', 'Uniform'}
            Choose 'Spectral' for the Gauss-Lobatto collocation points, as
            required for WallGo's spectral representation, or 'Uniform' for
            a uniform grid. Default is 'Spectral'.

        """
        self.M = M
        self.N = N #This number has to be odd
        self.L_xi = L_xi
        assert spacing in ["Spectral", "Uniform"], \
            f"Unknown spacing {spacing}, not 'Spectral' or 'Uniform'"
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
                -1 + dchi, 1, num=self.M - 1, endpoint=False,
            )
            self.rzValues = np.arange(
                -1 + drz, 1, num=self.N - 1, endpoint=False,
            )
            self.rpValues = np.linspace(-1, 1, num=self.N - 1, endpoint=False)

        self._cacheCoordinates()


    def _cacheCoordinates(self) -> None:
        """Compute physical coordinates and store them internally.
        """
        (self.xiValues, self.pzValues, self.ppValues) = self.decompactify(self.chiValues, self.rzValues, self.rpValues)
    

    def changeMomentumFalloffScale(self, newScale: float) -> None:
        """"""
        self.momentumFalloffT = newScale
        self._cacheCoordinates()
        
    def changePositionFalloffScale(self, newScale: float) -> None:
        self.L_xi = newScale
        self._cacheCoordinates()


    def getCompactCoordinates(self, endpoints=False, direction=None):
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
            rz = np.array([-1] + list(self.rzValues) + [1])
            rp = np.array(list(self.rpValues) + [1])
        else:
            chi, rz, rp = self.chiValues, self.rzValues, self.rpValues
            
        if direction == 'z':
            return chi
        elif direction == 'pz':
            return rz
        elif direction == 'pp':
            return rp
        else:
            return chi, rz, rp

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

    def getCompactificationDerivatives(self, endpoints=False):
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
        xi, pz, pp = self.getCoordinates(endpoints)
        return self.compactificationDerivatives(xi, pz, pp)

    def compactify(self, z, pz, pp):
        r"""
        Transforms coordinates to [-1, 1] interval
        """
        #shouldn't you call this xi instead of z?
        z_compact = z / np.sqrt(self.L_xi**2 + z**2)
        pz_compact = np.tanh(pz / 2 / self.momentumFalloffT)
        pp_compact = 1 - 2 * np.exp(-pp / self.momentumFalloffT)
        return z_compact, pz_compact, pp_compact

    def decompactify(self, z_compact, pz_compact, pp_compact):
        r"""
        Transforms coordinates from [-1, 1] interval (inverse of compactify).
        """
        #shouldn't you call this xi instead of z?
        z = self.L_xi * z_compact / np.sqrt(1 - z_compact**2)
        pz = 2 * self.momentumFalloffT * np.arctanh(pz_compact)
        pp = -self.momentumFalloffT * np.log((1 - pp_compact) / 2)
        return z, pz, pp

    def compactificationDerivatives(self, z, pz, pp):
        r"""
        Derivative of transforms coordinates to [-1, 1] interval
        """
        #shouldn't you call this xi instead of z?
        dz_compact = self.L_xi**2 / (self.L_xi**2 + z**2)**1.5
        dpz_compact = 1 / 2 / self.momentumFalloffT / np.cosh(pz / 2 / self.momentumFalloffT)**2
        dpp_compact = 2 / self.momentumFalloffT * np.exp(-pp / self.momentumFalloffT)
        return dz_compact, dpz_compact, dpp_compact
