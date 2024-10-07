"""
Class for computing and storing the coordinates on the grid and other related 
quantities.
"""

import numpy as np
from .grid import Grid


class Grid3Scales(Grid):
    r"""
    Redefinition of the Grid class to take into account the different scales present in
    the z direction. More specifically, the :math:`z` mapping function should scale as

    .. math::
        z(\chi \to -1) &= \lambda_- \log(1+\chi),\\
        z(\chi \to 1) &= -\lambda_+ \log(1-\chi),
    
    where :math:`\lambda_-` and :math:`\lambda_+` are the lengths of the solution's
    tails inside and outside the bubble, respectively. Furthermore, the mapping
    should be approximately linear in the region :math:`-r<\chi<r`, where :math:`r`
    is roughly the ratio of points that are used to resolve the wall's interior.
    The slope in that region should be :math:`L/r`, where :math:`L` is the wall
    thickness, so that

    .. math::
        z'(\chi) \approx \frac{L}{r}\chi, \quad \chi \in [-r, r].

    It is easier to find the derivative of a function that has these properties,
    and then integrate it. We choose here
    
    .. math::
        z'(\chi)=\frac{f(\chi)}{1-\chi^2},

    where :math:`f(\chi)` is a smoothed step function equal to
    
    .. math::
        f(\chi) \approx \begin{cases}
            \lambda_-,& \chi<-r,\\
            L/r,& -r<\chi<r,\\
            \lambda_+,& \chi>r.
        \end{cases}
    
    We choose :math:`f(\chi)` to be a sum of functions like
    :math:`\frac{\chi-\chi_0}{\sqrt{a^2+(\chi-\chi_0)^2}}`, which allows us to find
    analytically the mapping function with
    
    .. math::
        z(\chi) - z(\chi_0) =\int^\chi_{\chi_0} \frac{f(\chi')}{1-\chi'^2}  d\chi'.

    The parameter :math:`a` can be adjusted to control the smoothness of the mapping
    function.
    """

    def __init__(
        self,
        M: int,
        N: int,
        tailLengthInside: float,
        tailLengthOutside: float,
        wallThickness: float,
        momentumFalloffT: float,
        ratioPointsWall: float = 0.5,
        smoothing: float = 0.1,
        wallCenter: float = 0,
        spacing: str = "Spectral",
    ):
        r"""


        Parameters
        ----------
        M : int
            Number of basis functions in the :math:`\xi` (and :math:`\chi`)
            direction.
        N : int
            Number of basis functions in the :math:`p_z` and :math:`p_\Vert`
            (and :math:`\rho_z` and :math:`\rho_\Vert`) directions.
        tailLengthInside : float
            Decay length of the solution's tail inside the wall. Should be larger
            than wallThickness*(1+2*smoothing)/ratioPointsWall. Should be
            expressed in physical units (the units used in EffectivePotential).
        tailLengthOutside : float
            Decay length of the solution's tail outside the wall. Should be larger
            than wallThickness*(1+2*smoothing)/ratioPointsWall. Should be
            expressed in physical units (the units used in EffectivePotential).
        wallThickness : float
            Thickness of the wall. Should be expressed in physical units
            (the units used in EffectivePotential).
        momentumFalloffT : float
            Temperature scale determining transform in momentum directions. 
            Should be close to the plasma temperature.
        ratioPointsWall : float, optional
            Ratio of grid points inside the wall. The remaining points are
            distributed equally between the 2 tails. The default is 0.5.
        smoothing : float, optional
            Controls the smoothness of the mapping function. Its first derivative
            becomes discontinuous at :math:`\chi=\pm r` when smoothness is 0.
            Should be smaller than 1, otherwise the function would not be linear
            at :math:`\chi=0` anymore. As explained above, the decay length is
            controlled by adding 2 smoothed step functions. 'smoothing' is the
            value of these functions at the origin, in units of :math:`L/r`.
            The default is 0.1.
        wallCenter : float, optional
            Position of the wall's center (in the z coordinates). Default is 0.
        spacing : {'Spectral', 'Uniform'}
            Choose 'Spectral' for the Gauss-Lobatto collocation points, as
            required for WallGo's spectral representation, or 'Uniform' for
            a uniform grid. Default is 'Spectral'.

        Returns
        -------
        None.

        """
        self._updateParameters(
            tailLengthInside,
            tailLengthOutside,
            wallThickness,
            ratioPointsWall,
            smoothing,
            wallCenter,
        )

        super().__init__(M, N, wallThickness, momentumFalloffT, spacing)

    def changePositionFalloffScale(
        self,
        tailLengthInside: float,
        tailLengthOutside: float,
        wallThickness: float,
        wallCenter: float,
    ) -> None:
        self._updateParameters(
            tailLengthInside,
            tailLengthOutside,
            wallThickness,
            self.ratioPointsWall,
            self.smoothing,
            wallCenter,
        )

        self._cacheCoordinates()

    def _updateParameters(
        self,
        tailLengthInside: float,
        tailLengthOutside: float,
        wallThickness: float,
        ratioPointsWall: float,
        smoothing: float,
        wallCenter: float,
    ) -> None:
        assert wallThickness > 0, "Grid3Scales error: wallThickness must be positive."
        assert smoothing > 0, "Grid3Scales error: smoothness must be positive."
        assert (
            tailLengthInside > wallThickness * (1 + 2 * smoothing) / ratioPointsWall
        ), """Grid3Scales error: tailLengthInside must be greater than
        wallThickness*(1+2*smoothness)/ratioPointsWall."""
        assert (
            tailLengthOutside > wallThickness * (1 + 2 * smoothing) / ratioPointsWall
        ), """Grid3Scales error: tailLengthOutside must be greater than
        wallThickness*(1+2*smoothness)/ratioPointsWall."""
        assert (
            0 < ratioPointsWall < 1
        ), "Grid3Scales error: ratioPointsWall must be between 0 and 1."

        self.tailLengthInside = tailLengthInside
        self.tailLengthOutside = tailLengthOutside
        self.wallThickness = wallThickness
        self.ratioPointsWall = ratioPointsWall
        self.smoothing = smoothing
        self.wallCenter = wallCenter

        # Defining parameters used in the mapping functions.
        # These are set to insure that the smoothed step functions used to get
        # the right decay length have a value of smoothing*L/ratioPointsWall
        # at the origin.
        self.aIn = np.sqrt(
            4
            * smoothing
            * wallThickness
            * ratioPointsWall**2
            * (ratioPointsWall * tailLengthInside - wallThickness * (1 + smoothing))
        ) / abs(
            ratioPointsWall * tailLengthInside - wallThickness * (1 + 2 * smoothing)
        )
        self.aOut = np.sqrt(
            4
            * smoothing
            * wallThickness
            * ratioPointsWall**2
            * (ratioPointsWall * tailLengthOutside - wallThickness * (1 + smoothing))
        ) / abs(
            ratioPointsWall * tailLengthOutside - wallThickness * (1 + 2 * smoothing)
        )

    def decompactify(
            self,
            zCompact: np.ndarray,
            pzCompact: np.ndarray,
            ppCompact: np.ndarray,
            ) -> tuple[np.ndarray, ...]:
        r"""
        Transforms coordinates from [-1, 1] interval (inverse of compactify).
        """
        L = self.wallThickness # pylint: disable=invalid-name
        r = self.ratioPointsWall # pylint: disable=invalid-name
        tailIn = self.tailLengthInside
        tailOut = self.tailLengthOutside
        aIn = self.aIn
        aOut = self.aOut

        def term1(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array((1 - r)
                * (r * tailOut - L)
                * np.arctanh(
                    (1 - x + np.sqrt(aOut**2 + (x - r) ** 2))
                    / np.sqrt(aOut**2 + (1 - r) ** 2)
                    + 0j
                ).real
                / np.sqrt(aOut**2 + (1 - r) ** 2)
                / r
            )

        def term2(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array(-(1 + r)
                * (r * tailOut - L)
                * np.arctanh(
                    (1 + x - np.sqrt(aOut**2 + (x - r) ** 2))
                    / np.sqrt(aOut**2 + (1 + r) ** 2)
                    + 0j
                ).real
                / np.sqrt(aOut**2 + (1 + r) ** 2)
                / r
            )
        def term3(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array((1 - r)
                * (r * tailIn - L)
                * np.arctanh(
                    (1 + x - np.sqrt(aIn**2 + (x + r) ** 2))
                    / np.sqrt(aIn**2 + (1 - r) ** 2)
                    + 0j
                ).real
                / np.sqrt(aIn**2 + (1 - r) ** 2)
                / r
            )
        def term4(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array(-(1 + r)
                * (r * tailIn - L)
                * np.arctanh(
                    (1 - x + np.sqrt(aIn**2 + (x + r) ** 2))
                    / np.sqrt(aIn**2 + (1 + r) ** 2)
                    + 0j
                ).real
                / np.sqrt(aIn**2 + (1 + r) ** 2)
                / r
            )
        def term5(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array((tailIn + tailOut - 4 * self.smoothing * L / r)
                            * np.arctanh(x)
            )
        def totalMapping(x: np.ndarray) -> np.ndarray: # pylint: disable=invalid-name
            return np.array((term1(x) + term2(x) + term3(x) + term4(x) + term5(x)) / 2)

        z = ( # pylint: disable=invalid-name
             totalMapping(zCompact) - totalMapping(np.array(0.0)) + self.wallCenter)
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
        Derivative of transforms coordinates to [-1, 1] interval
        """
        L = self.wallThickness # pylint: disable=invalid-name
        r = self.ratioPointsWall # pylint: disable=invalid-name
        tailIn = self.tailLengthInside
        tailOut = self.tailLengthOutside
        aIn = self.aIn
        aOut = self.aOut

        dzdzCompact = (
            (tailIn - L / r)
            * (1 - (zCompact + r) / np.sqrt(aIn**2 + (zCompact + r) ** 2))
            / 2
        )
        dzdzCompact += (
            (tailOut - L / r)
            * (1 + (zCompact - r) / np.sqrt(aOut**2 + (zCompact - r) ** 2))
            / 2
        )
        dzdzCompact += (1 - 2 * self.smoothing) * L / r
        dzdzCompact /= 1 - zCompact**2

        dpzdpzCompact = 2 * self.momentumFalloffT / (1 - pzCompact**2)
        dppdppCompact = self.momentumFalloffT / (1 - ppCompact)

        return dzdzCompact, dpzdpzCompact, dppdppCompact
