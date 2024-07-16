"""
Class that stores and perfom operation on the coefficients of a polynomial series.
"""

from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt
from scipy.special import eval_chebyt, eval_chebyu

from .grid import Grid


class Polynomial:
    """
    Class that stores and perfoms operations on the coefficients of a polynomial series.
    """

    # Static tuples of allowed basis and direction strings
    ALLOWED_BASES: typing.Final[tuple[str, ...]] = ("Cardinal", "Chebyshev", "Array")
    ALLOWED_DIRECTIONS: typing.Final[tuple[str, ...]] = ("z", "pz", "pp", "Array")

    def __init__(
        self,
        coefficients: np.ndarray,
        grid: Grid, basis: str | tuple[str,...]="Cardinal",
        direction: str | tuple[str,...]="z",
        endpoints: bool | tuple[bool,...]=False,
    ):
        """
        Initialization of Polynomial object.

        Parameters
        ----------
        coefficients : array-like
            Array of rank N containing the coefficients of a polynomial defined
            by the object grid.
        grid : Grid
            An object of the Grid class defining the polynomial.
        basis : string or tuple of strings, optional
            Tuple of length N specifying in what basis each dimension of
            coefficients is defined. Each component can either be 'Cardinal',
            'Chebyshev' or 'Array'. The latter is to be used if the axis does
            not corresponds to a polynomial series. Can also be a single
            string, in which case all the dimensions are assumed to be in that
            basis. The default is 'Cardinal'.
        direction : string or tuple of strings, optional
            Tuple of length N specifying what direction each dimension of
            coefficients represents. Each component can either be 'z', 'pz',
            'pp' or 'Array'. The latter can only be used if basis is 'Array'
            for that axis. Can also be a single string, in which case all the
            dimensions are assumed to be in that direction. The default is 'z'.
        endpoints : bool or tuple of bool, optional
            Tuple of length N specifying wheither each dimension includes the
            endpoints. Can also be a single bool, in which case all the
            dimensions are assumed to be the same. If False, the polynomial is
            assumed to be 0 at the endpoints. The default is False.

        Returns
        -------
        None.
        """

        self.coefficients = np.asanyarray(coefficients)
        self.rank = len(self.coefficients.shape)
        self.grid = grid

        if isinstance(basis, str):
            basis = self.rank * (basis,)
        self._checkBasis(basis)
        self.basis = basis

        if isinstance(direction, str):
            direction = self.rank * (direction,)
        self._checkDirection(direction)

        if isinstance(endpoints, bool):
            endpoints = self.rank * (endpoints,)
        self._checkEndpoints(endpoints)

        self.direction = direction
        self.endpoints = endpoints

        self._checkCoefficients(self.coefficients)

    def __getitem__(self, key: int | tuple[int,...]) -> Polynomial:
        basisList: list[str] = []
        endpointsList: list[bool] = []
        directionList: list[str] = []
        if not isinstance(key, tuple):
            key = (key,)
        n = 0 # pylint: disable=invalid-name
        for i, k in enumerate(key):
            if isinstance(k, int):
                n += 1 # pylint: disable=invalid-name
            elif isinstance(k, slice):
                basisList.append(self.basis[i])
                directionList.append(self.direction[i])
                endpointsList.append(self.endpoints[i])
                n += 1 # pylint: disable=invalid-name
            elif k is None:
                basisList.append("Array")
                directionList.append("z")
                endpointsList.append(False)
            else:
                raise ValueError("Polynomial error: invalid key.")
        basis = tuple(basisList) + self.basis[n:]
        direction = tuple(directionList) + self.direction[n:]
        endpoints = tuple(endpointsList) + self.endpoints[n:]

        coefficients = np.array(self.coefficients[key])
        return Polynomial(coefficients, self.grid, basis, direction, endpoints)

    def __mul__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        if isinstance(poly, Polynomial):
            assert self._isBroadcastable(
                self.coefficients, poly.coefficients
            ), "Polynomial error: the two Polynomial objects are not broadcastable."
            basis, direction, endpoints = self._findContraction(poly)
            return Polynomial(
                self.coefficients * poly.coefficients,
                self.grid,
                basis,
                direction,
                endpoints,
            )
        newCoeff = np.array(poly * self.coefficients)
        assert (
            len(newCoeff.shape) == self.rank
        ), """Polynomial error: the rank of the resulting Polynomial object must be
        the same as the original one."""
        return Polynomial(
            newCoeff, self.grid, self.basis, self.direction, self.endpoints
        )

    def __add__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        if isinstance(poly, Polynomial):
            assert self._isBroadcastable(
                self.coefficients, poly.coefficients
            ), "Polynomial error: the two Polynomial objects are not broadcastable."
            basis, direction, endpoints = self._findContraction(poly)
            return Polynomial(
                self.coefficients + poly.coefficients,
                self.grid,
                basis,
                direction,
                endpoints,
            )
        newCoeff = poly + self.coefficients

        newCoeff = np.asanyarray(newCoeff)
        assert (
            newCoeff.ndim == self.rank
        ), """Polynomial error: the rank of the resulting Polynomial object must be
        the same as the original one."""

        return Polynomial(
            newCoeff, self.grid, self.basis, self.direction, self.endpoints
        )

    def __sub__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        return self.__add__((-1.0) * poly)

    def __rmul__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        return self.__mul__(poly)

    def __radd__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        return self.__add__(poly)

    def __rsub__(self, poly: Polynomial | np.ndarray | float) -> Polynomial:
        return (-1) * self.__sub__(poly)

    def _findContraction(
            self,
            poly: Polynomial,
            ) -> tuple[tuple[str,...], tuple[str,...], tuple[bool,...]]:
        """
        Find the tuples basis, direction and endpoints resulting from the
        contraction of self and poly

        Parameters
        ----------
        poly : Polynomial
            Polynomial object.

        Returns
        -------
        basis : tuple
            basis tuple of the contracted polynomial.
        direction : tuple
            direction tuple of the contracted polynomial.
        endpoints : tuple
            endpoints tuple of the contracted polynomial.

        """
        assert (
            self.rank == poly.rank
        ), """Polynomial error: you can only combine two Polynomial objects with the
        same rank."""
        basis, endpoints, direction = [], [], []
        for i in range(self.rank):
            assert (
                self.coefficients.shape[i] == 1
                or poly.coefficients.shape[i] == 1
                or (
                    self.basis[i] == poly.basis[i]
                    and self.direction[i] == poly.direction[i]
                    and self.endpoints[i] == poly.endpoints[i]
                )
            ), "Polynomial error: the two Polynomial objects are not broadcastable."
            if self.coefficients.shape[i] > 1:
                basis.append(self.basis[i])
                direction.append(self.direction[i])
                endpoints.append(self.endpoints[i])
            else:
                basis.append(poly.basis[i])
                direction.append(poly.direction[i])
                endpoints.append(poly.endpoints[i])
        return tuple(basis), tuple(direction), tuple(endpoints)

    def changeBasis(
            self,
            newBasis: str | tuple[str,...],
            inverseTranspose: bool=False,
            ) -> None:
        """
        Change the basis of the polynomial. Will change self.coefficients
        accordingly.

        Parameters
        ----------
        newBasis : string or tuple of strings, optional
            Tuple of length N specifying in what basis each dimension of
            self.coefficients is defined. Each component can either be
            'Cardinal' or 'Chebyshev'. Can also be a single string, in which
            case all the dimensions are assumed to be in that basis.
        inverseTranspose : bool, optional
            If True, take the inverse-transpose of the transformation matrix.
            This is useful, for example, when changing the basis of the
            collision array.

        Returns
        -------
        None.

        """
        if isinstance(newBasis, str):
            newBasis = self.rank * (newBasis,)
        self._checkBasis(newBasis)

        for i in range(self.rank):
            if (
                newBasis[i] != self.basis[i]
                and newBasis[i] != "Array"
                and self.basis[i] != "Array"
            ):
                # Choosing the appropriate x, n and restriction
                x = self.grid.getCompactCoordinates( # pylint: disable=invalid-name
                    self.endpoints[i], self.direction[i]
                )
                n, restriction = None, None # pylint: disable=invalid-name
                if self.endpoints[i]:
                    if self.direction[i] == "z":
                        n = np.arange(self.grid.M + 1) # pylint: disable=invalid-name
                    elif self.direction[i] == "pz":
                        n = np.arange(self.grid.N + 1) # pylint: disable=invalid-name
                    else:
                        n = np.arange(self.grid.N) # pylint: disable=invalid-name
                else:
                    if self.direction[i] == "z":
                        n = np.arange(2, self.grid.M + 1) # pylint: disable=invalid-name
                        restriction = "full"
                    elif self.direction[i] == "pz":
                        n = np.arange(2, self.grid.N + 1) # pylint: disable=invalid-name
                        restriction = "full"
                    else:
                        n = np.arange(1, self.grid.N) # pylint: disable=invalid-name
                        restriction = "partial"

                # Computing the Tn matrix
                tnMatrix = np.array(self.chebyshev(x[:, None], n[None, :], restriction))
                if newBasis[i] == "Chebyshev":
                    tnMatrix = np.linalg.inv(tnMatrix)

                if inverseTranspose:
                    tnMatrix = np.transpose(np.linalg.inv(tnMatrix))

                tnMatrix = np.expand_dims(
                    tnMatrix,
                    tuple(np.arange(i)) + tuple(np.arange(i + 2, self.rank + 1))
                )

                # Contracting M with self.coefficient
                self.coefficients = np.sum(
                    tnMatrix * np.expand_dims(self.coefficients, i), axis=i + 1
                )
        self.basis = newBasis

    def evaluate(
            self,
            compactCoord: np.ndarray,
            axes: tuple[int,...] | None=None,
            ) -> np.ndarray | float:
        """
        Evaluates the polynomial at the compact coordinates x.

        Parameters
        ----------
        compactCoord : array-like
            Compact coordinates at which to evaluate the polynomial. Must have
            a shape (len(axes),:) or (len(axes),).
        axes : tuple or None, optional
            Axes along which to be evaluated. If None, evaluate the polynomial
            along all the axes. Default is None.

        Returns
        -------
        array-like
            Values of the polynomial at x.

        """
        compactCoord = np.asarray(compactCoord)
        if axes is None:
            axes = tuple(np.arange(self.rank))

        assert (
            compactCoord.shape[0] == len(axes) and 1 <= len(compactCoord.shape) <= 2
        ), f"""Polynomial error: compactCoord has shape {compactCoord.shape} but must
        be ({self.rank},:) or ({self.rank},)."""
        singlePoint = False
        if len(compactCoord.shape) == 1:
            compactCoord = compactCoord.reshape((len(axes), 1))
            singlePoint = True

        polynomials = np.ones((compactCoord.shape[1],) + self.coefficients.shape)
        for j, i in enumerate(axes):
            assert (
                self.basis[i] != "Array"
            ), "Polynomial error: cannot evaluate along an 'Array' axis."
            # Choosing the appropriate n
            n: np.ndarray # pylint: disable=invalid-name
            if self.endpoints[i]:
                if self.direction[i] == "z":
                    n = np.arange(self.grid.M + 1) # pylint: disable=invalid-name
                elif self.direction[i] == "pz":
                    n = np.arange(self.grid.N + 1) # pylint: disable=invalid-name
                else:
                    n = np.arange(self.grid.N) # pylint: disable=invalid-name
            else:
                if self.direction[i] == "z":
                    n = np.arange(1, self.grid.M) # pylint: disable=invalid-name
                elif self.direction[i] == "pz":
                    n = np.arange(1, self.grid.N) # pylint: disable=invalid-name
                else:
                    n = np.arange(self.grid.N - 1) # pylint: disable=invalid-name

            # Computing the polynomial basis in the i direction
            pn: np.ndarray # pylint: disable=invalid-name
            if self.basis[i] == "Cardinal":
                pn = np.array( # pylint: disable=invalid-name
                    self.cardinal(compactCoord[j, :, None], n[None, :],
                                  self.direction[i]))

            elif self.basis[i] == "Chebyshev":
                restriction = None
                if not self.endpoints[i]:
                    n += 1 # pylint: disable=invalid-name
                    if self.direction[i] == "z":
                        restriction = "full"
                    elif self.direction[i] == "pz":
                        restriction = "full"
                    else:
                        restriction = "partial"
                pn = np.array( # pylint: disable=invalid-name
                    self.chebyshev(compactCoord[j, :, None], n[None, :], restriction))

            polynomials *= np.expand_dims(
                pn, tuple(np.arange(1, i + 1)) + tuple(np.arange(i + 2, self.rank + 1))
            )

        result = np.sum(
            self.coefficients[None, ...] * polynomials, axis=tuple(np.array(axes) + 1)
        )
        if singlePoint:
            return float(result[0])
        return np.array(result)

    def cardinal(
            self,
            compactCoord: npt.ArrayLike,
            n: npt.ArrayLike, # pylint: disable=invalid-name
            direction: str,
            ) -> np.ndarray:
        r"""
        Computes the cardinal polynomials :math:`C_n(x)` defined by grid.

        Parameters
        ----------
        compactCoord : array_like
            Compact coordinate at which to evaluate the Chebyshev polynomial. Must be
            broadcastable with n.
        n : array_like
            Order of the cardinal polynomial to evaluate. Must be
            broadcastable with x.
        direction : string
            Select the direction in which to compute the matrix.
            Can either be 'z', 'pz' or 'pp'.

        Returns
        -------
        cn : array_like
            Values of the cardinal functions.
        """

        compactCoord = np.asarray(compactCoord)
        n = np.asarray(n)

        assert self._isBroadcastable(
            compactCoord, n
        ), "Polynomial error: x and n are not broadcastable."
        assert direction in self.ALLOWED_DIRECTIONS, (
            f"Polynomial error: unkown direction {direction}"
        )

        # Selecting the appropriate grid and resizing it
        grid = self.grid.getCompactCoordinates(True, direction)
        completeGrid = np.expand_dims(
            grid, tuple(np.arange(1, len((n * compactCoord).shape) + 1)))
        nGrid = grid[n]

        # Computing all the factor in the product defining the cardinal functions
        cardinalPartial = np.divide(
            compactCoord - completeGrid,
            nGrid - completeGrid,
            where=nGrid - completeGrid != 0
        )

        # Multiplying all the factors to get the cardinal functions
        cardinal = np.prod(np.where(nGrid - completeGrid == 0, 1, cardinalPartial),
                           axis=0)

        return np.array(cardinal)

    def chebyshev(
            self,
            compactCoord: npt.ArrayLike,
            n: npt.ArrayLike, # pylint: disable=invalid-name
            restriction: str | None=None
            ) -> np.ndarray:
        r"""
        Computes the Chebyshev polynomial :math:`T_n(x)`.

        Parameters
        ----------
        compactCoord : array_like
            Compact coordinate at which to evaluate the Chebyshev polynomial. Must be
            broadcastable with n.
        n : array_like
            Order of the Chebyshev polynomial to evaluate. Must be
            broadcastable with x.
        restriction : None or string, optional
            Select the restriction on the Chebyshev basis. If None, evaluates
            the unrestricted basis. If 'full', the polynomials are 0 at
            :math:`x=\pm 1`. If 'partial', the polynomials are 0 at :math:`x=+1`.

        Returns
        -------
        chen : array_like
            Values of the polynomial

        """

        compactCoord = np.asarray(compactCoord)
        n = np.asarray(n)

        assert self._isBroadcastable(
            compactCoord, n
        ), "Polynomial error: compactCoord and n are not broadcastable."

        # Computing the unrestricted basis
        cheb = eval_chebyt(n, compactCoord)

        # Applying the restriction
        if restriction == "partial":
            cheb -= 1
        elif restriction == "full":
            cheb -= np.where(n % 2 == 0, 1, compactCoord)

        return np.array(cheb)

    def integrate(
            self,
            axis: int | tuple[int,...] | None=None,
            weight: npt.ArrayLike=1,
            ) -> Polynomial | float:
        r"""
        Computes the integral of the polynomial :math:`\int_{-1}^1 dx P(x)w(x)`
        along some axis using Gauss-Chebyshev-Lobatto quadrature.

        Parameters
        ----------
        axis : None, int or tuple
            axis along which the integral is taken. Can either be None, a int or a
            tuple of int. If None, integrate along all the axes.
        weight : array-like, optional
            Integration weight. Must be an object broadcastable with
            self.coefficients. Default is 1.

        Returns
        -------
        Polynomial or float
            If axis=None, returns a float. Otherwise, returns an object of the
            class Polynomial containing the coefficients of the
            integrated polynomial along the remaining axes.

        """
        if weight is None:
            weight = 1

        if axis is None:
            axis = tuple(np.arange(self.rank))
        if isinstance(axis, int):
            axis = (axis,)
            self._checkAxis(axis)

        # Express the integrated axes in the cardinal basis
        basis = []
        for i in range(self.rank):
            if i in axis:
                assert (
                    self.basis[i] != "Array"
                ), "Polynomial error: cannot integrate along an 'Array' axis."
                basis.append("Cardinal")
            else:
                basis.append(self.basis[i])
        self.changeBasis(tuple(basis))

        integrand = weight * self.coefficients
        newBasis, newDirection, newEndpoints = [], [], []
        for i in range(self.rank):
            if i in axis:
                compactCoord = self.grid.getCompactCoordinates(
                    self.endpoints[i], self.direction[i]
                )
                weights = np.pi * np.ones(compactCoord.size)
                if self.direction[i] == "z":
                    weights /= self.grid.M
                elif self.direction[i] == "pz":
                    weights /= self.grid.N
                elif self.direction[i] == "pp":
                    weights /= self.grid.N - 1
                    if not self.endpoints[i]:
                        weights[0] /= 2
                if self.endpoints[i]:
                    weights[0] /= 2
                    weights[-1] /= 2
                integrand *= np.expand_dims(
                    np.sqrt(1 - compactCoord**2) * weights,
                    tuple(np.arange(i)) + tuple(np.arange(i + 1, self.rank)),
                )
            else:
                newBasis.append(self.basis[i])
                newDirection.append(self.direction[i])
                newEndpoints.append(self.endpoints[i])

        result = np.sum(integrand, axis)

        ## This check fails too easily with extended types
        """
        if isinstance(result, float):
            return result
        """

        if np.asanyarray(result).ndim == 0:
            return float(result)
        return Polynomial(
            result,
            self.grid,
            tuple(newBasis),
            tuple(newDirection),
            tuple(newEndpoints),
            )

    def derivative(self, axis: int | tuple[int,...]) -> Polynomial:
        """
        Computes the derivative of the polynomial and returns it in a
        Polynomial object.

        Parameters
        ----------
        axis : int or tuple
            axis along which the derivative is taken. Can either be a int or a
            tuple of int.

        Returns
        -------
        Polynomial
            Object of the class Polynomial containing the coefficients of the
            derivative polynomial (in the compact coordinates). The axis along
            which the derivative is taken is always returned with the endpoints
            in the cardinal basis.

        """

        if isinstance(axis, int):
            axis = (axis,)
        self._checkAxis(axis)

        coeffDeriv = np.array(self.coefficients)
        basis, endpoints = [], []

        for i in range(self.rank):
            if i in axis:
                assert (
                    self.basis[i] != "Array"
                ), "Polynomial error: cannot differentiate along an 'Array' axis."
                derivMatrix = self.derivMatrix(
                    self.basis[i], self.direction[i], self.endpoints[i]
                )
                derivMatrix = np.expand_dims(
                    derivMatrix,
                    tuple(np.arange(i)) + tuple(np.arange(i + 2, self.rank + 1))
                )
                coeffDeriv = np.sum(derivMatrix*np.expand_dims(coeffDeriv, i), axis=i+1)
                basis.append("Cardinal")
                endpoints.append(True)
            else:
                basis.append(self.basis[i])
                endpoints.append(self.endpoints[i])
        return Polynomial(
            coeffDeriv, self.grid, tuple(basis), self.direction, tuple(endpoints)
        )

    def matrix(self, basis: str, direction: str, endpoints: bool=False) -> np.ndarray:
        r"""
        Returns the matrix :math:`M_{ij}=T_j(x_i)` or :math:`M_{ij}=C_j(x_i)` computed 
        in a specific direction.

        Parameters
        ----------
        basis : string
            Select the basis of polynomials. Can be 'Cardinal' or 'Chebyshev'
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' 
            or 'pp'
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """

        if basis == "Cardinal":
            return self._cardinalMatrix(direction, endpoints)
        if basis == "Chebyshev":
            return self._chebyshevMatrix(direction, endpoints)
        raise ValueError("basis must be either 'Cardinal' or 'Chebyshev'.")

    def derivMatrix(
            self,
            basis: str,
            direction: str,
            endpoints: bool=False,
            ) -> np.ndarray:
        """
        Computes the derivative matrix of either the Chebyshev or cardinal polynomials 
        in some direction.

        Parameters
        ----------
        basis : string
            Select the basis of polynomials. Can be 'Cardinal' or 'Chebyshev'
        direction : string
            Select the direction in which to compute the matrix. Can be 'z', 'pz' or 
            'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """
        assert basis in ['Cardinal', 'Chebyshev'], """basis must be either
                                                        'Cardinal' or 'Chebyshev'."""

        if basis == "Cardinal":
            return self._cardinalDeriv(direction, endpoints)
        return self._chebyshevDeriv(direction, endpoints)

    def _cardinalMatrix(self, direction: str, endpoints: bool=False) -> np.ndarray:
        r"""
        Returns the matrix :math:`M_{ij}=C_j(x_i)` computed in a specific direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz'
            or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """
        assert direction in ['z','pz','pp'], """direction must be either 'z',
                                                        'pz' or 'pp'."""

        if direction == "z":
            return np.identity(self.grid.M - 1 + 2 * endpoints)
        if direction == "pz":
            return np.identity(self.grid.N - 1 + 2 * endpoints)
        return np.identity(self.grid.N - 1 + endpoints)

    def _chebyshevMatrix(self, direction: str, endpoints: bool=False) -> np.ndarray:
        r"""
        Returns the matrix :math:`M_{ij}=T_j(x_i)` computed in a specific direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz'
            or 'pp'
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """

        grid: np.ndarray
        n: np.ndarray # pylint: disable=invalid-name
        restriction = None
        if direction == "z":
            grid = self.grid.getCompactCoordinates(endpoints)[0]
            n = np.arange(grid.size) + 2 - 2 * endpoints # pylint: disable=invalid-name
            restriction = "full"
        elif direction == "pz":
            grid = self.grid.getCompactCoordinates(endpoints)[1]
            n = np.arange(grid.size) + 2 - 2 * endpoints # pylint: disable=invalid-name
            restriction = "full"
        elif direction == "pp":
            grid = self.grid.getCompactCoordinates(endpoints)[2]
            n = np.arange(grid.size) + 1 - endpoints # pylint: disable=invalid-name
            restriction = "partial"
        if endpoints:
            restriction = None

        return self.chebyshev(grid[:, None], n[None, :], restriction)

    def _cardinalDeriv(self, direction: str, endpoints: bool=False) -> np.ndarray:
        """
        Computes the derivative matrix of the cardinal functions in some direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz'
            or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """

        grid = self.grid.getCompactCoordinates(True, direction)

        # Computing the diagonal part
        diagonal = np.sum(
            np.where(
                grid[:, None] - grid[None, :] == 0,
                0,
                np.divide(
                    1,
                    grid[:, None] - grid[None, :],
                    where=grid[:, None] - grid[None, :] != 0,
                ),
            ),
            axis=1,
        )

        # Computing the off-diagonal part
        offDiagonal = np.prod(
            np.where(
                (grid[:, None, None] - grid[None, None, :])
                * (grid[None, :, None] - grid[None, None, :])
                == 0,
                1,
                np.divide(
                    grid[None, :, None] - grid[None, None, :],
                    grid[:, None, None] - grid[None, None, :],
                    where=grid[:, None, None] - grid[None, None, :] != 0,
                ),
            ),
            axis=-1,
        )

        # Putting all together
        derivWithEndpoints = np.where(
            grid[:, None] - grid[None, :] == 0,
            diagonal[:, None],
            np.divide(
                offDiagonal,
                grid[:, None] - grid[None, :],
                where=grid[:, None] - grid[None, :] != 0,
            ),
        )

        deriv: np.ndarray
        if not endpoints:
            if direction in ["z", "pz"]:
                deriv = derivWithEndpoints[1:-1, :]
            elif direction == "pp":
                deriv = derivWithEndpoints[:-1, :]
        else:
            deriv = derivWithEndpoints

        return np.transpose(deriv)

    def _chebyshevDeriv(self, direction: str, endpoints: bool=False) -> np.ndarray:
        """
        Computes the derivative matrix of the Chebyshev polynomials in some direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz'
            or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """

        grid = self.grid.getCompactCoordinates(True, direction)
        n: np.ndarray # pylint: disable=invalid-name
        restriction = None
        if direction == "z":
            n = np.arange(2 - 2 * endpoints, grid.size) # pylint: disable=invalid-name
            restriction = "full"
        elif direction == "pz":
            n = np.arange(2 - 2 * endpoints, grid.size) # pylint: disable=invalid-name
            restriction = "full"
        elif direction == "pp":
            n = np.arange(1 - endpoints, grid.size) # pylint: disable=invalid-name
            restriction = "partial"

        deriv = n[None, :] * eval_chebyu(n[None, :] - 1, grid[:, None])

        if restriction == "full" and not endpoints:
            deriv -= np.where(n[None, :] % 2 == 0, 0, 1)

        return np.array(deriv)

    def _checkBasis(self, basis: tuple[str,...]) -> None:
        assert isinstance(
            basis, tuple
        ), "Polynomial error: basis must be a tuple or a string."
        assert (
            len(basis) == self.rank
        ), 'Polynomial error: basis must be a tuple of length "rank".'
        for x in basis: # pylint: disable=invalid-name
            assert x in self.ALLOWED_BASES, f"Polynomial error: unkown basis {x}"

    def _checkDirection(self, direction: tuple[str,...]) -> None:
        assert isinstance(
            direction, tuple
        ), "Polynomial error: direction must be a tuple or a string."
        assert (
            len(direction) == self.rank
        ), 'Polynomial error: direction must be a tuple of length "rank".'
        for i, x in enumerate(direction): # pylint: disable=invalid-name
            assert x in self.ALLOWED_DIRECTIONS, (
                f"Polynomial error: unkown direction {x}"
            )
            if x == "Array":
                assert (
                    self.basis[i] == "Array"
                ), """Polynomial error: if the direction is 'Array', the basis must be
                'Array' too."""

    def _checkEndpoints(self, endpoints: tuple[bool,...]) -> None:
        assert isinstance(
            endpoints, tuple
        ), "Polynomial error: endpoints must be a tuple or a bool."
        assert (
            len(endpoints) == self.rank
        ), 'Polynomial error: endpoints must be a tuple of length "rank".'
        for x in endpoints: # pylint: disable=invalid-name
            assert isinstance(
                x, bool
            ), "Polynomial error: endpoints can only contain bool."

    def _checkCoefficients(self, coefficients: np.ndarray) -> None:
        for i, size in enumerate(coefficients.shape):
            if self.basis[i] != "Array":
                if self.direction[i] == "z":
                    assert (
                        size + 2 * (1 - self.endpoints[i]) == self.grid.M + 1
                    ), f"""Polynomial error: coefficients with invalid size in
                    dimension {i}."""
                elif self.direction[i] == "pz":
                    assert (
                        size + 2 * (1 - self.endpoints[i]) == self.grid.N + 1
                    ), f"""Polynomial error: coefficients with invalid size in
                    dimension {i}."""
                else:
                    assert (
                        size + (1 - self.endpoints[i]) == self.grid.N
                    ), f"""Polynomial error: coefficients with invalid size in
                    dimension {i}."""

    def _checkAxis(self, axis: tuple[int,...]) -> None:
        assert isinstance(
            axis, tuple
        ), "Polynomial error: axis must be a tuple or a int."
        for x in axis: # pylint: disable=invalid-name
            assert isinstance(x, int), "Polynomial error: axis must be a tuple of int."
            assert 0 <= x < self.rank, "Polynomial error: axis out of range."

    def _isBroadcastable(self, array1: np.ndarray, array2: np.ndarray) -> bool:
        """
        Verifies that array1 and array2 are broadcastable, which mean that they
        can be multiplied together.

        Parameters
        ----------
        array1 : array_like
            First array.
        array2 : array_like
            Second array.

        Returns
        -------
        bool
            True if the two arrays are broadcastable, otherwise False.

        """
        for a, b in zip( # pylint: disable=invalid-name
            np.asanyarray(array1).shape[::-1], np.asanyarray(array2).shape[::-1]
        ):
            if a == 1 or b == 1 or a == b:
                pass
            else:
                return False
        return True
