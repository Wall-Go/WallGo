import numpy as np
from scipy.special import eval_chebyt,eval_chebyu

class Polynomial:
    r"""
    Evaluates the cardinal basis and polynomial series.

    Parameters
    ----------
    grid : Grid
        An object of the class Grid
    """
    def __init__(self,grid):
        self.grid = grid

        self.gridValues = self.grid.getCompactCoordinates(True)

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

    def evaluateCardinal(self,x,f,directions=('z','pz','pp')):
        """
        Evaluates the cardinal series with coefficients f at the point x.

        Parameters
        ----------
        x : array_like, shape (...,N)
            Coordinate at which to evaluate the polynomial series.
        f : array_like
            Coefficients of the series, which are the values of the function evaluated on the grid. Must contain the endpoints.
        directions : tuple of length N, optional
            Tuple containing all the directions along which to evaluate the series. Default is ('z','pz','pp').

        Returns
        -------
        series : float
            Value of the series at the point x.
        """

        x = np.asarray(x)
        f = np.asarray(f)
        xShapeSize = len(x.shape)-1
        N = len(directions)

        #Computing and multiplying the cardinal functions in all the directions
        cardinals = 1
        for i in range(N):
            n = None
            match directions[i]:
                case 'z': n = np.arange(self.grid.M+1)
                case 'pz': n = np.arange(self.grid.N+1)
                case 'pp': n = np.arange(self.grid.N)
            createAxes = tuple(np.delete(-np.arange(N)-1,N-i-1))
            cardinals = cardinals*np.expand_dims(self.cardinal(x[...,i], n, directions[i]), createAxes)

        #Resizing f and summing over all the terms
        f = np.expand_dims(f, tuple(np.arange(xShapeSize)))
        series = np.sum(f*cardinals, axis=tuple(-np.arange(N)-1))

        return series

    def evaluateChebyshev(self,x,f,directions=('z','pz','pp')):
        """
        Evaluates the Chebyshev series with coefficients f at the point x.

        Parameters
        ----------
        x : array_like, shape (...,N)
            Coordinate at which to evaluate the Chebyshev series.
        f : array_like
            Spectral coefficients of the series. Should not include the endpoints.
        directions : tuple of length N, optional
            Tuple containing all the directions along which to evaluate the series. Default is ('z','pz','pp').

        Returns
        -------
        series : float
            Value of the series at the point x.
        """

        x = np.asarray(x)
        f = np.asarray(f)
        xShapeSize = len(x.shape)-1
        N = len(directions)

        #Computing and multiplying the cardinal functions in all the directions
        chebyshevs = 1
        for i in range(N):
            n,restriction = None,None
            match directions[i]:
                case 'z':
                    n = np.arange(2,self.grid.M+1)
                    restriction = 'full'
                case 'pz':
                    n = np.arange(2,self.grid.N+1)
                    restriction = 'full'
                case 'pp':
                    n = np.arange(1,self.grid.N)
                    restriction = 'partial'
            createAxes = tuple(np.delete(-np.arange(N)-1,N-i-1))
            chebyshevs = chebyshevs*np.expand_dims(self.chebyshev(x[...,i], n, restriction), createAxes)

        #Resizing f and summing over all the terms
        f = np.expand_dims(f, tuple(np.arange(xShapeSize)))
        series = np.sum(f*chebyshevs, axis=tuple(-np.arange(N)-1))

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

    def matrix(self, basis,direction, endpoints=False):
        r"""
        Returns the matrix :math:`M_{ij}=T_j(x_i)` or :math:`M_{ij}=C_j(x_i)` computed in a specific direction.

        Parameters
        ----------
        basis : string
            Select the basis of polynomials. Can be 'Cardinal' or 'Chebyshev'
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        """

        if basis == 'Cardinal':
            return self.cardinalMatrix(direction,endpoints)
        elif basis == 'Chebyshev':
            return self.chebyshevMatrix(direction,endpoints)
        else:
            raise ValueError("basis must be either 'Cardinal' or 'Chebyshev'.")


    def cardinalDeriv(self,direction,endpoints=False):
        """
        Computes the derivative matrix of the cardinal functions in some direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """

        grid = None
        match direction:
            case 'z': grid = self.gridValues[0]
            case 'pz': grid = self.gridValues[1]
            case 'pp': grid = self.gridValues[2]

        #Computing the diagonal part
        diagonal = np.sum(np.where(grid[:,None]-grid[None,:] == 0, 0, np.divide(1, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0)),axis=1)

        #Computing the off-diagonal part
        offDiagonal = np.prod(np.where((grid[:,None,None]-grid[None,None,:])*(grid[None,:,None]-grid[None,None,:]) == 0, 1, np.divide(grid[None,:,None]-grid[None,None,:], grid[:,None,None]-grid[None,None,:], where=grid[:,None,None]-grid[None,None,:]!=0)),axis=-1)

        #Putting all together
        derivWithEndpoints = np.where(grid[:,None]-grid[None,:] == 0,diagonal[:,None], np.divide(offDiagonal, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0))

        deriv = None
        if not endpoints:
            if direction == 'z' or direction == 'pz':
                deriv = derivWithEndpoints[1:-1,1:-1]
            elif direction == 'pp':
                deriv = derivWithEndpoints[:-1,:-1]
        else:
            deriv = derivWithEndpoints

        return np.transpose(deriv)

    def chebyshevDeriv(self,direction,endpoints=False):
        """
        Computes the derivative matrix of the Chebyshev polynomials in some direction.

        Parameters
        ----------
        direction : string
            Select the direction in which to compute the matrix. Can either be 'z', 'pz' or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

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

        deriv = n[None,:]*eval_chebyu(n[None,:]-1,grid[:,None])

        if restriction == 'full':
            deriv -= np.where(n[None,:]%2==0,0,1)

        return deriv

    def deriv(self,basis,direction,endpoints=False):
        """
        Computes the derivative matrix of either the Chebyshev or cardinal polynomials in some direction.

        Parameters
        ----------
        basis : string
            Select the basis of polynomials. Can be 'Cardinal' or 'Chebyshev'
        direction : string
            Select the direction in which to compute the matrix. Can be 'z', 'pz' or 'pp'.
        endpoints : Bool, optional
            If True, include endpoints of grid. Default is False.

        Returns
        -------
        deriv : array_like
            Derivative matrix.

        """

        if basis == 'Cardinal':
            return self.cardinalDeriv(direction,endpoints)
        elif basis == 'Chebyshev':
            return self.chebyshevDeriv(direction,endpoints)
        else:
            raise ValueError("basis must be either 'Cardinal' or 'Chebyshev'.")
