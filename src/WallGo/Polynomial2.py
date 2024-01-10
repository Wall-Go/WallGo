import numpy as np
from scipy.special import eval_chebyt,eval_chebyu


class Polynomial:
    def __init__(self, coefficients, grid, basis='Cardinal', direction='z', endpoints=False):
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
            coefficients is defined. Each component can either be 'Cardinal' or
            'Chebyshev'. Can also be a single string, in which case all the 
            dimensions are assumed to be in that basis. The default is 
            'Cardinal'.
        direction : string or tuple of strings, optional
            Tuple of length N specifying what direction each dimension of 
            coefficients represents. Each component can either be 'z', 'pz' or
            'pp'. Can also be a single string, in which case all the 
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
        self.N = len(self.coefficients.shape) #Can we use another symbol here? Easy to confuse this with Grid.N
        self.grid = grid
        
        self.allowedBasis = ['Cardinal','Chebyshev']
        self.allowedDirection = ['z','pz','pp']
        
        if isinstance(basis, str):
            basis = self.N*(basis,)
        self.__checkBasis(basis)
            
        if isinstance(direction, str):
            direction = self.N*(direction,)
        self.__checkDirection(direction)
            
        if isinstance(endpoints, bool):
            endpoints = self.N*(endpoints,)
        self.__checkEndpoints(endpoints)
            
        self.basis = basis
        self.direction = direction
        self.endpoints = endpoints
        
        self.__checkCoefficients(coefficients)
        
    def __getitem__(self, key):
        basis, endpoints, direction = [],[],[]
        if not isinstance(key,tuple):
            key = (key,)
        n = 0
        for i,k in enumerate(key):
            if isinstance(k, int):
                n += 1
            elif isinstance(k, slice):
                basis.append(self.basis[i])
                direction.append(self.direction[i])
                endpoints.append(self.endpoints[i])
                n += 1
            elif k is None:
                basis.append('Cardinal')
                direction.append('z')
                endpoints.append(False)
            else: 
                raise ValueError('Polynomial error: invalid key.')
        basis = tuple(basis) + self.basis[n:]
        direction = tuple(direction) + self.direction[n:]
        endpoints = tuple(endpoints) + self.endpoints[n:]
        
        coefficients = np.array(self.coefficients[key])
        return Polynomial(coefficients, self.grid, basis, direction, endpoints)
    
    def __mul__(self, poly):
        if isinstance(poly, Polynomial):
            assert self.__is_broadcastable(self.coefficients, poly.coefficients), 'Polynomial error: the two Polynomial objects are not broadcastable.'
            basis,direction,endpoints = self.__findContraction(poly)
            return Polynomial(self.coefficients*poly.coefficients)
        else:
            newCoeff = poly*self.coefficients
            assert len(newCoeff.shape) == self.N, 'Polynomial error: the rank of the resulting Polynomial object must be the same as the original one.'
            return Polynomial(newCoeff, self.grid, self.basis, self.direction, self.endpoints)
        
    def __add__(self, poly):
        if isinstance(poly, Polynomial):
            assert self.__is_broadcastable(self.coefficients, poly.coefficients), 'Polynomial error: the two Polynomial objects are not broadcastable.'
            basis,direction,endpoints = self.__findContraction(poly)
            return Polynomial(self.coefficients+poly.coefficients)
        else:
            newCoeff = poly+self.coefficients

            ## LN: Dunno how it's possible that I get errors from here, due to taking len() of a scalar! But here's a "fix"
            newCoeff = np.asanyarray(newCoeff)
            assert newCoeff.ndim == self.N, 'Polynomial error: the rank of the resulting Polynomial object must be the same as the original one.'
            
            return Polynomial(newCoeff, self.grid, self.basis, self.direction, self.endpoints)
        
    def __sub__(self, poly):
        return self.__add__((-1)*poly)
        
    def __rmul__(self, poly):
        return self.__mul__(poly)
    def __radd__(self, poly):
        return self.__add__(poly)
    def __rsub__(self, poly):
        return (-1)*self.__sub__(poly)
    
    def __findContraction(self, poly):
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
        assert self.N == poly.N, 'Polynomial error: you can only combine two Polynomial objects with the same rank.'
        basis, endpoints, direction = [],[],[]
        for i in range(self.N):
            assert self.coefficients.shape[i] == 1 or poly.coefficients.shape[i] == 1 or (self.basis[i] == poly.basis[i] and self.direction[i] == poly.direction[i] and self.endpoints[i] == poly.endpoints[i]), 'Polynomial error: the two Polynomial objects are not broadcastable.'
            if self.coefficients.shape[i] > 1:
                basis.append(self.basis[i])
                direction.append(self.direction[i])
                endpoints.append(self.endpoints[i])
            else:
                basis.append(poly.basis[i])
                direction.append(poly.direction[i])
                endpoints.append(poly.endpoints[i])
        return tuple(basis),tuple(direction),tuple(endpoints)
        
    def changeBasis(self, newBasis):
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

        Returns
        -------
        None.

        """
        if isinstance(newBasis, str):
            newBasis = self.N*(newBasis,)
        self.__checkBasis(newBasis)
        
        for i in range(self.N):
            if newBasis[i] != self.basis[i]:
                # Choosing the appropriate x, n and restriction
                x = self.grid.getCompactCoordinates(self.endpoints[i], self.direction[i])
                n,restriction = None,None
                if self.endpoints[i]:
                    if self.direction[i] == 'z':
                        n = np.arange(self.grid.M+1)
                    elif self.direction[i] == 'pz':
                        n = np.arange(self.grid.N+1)
                    else:
                        n = np.arange(self.grid.N)
                else:
                    if self.direction[i] == 'z':
                        n = np.arange(2, self.grid.M+1)
                        restriction = 'full'
                    elif self.direction[i] == 'pz': 
                        n = np.arange(2, self.grid.N+1)
                        restriction = 'full'
                    else:
                        n = np.arange(1, self.grid.N)
                        restriction = 'partial'
                        
                # Computing the Tn matrix
                M = self.chebyshev(x[:,None], n[None,:], restriction)
                if newBasis[i] == 'Chebyshev':
                    M = np.linalg.inv(M)
                M = np.expand_dims(M, tuple(np.arange(i))+tuple(np.arange(i+2, self.N+1)))
                
                # Contracting M with self.coefficient
                self.coefficients = np.sum(M*np.expand_dims(self.coefficients, i), axis=i+1)
        self.basis = newBasis
        
    def evaluate(self, x):
        """
        Evaluates the polynomial at the compact coordinates x.

        Parameters
        ----------
        x : array-like
            Compact coordinates at which to evaluate the polynomial. Must have 
            a shape (self.N,:) or (self.N,).

        Returns
        -------
        array-like
            Values of the polynomial at x.

        """
        x = np.asarray(x)
        assert x.shape[0] == self.N and 1 <= len(x.shape) <= 2, 'Polynomial error: x must have a shape (self.N,:) or (self.N,).'
        singlePoint = False
        if len(x.shape) == 1:
            x = x.reshape((self.N,1))
            singlePoint = True
            
        polynomials = np.ones((x.shape[1],)+self.coefficients.shape)
        for i in range(self.N):
            # Choosing the appropriate n
            n = None
            if self.endpoints[i]:
                if self.direction[i] == 'z':
                    n = np.arange(self.grid.M+1)
                elif self.direction[i] == 'pz':
                    n = np.arange(self.grid.N+1)
                else:
                    n = np.arange(self.grid.N)
            else:
                if self.direction[i] == 'z':
                    n = np.arange(1, self.grid.M)
                elif self.direction[i] == 'pz':
                    n = np.arange(1, self.grid.N)
                else:
                    n = np.arange(self.grid.N-1)  
                    
            # Computing the polynomial basis in the i direction
            pn = None
            if self.basis[i] == 'Cardinal':
                pn = self.cardinal(x[i,:,None], n[None,:], self.direction[i])
                
            elif self.basis[i] == 'Chebyshev':
                restriction = None
                if not self.endpoints[i]:
                    n += 1
                    if self.direction[i] == 'z':
                        restriction = 'full'
                    elif self.direction[i] == 'pz':
                        restriction = 'full'
                    else:
                        restriction = 'partial'
                pn = self.chebyshev(x[i,:,None], n[None,:], restriction)
                
            polynomials *= np.expand_dims(pn, tuple(np.arange(1,i+1))+tuple(np.arange(i+2,self.N+1)))
            
        result = np.sum(self.coefficients[None,...]*polynomials, axis=tuple(np.arange(1,self.N+1)))
        if singlePoint:
            return result[0]
        else:
            return result
        
                        
    def cardinal(self,x,n,direction):
        r"""
        Computes the cardinal polynomials :math:`C_n(x)` defined by grid.

        Parameters
        ----------
        x : array_like
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

        x = np.asarray(x)
        n = np.asarray(n)
        
        assert self.__is_broadcastable(x, n), 'Polynomial error: x and n are not broadcastable.'
        assert direction in self.allowedDirection, "Polynomial error: unkown direction %s" % direction

        #Selecting the appropriate grid and resizing it
        grid = self.grid.getCompactCoordinates(True, direction)
        completeGrid = np.expand_dims(grid, tuple(np.arange(1,len((n*x).shape)+1)))
        nGrid = grid[n]

        #Computing all the factor in the product defining the cardinal functions
        cn_partial = np.divide(x-completeGrid, nGrid-completeGrid, where=nGrid-completeGrid!=0)

        #Multiplying all the factors to get the cardinal functions
        cn = np.prod(np.where(nGrid-completeGrid==0, 1, cn_partial),axis=0)

        return cn
    
    def chebyshev(self, x, n, restriction=None):
        r"""
        Computes the Chebyshev polynomial :math:`T_n(x)`.

        Parameters
        ----------
        x : array_like
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
        tn : array_like
            Values of the polynomial

        """

        x = np.asarray(x)
        n = np.asarray(n)
        
        assert self.__is_broadcastable(x, n), 'Polynomial error: x and n are not broadcastable.'

        #Computing the unrestricted basis
        #tn = np.cos(n*np.arccos(x))
        tn = eval_chebyt(n, x)

        #Applying the restriction
        if restriction == 'partial':
            tn -= 1
        elif restriction == 'full':
            tn -= np.where(n%2==0,1,x)

        return tn
    

    ## LN: This doesn't seem to work as intended. in EOM.action() I still got a Polynomial object as result even if axis=None
    def integrate(self, axis=None, w=None):
        r"""
        Computes the integral of the polynomial :math:`\int_{-1}^1 dx P(x)w(x)` 
        along some axis using Gauss-Chebyshev-Lobatto quadrature.

        Parameters
        ----------
        axis : None, int or tuple
            axis along which the integral is taken. Can either be None, a int or a
            tuple of int. If None, integrate along all the axes.
        w : array-like or None
            Integration weight. Must be None or an array broadcastable with 
            self.coefficients. If None, w=1. Default is None.

        Returns
        -------
        Polynomial or float
            If axis=None, returns a float. Otherwise, returns an object of the 
            class Polynomial containing the coefficients of the 
            integrated polynomial along the remaining axes. 

        """
        if w is None:
            w = 1
            
        if axis is None:
            axis = tuple(np.arange(self.N))
        if isinstance(axis, int):
            axis = (axis,)
            self.__checkAxis(axis)
        
        # Express the integrated axes in the cardinal basis
        basis = []
        for i in range(self.N):
            if i in axis:
                basis.append('Cardinal')
            else:
                basis.append(self.basis[i])
        self.changeBasis(tuple(basis))
        
        integrand = w*self.coefficients
        newBasis, newDirection, newEndpoints = [],[],[]
        for i in range(self.N):
            if i in axis:
                x = self.grid.getCompactCoordinates(self.endpoints[i], self.direction[i])
                weights = np.pi*np.ones(x.size)
                if self.direction[i] == 'z':
                    weights /= self.grid.M
                elif self.direction[i] == 'pz':
                    weights /= self.grid.N
                elif self.direction[i] == 'pp':
                    weights /= self.grid.N-1
                    if not self.endpoints[i]:
                        weights[0] /= 2
                if self.endpoints[i]:
                    weights[0] /= 2
                    weights[-1] /= 2
                integrand *= np.expand_dims(np.sqrt(1-x**2)*weights, tuple(np.arange(i))+tuple(np.arange(i+1, self.N)))
            else:
                newBasis.append(self.basis[i])
                newDirection.append(self.direction[i])
                newEndpoints.append(self.endpoints[i])
                
        result = np.sum(integrand, axis)
        if isinstance(result, float):
            return result
        else:
            return Polynomial(result, self.grid, tuple(newBasis), tuple(newDirection), tuple(newEndpoints))
    
    def derivative(self, axis):
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
        self.__checkAxis(axis)
        
        coeffDeriv = np.array(self.coefficients)
        basis, endpoints = [],[]
        
        for i in range(self.N):
            if i in axis:
                D = self.derivMatrix(self.basis[i], self.direction[i], self.endpoints[i])
                D = np.expand_dims(D, tuple(np.arange(i))+tuple(np.arange(i+2, self.N+1)))
                coeffDeriv = np.sum(D*np.expand_dims(coeffDeriv, i), axis=i+1)
                basis.append('Cardinal')
                endpoints.append(True)
            else:
                basis.append(self.basis[i])
                endpoints.append(self.endpoints[i])
        return Polynomial(coeffDeriv, self.grid, tuple(basis), self.direction, tuple(endpoints))
    
    def matrix(self, basis, direction, endpoints=False):
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
            return self.__cardinalMatrix(direction, endpoints)
        elif basis == 'Chebyshev':
            return self.__chebyshevMatrix(direction, endpoints)
        else:
            raise ValueError("basis must be either 'Cardinal' or 'Chebyshev'.")        
    
    def derivMatrix(self, basis, direction, endpoints=False):
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
            return self.__cardinalDeriv(direction,endpoints)
        elif basis == 'Chebyshev':
            return self.__chebyshevDeriv(direction,endpoints)
        else:
            raise ValueError("basis must be either 'Cardinal' or 'Chebyshev'.")
         
    def __cardinalMatrix(self, direction, endpoints=False):
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

    def __chebyshevMatrix(self, direction, endpoints=False):
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
        if direction == 'z':
            grid = self.grid.getCompactCoordinates(endpoints)[0]
            n = np.arange(grid.size)+2-2*endpoints
            restriction = 'full'
        elif direction == 'pz':
            grid = self.grid.getCompactCoordinates(endpoints)[1]
            n = np.arange(grid.size)+2-2*endpoints
            restriction = 'full'
        elif direction == 'pp':
            grid = self.grid.getCompactCoordinates(endpoints)[2]
            n = np.arange(grid.size)+1-endpoints
            restriction = 'partial'
        if endpoints:
            restriction = None

        return self.chebyshev(grid[:,None], n[None,:], restriction)
    
    def __cardinalDeriv(self, direction, endpoints=False):
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

        grid = self.grid.getCompactCoordinates(True, direction)

        #Computing the diagonal part
        diagonal = np.sum(np.where(grid[:,None]-grid[None,:] == 0, 0, np.divide(1, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0)),axis=1)

        #Computing the off-diagonal part
        offDiagonal = np.prod(np.where((grid[:,None,None]-grid[None,None,:])*(grid[None,:,None]-grid[None,None,:]) == 0, 1, np.divide(grid[None,:,None]-grid[None,None,:], grid[:,None,None]-grid[None,None,:], where=grid[:,None,None]-grid[None,None,:]!=0)),axis=-1)

        #Putting all together
        derivWithEndpoints = np.where(grid[:,None]-grid[None,:] == 0,diagonal[:,None], np.divide(offDiagonal, grid[:,None]-grid[None,:], where=grid[:,None]-grid[None,:]!=0))

        deriv = None
        if not endpoints:
            if direction == 'z' or direction == 'pz':
                deriv = derivWithEndpoints[1:-1,:]
            elif direction == 'pp':
                deriv = derivWithEndpoints[:-1,:]
        else:
            deriv = derivWithEndpoints

        return np.transpose(deriv)

    def __chebyshevDeriv(self, direction, endpoints=False):
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

        grid = self.grid.getCompactCoordinates(True, direction)
        n,restriction = None,None
        if direction == 'z':
            n = np.arange(2-2*endpoints, grid.size)
            restriction = 'full'
        elif direction == 'pz':
            n = np.arange(2-2*endpoints, grid.size)
            restriction = 'full'
        elif direction == 'pp':
            n = np.arange(1-endpoints, grid.size)
            restriction = 'partial'

        deriv = n[None,:]*eval_chebyu(n[None,:]-1,grid[:,None])

        if restriction == 'full' and not endpoints:
            deriv -= np.where(n[None,:]%2==0,0,1)

        return deriv
                
    def __checkBasis(self, basis):
        assert isinstance(basis, tuple), 'Polynomial error: basis must be a tuple or a string.'
        assert len(basis) == self.N, 'Polynomial error: basis must be a tuple of length N.'
        for x in basis:
            assert x in self.allowedBasis, "Polynomial error: unkown basis %s" % x
            
    def __checkDirection(self, direction):
        assert isinstance(direction, tuple), 'Polynomial error: direction must be a tuple or a string.'
        assert len(direction) == self.N, 'Polynomial error: direction must be a tuple of length N.'
        for x in direction:
            assert x in self.allowedDirection, "Polynomial error: unkown direction %s" % x
            
    def __checkEndpoints(self, endpoints):
        assert isinstance(endpoints, tuple), 'Polynomial error: endpoints must be a tuple or a bool.'
        assert len(endpoints) == self.N, 'Polynomial error: endpoints must be a tuple of length N.'
        for x in endpoints:
            assert isinstance(x, bool), "Polynomial error: endpoints can only contain bool."
            
    def __checkCoefficients(self, coefficients):
        for i,size in enumerate(self.coefficients.shape):
            if size > 1:
                if self.direction[i] == 'z':
                    assert size + 2*(1-self.endpoints[i]) == self.grid.M + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
                elif self.direction[i] == 'pz':
                    assert size + 2*(1-self.endpoints[i]) == self.grid.N + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
                else:
                    assert size + (1-self.endpoints[i]) == self.grid.N, f"Polynomial error: coefficients with invalid size in dimension {i}."
                
    def __checkAxis(self, axis):
        assert isinstance(axis, tuple), 'Polynomial error: axis must be a tuple or a int.'
        for x in axis:
            assert isinstance(x, int), 'Polynomial error: axis must be a tuple of int.'
            assert 0 <= x < self.N, 'Polynomial error: axis out of range.'
                
    def __is_broadcastable(self, array1, array2):
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
        for a, b in zip(np.asanyarray(array1).shape[::-1], np.asanyarray(array2).shape[::-1]):
            if a == 1 or b == 1 or a == b:
                pass
            else:
                return False
        return True
        
        
        
        