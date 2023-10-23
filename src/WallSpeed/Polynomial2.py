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
        self.N = len(self.coefficients.shape)
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
        assert x.shape[0] == self.N and 1 <= x.shape.size <= 2, 'Polynomial error: x must have a shape (self.N,:) or (self.N,).'
        singlePoint = False
        if x.shape.size == 1:
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
        completeGrid = np.expand_dims(grid, tuple(np.arange(1,(n*x).shape.size+1)))
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
            if self.direction[i] == 'z':
                assert size + 2*(1-self.endpoints[i]) == self.grid.M + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
            elif self.direction[i] == 'pz':
                assert size + 2*(1-self.endpoints[i]) == self.grid.N + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
            else:
                assert size + (1-self.endpoints[i]) == self.grid.N, f"Polynomial error: coefficients with invalid size in dimension {i}."
                
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
        
        
        
        