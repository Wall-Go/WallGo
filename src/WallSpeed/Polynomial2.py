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
        
    def chebyshev(self, x, n, restriction=None):
        r"""
        Computes the Chebyshev polynomial :math:`T_n(x)`

        Parameters
        ----------
        x : array_like
            Coordinate at which to evaluate the Chebyshev polynomial. Must be 
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
        
        
        
        