import numpy as np

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
            dimensions are assumed to be the same. The default is False.

        Returns
        -------
        None.

        """
        
        self.coefficients = np.asanyarray(coefficients)
        self.N = len(self.coefficients.shape)
        self.grid = grid
        
        self.allowedBasis = ['Cardinal','Chebyshev']
        self.allowedDirection = ['z','pz','pp']
        
        # Check that basis, direction and endpoints have the right type.
        if isinstance(basis, str):
            basis = self.N*(basis,)
        assert isinstance(basis, tuple), 'Polynomial error: basis must be a tuple or a string.'
        assert len(basis) == self.N, 'Polynomial error: basis must be a tuple of length N.'
        for x in basis:
            assert x in self.allowedBasis, "Polynomial error: unkown basis %s" % x
            
        if isinstance(direction, str):
            direction = self.N*(direction,)
        assert isinstance(direction, tuple), 'Polynomial error: direction must be a tuple or a string.'
        assert len(direction) == self.N, 'Polynomial error: direction must be a tuple of length N.'
        for x in direction:
            assert x in self.allowedDirection, "Polynomial error: unkown direction %s" % x
            
        if isinstance(endpoints, bool):
            endpoints = self.N*(endpoints,)
        assert isinstance(endpoints, tuple), 'Polynomial error: endpoints must be a tuple or a bool.'
        assert len(endpoints) == self.N, 'Polynomial error: endpoints must be a tuple of length N.'
        for x in endpoints:
            assert isinstance(x, bool), "Polynomial error: endpoints can only contain bool."
            
        self.basis = basis
        self.direction = direction
        self.endpoints = endpoints
        
        # Check that each dimension of coefficients has the right size.
        for i,size in enumerate(self.coefficients.shape):
            if self.direction[i] == 'z':
                assert size + 2*(1-endpoints[i]) == self.grid.M + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
            elif self.direction[i] == 'pz':
                assert size + 2*(1-endpoints[i]) == self.grid.N + 1, f"Polynomial error: coefficients with invalid size in dimension {i}."
            else:
                assert size + (1-endpoints[i]) == self.grid.N, f"Polynomial error: coefficients with invalid size in dimension {i}."
        
        
        
        