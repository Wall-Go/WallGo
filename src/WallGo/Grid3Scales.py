import numpy as np
from .Grid import Grid

class Grid3Scales(Grid):
    r"""
    Redefinition of the Grid class to take into account the different scales present in the z direction. 
    More specifically, the z mapping function should scale as :math:`\lambda_- \log(1+\chi)` 
    when :math:`\chi\to -1` and :math:`-\lambda_+ \log(1-\chi)` when :math:`\chi\to 1`,
    where :math:`\lambda_-` and :math:`\lambda_+` are the lengths of the solution's 
    tails inside and outside the bubble, respectively. Furthermore, the mapping 
    should be approximately linear in the region :math:`-r<\chi<r`, where :math:`r`
    is roughly the ratio of points that are used to resolve the wall's interior.
    The slope in that region should be :math:`L/R`, where :math:`L` is the wall thickness.
    
    It is easier to find the derivative of a function that has these properties, 
    and then integrate it. We choose here :math:`z'(\chi)=\frac{f(\chi)}{1-\chi^2}`,
    where :math:`f(\chi)` is a smoothed step function equal to :math:`\lambda_-`
    when :math:`\chi<-r`, :math:`L/r` when :math:`-r<\chi<r` and :math:`\lambda_+`
    when :math:`\chi>r`. We choose :math:`f(\chi)` to be a sum of functions like
    :math:`\frac{\chi-\chi_0}{\sqrt{a^2+(\chi-\chi_0)^2}}`, which allows us to find
    analytically the mapping function with :math:`z(\chi)=\int d\chi\ z'(\chi)`.
    The parameter :math:`a` can be adjusted to control the smoothness of the mapping
    function.
    """
    
    def __init__(self, M: int, N: int, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, momentumFalloffT: float, ratioPointsWall: float=0.5, smoothing: float=0.1, wallCenter: float=0, spacing: str="Spectral"):
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
            than wallThickness*(1+2*smoothing)/ratioPointsWall
        tailLengthOutside : float
            Decay length of the solution's tail outside the wall. Should be larger
            than wallThickness*(1+2*smoothing)/ratioPointsWall
        wallThickness : float
            Thickness of the wall.
        momentumFalloffT : float
            Temperature scale determining transform in momentum directions. Should be close to the plasma temperature.
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
        self._updateParameters(tailLengthInside, tailLengthOutside, wallThickness, ratioPointsWall, smoothing, wallCenter)
        
        super().__init__(M, N, wallThickness, momentumFalloffT, spacing)
        
    def changePositionFalloffScale(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, wallCenter: float) -> None:
        self._updateParameters(tailLengthInside, tailLengthOutside, wallThickness, self.ratioPointsWall, self.smoothing, wallCenter)
        
        self._cacheCoordinates()
        
    def _updateParameters(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, ratioPointsWall: float, smoothing: float, wallCenter: float) -> None:
        assert wallThickness > 0, "Grid3Scales error: wallThickness must be positive."
        assert smoothing >= 0, "Grid3Scales error: smoothness must be positive."
        assert tailLengthInside > wallThickness*(1+2*smoothing)/ratioPointsWall, "Grid3Scales error: tailLengthInside must be greater than wallThickness*(1+2*smoothness)/ratioPointsWall."
        assert tailLengthOutside > wallThickness*(1+2*smoothing)/ratioPointsWall, "Grid3Scales error: tailLengthOutside must be greater than wallThickness*(1+2*smoothness)/ratioPointsWall."
        assert 0 <= ratioPointsWall <= 1, "Grid3Scales error: ratioPointsWall must be between 0 and 1."
        
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
        self.aIn = np.sqrt(4*smoothing*wallThickness*ratioPointsWall**2*(ratioPointsWall*tailLengthInside-wallThickness*(1+smoothing)))/abs(ratioPointsWall*tailLengthInside-wallThickness*(1+2*smoothing)) + 1e-50
        self.aOut = np.sqrt(4*smoothing*wallThickness*ratioPointsWall**2*(ratioPointsWall*tailLengthOutside-wallThickness*(1+smoothing)))/abs(ratioPointsWall*tailLengthOutside-wallThickness*(1+2*smoothing)) + 1e-50
        
    def decompactify(self, z_compact, pz_compact, pp_compact):
        r"""
        Transforms coordinates from [-1, 1] interval (inverse of compactify).
        """
        L = self.wallThickness
        r = self.ratioPointsWall
        tailIn = self.tailLengthInside
        tailOut = self.tailLengthOutside
        aIn = self.aIn 
        aOut = self.aOut
        
        term1 = lambda x: (1-r)*(r*tailOut-L)*np.arctanh((1-x+np.sqrt(aOut**2+(x-r)**2))/np.sqrt(aOut**2+(1-r)**2)+0j).real/np.sqrt(aOut**2+(1-r)**2)/r
        term2 = lambda x: -(1+r)*(r*tailOut-L)*np.arctanh((1+x-np.sqrt(aOut**2+(x-r)**2))/np.sqrt(aOut**2+(1+r)**2)+0j).real/np.sqrt(aOut**2+(1+r)**2)/r
        term3 = lambda x: (1-r)*(r*tailIn-L)*np.arctanh((1+x-np.sqrt(aIn**2+(x+r)**2))/np.sqrt(aIn**2+(1-r)**2)+0j).real/np.sqrt(aIn**2+(1-r)**2)/r
        term4 = lambda x: -(1+r)*(r*tailIn-L)*np.arctanh((1-x+np.sqrt(aIn**2+(x+r)**2))/np.sqrt(aIn**2+(1+r)**2)+0j).real/np.sqrt(aIn**2+(1+r)**2)/r
        term5 = lambda x: (tailIn+tailOut-4*self.smoothing*L/r)*np.arctanh(x)
        totalMapping = lambda x: (term1(x)+term2(x)+term3(x)+term4(x)+term5(x))/2
        
        z = totalMapping(z_compact) - totalMapping(0) + self.wallCenter
        pz = 2 * self.momentumFalloffT * np.arctanh(pz_compact)
        pp = -self.momentumFalloffT * np.log((1 - pp_compact) / 2)
        
        return z, pz, pp
    
    def compactificationDerivatives(self, z_compact, pz_compact, pp_compact):
        r"""
        Derivative of transforms coordinates to [-1, 1] interval
        """
        L = self.wallThickness
        r = self.ratioPointsWall
        tailIn = self.tailLengthInside
        tailOut = self.tailLengthOutside
        aIn = self.aIn 
        aOut = self.aOut
        
        dzdzCompact = (tailIn-L/r)*(1-(z_compact+r)/np.sqrt(aIn**2+(z_compact+r)**2))/2
        dzdzCompact += (tailOut-L/r)*(1+(z_compact-r)/np.sqrt(aOut**2+(z_compact-r)**2))/2
        dzdzCompact += (1-2*self.smoothing)*L/r
        dzdzCompact /= 1-z_compact**2        
        
        dpzdpzCompact = 2 * self.momentumFalloffT / (1-pz_compact**2)
        dppdppCompact = self.momentumFalloffT / (1-pp_compact)
        return dzdzCompact, dpzdpzCompact, dppdppCompact

    # def decompactify(self, z_compact, pz_compact, pp_compact):
    #     r"""
    #     Transforms coordinates from [-1, 1] interval (inverse of compactify).
    #     """
    #     z = (1-2*self.smoothness)*self.wallThickness/self.ratioPointsWall # Center of the wall
    #     z += (self.tailLengthInside-self.wallThickness/self.ratioPointsWall)*(1-np.tanh(self.aIn*(z_compact+self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail inside the wall
    #     z += (self.tailLengthOutside-self.wallThickness/self.ratioPointsWall)*(1+np.tanh(self.aOut*(z_compact-self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail outside the wall
    #     z *= np.arctanh(z_compact)
        
    #     pz = 2 * self.momentumFalloffT * np.arctanh(pz_compact)
    #     pp = -self.momentumFalloffT * np.log((1 - pp_compact) / 2)
    #     return z, pz, pp

    # def compactificationDerivatives(self, z_compact, pz_compact, pp_compact):
    #     r"""
    #     Derivative of transforms coordinates to [-1, 1] interval
    #     """
    #     dz1 = (1-2*self.smoothness)*self.wallThickness/self.ratioPointsWall # Center of the wall
    #     dz1 += (self.tailLengthInside-self.wallThickness/self.ratioPointsWall)*(1-np.tanh(self.aIn*(z_compact+self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail inside the wall
    #     dz1 += (self.tailLengthOutside-self.wallThickness/self.ratioPointsWall)*(1+np.tanh(self.aOut*(z_compact-self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail outside the wall
    #     dz1 /= 1-z_compact**2
        
    #     dz2 = (1+2*self.ratioPointsWall*z_compact+z_compact**2)*self.aIn*(self.wallThickness-self.ratioPointsWall*self.tailLengthInside)/2/(self.ratioPointsWall*(1-z_compact**2)*np.cosh(self.aIn*(z_compact+self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))**2
    #     dz2 += (-1+2*self.ratioPointsWall*z_compact-z_compact**2)*self.aOut*(self.wallThickness-self.ratioPointsWall*self.tailLengthOutside)/2/(self.ratioPointsWall*(1-z_compact**2)*np.cosh(self.aOut*(z_compact-self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))**2
    #     dz2 *= np.arctanh(z_compact)
        
    #     dzdzCompact = dz1 + dz2
    #     dpzdpzCompact = 2 * self.momentumFalloffT / (1-pz_compact**2)
    #     dppdppCompact = self.momentumFalloffT / (1-pp_compact)
    #     return dzdzCompact, dpzdpzCompact, dppdppCompact
