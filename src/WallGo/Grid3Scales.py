import numpy as np
from WallGo import Grid

class Grid3Scales(Grid):
    def __init__(self, M: int, N: int, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, momentumFalloffT: float, ratioPointsWall: float=0.5, smoothness: float=0.1, spacing: str="Spectral"):
        assert wallThickness > 0, "Grid3Scales error: wallThickness must be positive."
        assert tailLengthInside > wallThickness, "Grid3Scales error: tailLengthInside must be greater than wallThickness."
        assert tailLengthOutside > wallThickness, "Grid3Scales error: tailLengthOutside must be greater than wallThickness."
        assert 0 < ratioPointsWall < 1, "Grid3Scales error: ratioPointsWall must be between 0 and 1."
        
        self._updateParameters(tailLengthInside, tailLengthOutside, wallThickness, ratioPointsWall, smoothness)
        
        super().__init__(M, N, wallThickness, momentumFalloffT, spacing)
        
    def changePositionFalloffScale(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, ratioPointsWall: float=0.5, smoothing: float=0.1) -> None:
        self._updateParameters(tailLengthInside, tailLengthOutside, wallThickness, ratioPointsWall, smoothing)
        
        self._cacheCoordinates()
        
    def _updateParameters(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, ratioPointsWall: float=0.5, smoothing: float=0.1) -> None:
        assert wallThickness > 0, "Grid3Scales error: wallThickness must be positive."
        assert smoothing > 0, "Grid3Scales error: smoothness must be positive."
        assert tailLengthInside > wallThickness*(1+2*smoothing)/ratioPointsWall, "Grid3Scales error: tailLengthInside must be greater than wallThickness*(1+2*smoothness)/ratioPointsWall."
        assert tailLengthOutside > wallThickness*(1+2*smoothing)/ratioPointsWall, "Grid3Scales error: tailLengthOutside must be greater than wallThickness*(1+2*smoothness)/ratioPointsWall."
        assert 0 < ratioPointsWall < 1, "Grid3Scales error: ratioPointsWall must be between 0 and 1."
        
        self.tailLengthInside = tailLengthInside
        self.tailLengthOutside = tailLengthOutside
        self.wallThickness = wallThickness
        self.ratioPointsWall = ratioPointsWall
        self.smoothing = smoothing
        
        # Defining parameters used in the mapping functions
        self.aIn = np.sqrt(4*smoothing*wallThickness*ratioPointsWall**2*(ratioPointsWall*tailLengthInside-wallThickness*(1+smoothing)))/(ratioPointsWall*tailLengthInside-wallThickness*(1+2*smoothing))
        self.aOut = np.sqrt(4*smoothing*wallThickness*ratioPointsWall**2*(ratioPointsWall*tailLengthOutside-wallThickness*(1+smoothing)))/(ratioPointsWall*tailLengthOutside-wallThickness*(1+2*smoothing))
        # self.aIn = np.arctanh((ratioPointsWall*tailLengthInside/wallThickness-1-2*smoothness)/(ratioPointsWall*tailLengthInside/wallThickness-1))
        # self.aOut = np.arctanh((ratioPointsWall*tailLengthOutside/wallThickness-1-2*smoothness)/(ratioPointsWall*tailLengthOutside/wallThickness-1))
        
    def decompactify(self, z_compact, pz_compact, pp_compact):
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
        
        z = totalMapping(z_compact) - totalMapping(0)
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