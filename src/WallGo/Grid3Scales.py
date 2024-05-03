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
        
    def changePositionFalloffScale(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, ratioPointsWall: float=0.5, smoothness: float=0.1) -> None:
        self._updateParameters(tailLengthInside, tailLengthOutside, wallThickness, ratioPointsWall, smoothness)
        
        self._cacheCoordinates()
        
    def _updateParameters(self, tailLengthInside: float, tailLengthOutside: float, wallThickness: float, ratioPointsWall: float=0.5, smoothness: float=0.1) -> None:
        assert wallThickness > 0, "Grid3Scales error: wallThickness must be positive."
        assert tailLengthInside > wallThickness, "Grid3Scales error: tailLengthInside must be greater than wallThickness."
        assert tailLengthOutside > wallThickness, "Grid3Scales error: tailLengthOutside must be greater than wallThickness."
        assert 0 < ratioPointsWall < 1, "Grid3Scales error: ratioPointsWall must be between 0 and 1."
        
        self.tailLengthInside = tailLengthInside
        self.tailLengthOutside = tailLengthOutside
        self.wallThickness = wallThickness
        self.ratioPointsWall = ratioPointsWall
        self.smoothness = smoothness
        
        # Defining parameters used in the mapping functions
        self.aIn = np.arctanh((ratioPointsWall*tailLengthInside/wallThickness-1-2*smoothness)/(ratioPointsWall*tailLengthInside/wallThickness-1))
        self.aOut = np.arctanh((ratioPointsWall*tailLengthOutside/wallThickness-1-2*smoothness)/(ratioPointsWall*tailLengthOutside/wallThickness-1))

    def decompactify(self, z_compact, pz_compact, pp_compact):
        r"""
        Transforms coordinates from [-1, 1] interval (inverse of compactify).
        """
        z = (1-2*self.smoothness)*self.wallThickness/self.ratioPointsWall # Center of the wall
        z += (self.tailLengthInside-self.wallThickness/self.ratioPointsWall)*(1-np.tanh(self.aIn*(z_compact+self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail inside the wall
        z += (self.tailLengthOutside-self.wallThickness/self.ratioPointsWall)*(1+np.tanh(self.aOut*(z_compact-self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail outside the wall
        z *= np.arctanh(z_compact)
        
        pz = 2 * self.momentumFalloffT * np.arctanh(pz_compact)
        pp = -self.momentumFalloffT * np.log((1 - pp_compact) / 2)
        return z, pz, pp

    def compactificationDerivatives(self, z_compact, pz_compact, pp_compact):
        r"""
        Derivative of transforms coordinates to [-1, 1] interval
        """
        dz1 = (1-2*self.smoothness)*self.wallThickness/self.ratioPointsWall # Center of the wall
        dz1 += (self.tailLengthInside-self.wallThickness/self.ratioPointsWall)*(1-np.tanh(self.aIn*(z_compact+self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail inside the wall
        dz1 += (self.tailLengthOutside-self.wallThickness/self.ratioPointsWall)*(1+np.tanh(self.aOut*(z_compact-self.ratioPointsWall)/self.ratioPointsWall/(1-z_compact**2)))/2 # Tail outside the wall
        dz1 /= 1-z_compact**2
        
        dz_compact = self.L_xi**2 / (self.L_xi**2 + z**2)**1.5
        dpz_compact = 1 / 2 / self.momentumFalloffT / np.cosh(pz / 2 / self.momentumFalloffT)**2
        dpp_compact = 2 / self.momentumFalloffT * np.exp(-pp / self.momentumFalloffT)
        return dz_compact, dpz_compact, dpp_compact