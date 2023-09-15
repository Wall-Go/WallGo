import numpy as np
from .EOM import EOM

class EOMGeneralShape(EOM):
    def __init__(self, particle, freeEnergy, grid, nbrFields, errTol=1e-6):
        super().__init__(particle, freeEnergy, grid, nbrFields, errTol)
        
    def findWallVelocity(self):
        pass
    
    def action(self, deltaShape, vevLowT, vevHighT, Tprofile, offEquilDeltas):
        pass
    
    def wallProfile(self, z, deltaShape, vevLowT, vevHighT):
        pass