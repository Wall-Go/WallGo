from WallSpeed.eomHydro import initialWallParameters
import numpy as np



def VeffFunc(h, s, T):
    return -(h**2) + h**4 / 24 - s**2 / 2 + s**4 / 24 + h**2 * s**2 / 4


class MockPotential:
    def Veff(h, s, T):
        return -(h**2) + h**4 / 24 - s**2 / 2 + s**4 / 24 + h**2 * s**2 / 4

    def higgsVEV(T):
        return 2*np.sqrt(3)

    def singletVEV(T):
        return np.sqrt(6)

Veff = MockPotential()

higgsWidth, singletWidth, wallOffSet = initialWallParameters(
    higgsWidth, singletWidth, wallOffSet, 0, Veff
)

print(higgsWidth, singletWidth, wallOffSet)
