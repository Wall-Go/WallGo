#from WallSpeed.eomHydro import initialWallParameters
from eomHydro import initialWallParameters, wallProfile
import numpy as np
import matplotlib.pyplot as plt


class MockPotential:
    def V(self, phi, T):
        [h, s] = phi
        return -(h**2) + h**4 / 24 - s**2 / 2 + s**4 / 24 + h**2 * s**2 / 6

    def higgsVEV(self, T):
        return 2*np.sqrt(3)

    def singletVEV(self, T):
        return np.sqrt(6)

Veff = MockPotential()

higgsWidth = 1.2009453600775215
singletWidth = 1.3585904338018002
wallOffSet = 1

higgsWidth, singletWidth, wallOffSet = initialWallParameters(
    higgsWidth, singletWidth, wallOffSet, 0, Veff
)

print(higgsWidth, singletWidth, wallOffSet)

higgsVEV = Veff.higgsVEV(0)
singletVEV  = Veff.singletVEV(0)

plotRangeZ = (20+np.abs(wallOffSet))*max(higgsWidth, singletWidth)
zs = np.linspace(-plotRangeZ, plotRangeZ, 100)

fields = []
for z in zs:
    fields.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))

higgss = np.transpose(fields)[0]
singlets = np.transpose(fields)[1]

plt.plot(higgss, singlets)
plt.show()




#
