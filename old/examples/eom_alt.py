from scipy.integrate import quad
from scipy.optimize import minimize, brentq, root
import numpy as np
import matplotlib.pyplot as plt


def initialWallParameters(
    higgsWidthGuess, singletWidthGuess, wallOffSetGuess, TGuess, Veff
):
    higgsVEV = Veff.higgsVEV(TGuess)
    singletVEV = Veff.singletVEV(TGuess)

    initRes = minimize(
        lambda wallParams: oneDimAction(higgsVEV, singletVEV, wallParams, TGuess, Veff),
        x0=[higgsWidthGuess, singletWidthGuess, wallOffSetGuess],
        bounds=[(0, None), (0, None), (-10, 10)],
    )

    return initRes.x[0], initRes.x[1], initRes.x[2]

def oneDimAction(higgsVEV, singletVEV, wallParams, T, Veff):
    [higgsWidth, singletWidth, wallOffSet] = wallParams

    kinetic = (higgsVEV**2 / higgsWidth + singletVEV**2 / singletWidth) * 3 / 2

    integrationLength = (20 + np.abs(wallOffSet)) * max(higgsWidth, singletWidth)

    integral = quad(
        lambda z: Veff.V(
            wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z),
            T,
        ),
        -integrationLength,
        integrationLength,
    )

    potential = integral[0] - integrationLength * (
        Veff.V([higgsVEV, 0], T) + Veff.V([0, singletVEV], T)
    )

    #print(higgsWidth, singletWidth, wallOffSet)

    #print(kinetic + potential)

    return kinetic + potential

def wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z):
    h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
    s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))

    return [h, s]

def wallProfileOnWallGoGrid(staticWallParams, Tplus, Tminus, grid):
    [higgsWidth, singletWidth, wallOffSet] = staticWallParams

    higgsVEV = Veff.higgsVEV(Tminus)
    singletVEV = Veff.singletVEV(Tplus)

    wallProfileGrid = []
    for z in grid.xiValues:
        wallProfileGrid.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))

    return wallProfileGrid


class MockPotential:
    def V(self, phi, T):
        [h, s] = phi
        return -(h**2) + h**4 / 24 - s**2 / 2 + s**4 / 24 + h**2 * s**2 / 4
        return -(h**2)/2 + h**4 / 24 - s**2 / 2 + s**4 / 24 + h**2 * s**2 / 6

    def higgsVEV(self, T):
        return 2*np.sqrt(3)
        return np.sqrt(6)

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
zs = np.linspace(-plotRangeZ, plotRangeZ, 1000)

fields = []
fields0 = []
fieldsM = []
for z in zs:
    fields.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))
    fields0.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, 0, z))
    fieldsM.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, -wallOffSet, z))

higgss = np.transpose(fields)[0]
singlets = np.transpose(fields)[1]

plt.plot(zs, Veff.V(np.transpose(fields), 0), label='Minimum')
plt.plot(zs, Veff.V(np.transpose(fields0), 0), label='No offset')
plt.plot(zs, Veff.V(np.transpose(fieldsM), 0), label='Opposite offset')
plt.xlim(-8,8)
plt.ylabel("V")
plt.xlabel("z")
plt.legend()
plt.show()

plt.plot(np.transpose(fields)[0], np.transpose(fields)[1], label='Minimum')
plt.plot(np.transpose(fields0)[0], np.transpose(fields0)[1], label='No offset')
plt.plot(np.transpose(fieldsM)[0], np.transpose(fieldsM)[1], label='Opposite offset')
plt.xlabel("h")
plt.ylabel("s")
plt.legend()
plt.show()

plt.plot(zs, higgss)
plt.plot(zs, singlets)

plt.show()


#
