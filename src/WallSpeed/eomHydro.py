import numpy as np

def findWallVelocityLoop(Model,TNucl, wallVelocityLTE, hMass, sMass, errTol, thermalPotential, grid):

    # Initial conditions

    higgsWidth = 1/hMass
    singletWidth = 1/sMass
    wallOffSet = 0

    if wallVelocityLTE is not None:
        wallVelocity = 0.9 * wallVelocityLTE
    else:
        wallVelocity = np.sqrt(1/3)

    outOffEquilDeltas = 0

    c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)
    Tprofile = findTemperatureProfile(c1, c2, higgsWidth, singletWidth, wallOffSet, offEquilDeltas, Tplus, Tminus, thermalPotential, grid)

    error = errTol + 1
    while error > errTol:

        oldHiggsWidth = higgsWidth
        oldSingletWidth = singletWidth
        oldWallOffSet = wallOffSet
        oldWallVelocity = wallVelocity

        outOffEquilDeltas = solveBoltzmannEquation(Tprofile, higgsWidth, singletWidth, wallOffSet)

        c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)
        Tprofile = findTemperatureProfile(c1, c2, higgsWidth, singletWidth, wallOffSet, offEquilDeltas, Tplus, Tminus, thermalPotential, grid)

        wallVelocity, higgsWidth, singletWidth, wallOffSet = solveWallEoM()

        error = np.sqrt(((wallVelocity-oldWallVelocity)/wallVelocity)**2 + ((higgsWidth-oldHiggsWidth)/higgsWidth)**2 + ((singletWidth-oldSingletWidth)/singletWidth)**2 + (wallOffSet-oldWallOffSet)**2)

    return wallVelocity, higgsWidth, singletWidth, wallOffSet


def findTemperatureProfile(c1, c2, higgsWidth, singletWidth, wallOffSet, offEquilDeltas, Tplus, Tminus, Veff, grid):
    findTemperatureProfile = []
    for z in grid.xiValues:
        h = 0.5 * Veff.higgsVEV(Tminus) * (1 - np.tanh(z/higgsWidth))
        dhdz = -0.5 * Veff.higgsVEV(Tminus) / (higgsWidth * np.cosh(z/higgsWidth)**2)

        s = 0.5 * Veff.singletVEV(Tplus) * (1 + np.tanh(z/singletWidth + wallOffSet))
        dsdh = 0.5 * Veff.singletVEV(Tplus) / (singletWidth * np.cosh(z/singletWidth + wallOffSet)**2)

        T = findTemperaturePoint(c1, c2, Veff, h, dhdz, s, dsdz)

    return np.array(findTemperatureProfile)


def findTemperaturePoint(c1, c2, Veff, h, dhdz, s, dsdz):

    return None
