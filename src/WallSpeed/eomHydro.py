import numpy as np
from scipy.optimize import fsolve

def findWallVelocityLoop(Model,TNucl, wallVelocityLTE, hMass, sMass, errTol):

    # Initial conditions

    higgsWidth = 1/hMass
    singletWidth = 1/sMass
    wallOffSet = 0

    if wallVelocityLTE is not None:
        wallVelocity = 0.9 * wallVelocityLTE
    else:
        wallVelocity = np.sqrt(1/3)

    outOffEquilDeltas = 0

    c1, c2 = findHydroBoundaries(TNucl, wallVelocity)
    Tprofile = findTemperatureProfile(c1, c2, higgsWidth, singletWidth, wallOffSet, offEquilDeltas)

    error = errTol + 1
    while error > errTol:

        oldHiggsWidth = higgsWidth
        oldSingletWidth = singletWidth
        oldWallOffSet = wallOffSet
        oldWallVelocity = wallVelocity

        outOffEquilDeltas = solveBoltzmannEquation(Tprofile, higgsWidth, singletWidth, wallOffSet)

        c1, c2 = findHydroBoundaries(TNucl, wallVelocity)
        Tprofile = findTemperatureProfile(c1, c2, higgsWidth, singletWidth, wallOffSet, offEquilDeltas)

        wallVelocity, higgsWidth, singletWidth, wallOffSet = solveWallEoM()

        error = np.sqrt(((wallVelocity-oldWallVelocity)/wallVelocity)**2 + ((higgsWidth-oldHiggsWidth)/higgsWidth)**2 + ((singletWidth-oldSingletWidth)/singletWidth)**2 + (wallOffSet-oldWallOffSet)**2)

    return wallVelocity, higgsWidth, singletWidth, wallOffSet
        
        
