import numpy as np
from scipy.optimize import minimize, brentq


def findWallVelocityLoop(
    Model, TNucl, wallVelocityLTE, hMass, sMass, errTol, thermalPotential, grid
):

    # Initial conditions

    higgsWidth = 1 / hMass
    singletWidth = 1 / sMass
    wallOffSet = 0

    if wallVelocityLTE is not None:
        wallVelocity = 0.9 * wallVelocityLTE
    else:
        wallVelocity = np.sqrt(1 / 3)

    outOffEquilDeltas = 0

    c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)
    Tprofile = findTemperatureProfile(
        c1,
        c2,
        higgsWidth,
        singletWidth,
        wallOffSet,
        outOffEquilDeltas,
        Tplus,
        Tminus,
        thermalPotential,
        grid,
    )

    error = errTol + 1
    while error > errTol:

        oldHiggsWidth = higgsWidth
        oldSingletWidth = singletWidth
        oldWallOffSet = wallOffSet
        oldWallVelocity = wallVelocity

        outOffEquilDeltas = solveBoltzmannEquation(
            Tprofile, higgsWidth, singletWidth, wallOffSet
        )

        c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)
        Tprofile = findTemperatureProfile(
            c1,
            c2,
            higgsWidth,
            singletWidth,
            wallOffSet,
            outOffEquilDeltas,
            Tplus,
            Tminus,
            thermalPotential,
            grid,
        )

        wallVelocity, higgsWidth, singletWidth, wallOffSet = solveWallEoM()

        error = np.sqrt(
            ((wallVelocity - oldWallVelocity) / wallVelocity) ** 2
            + ((higgsWidth - oldHiggsWidth) / higgsWidth) ** 2
            + ((singletWidth - oldSingletWidth) / singletWidth) ** 2
            + (wallOffSet - oldWallOffSet) ** 2
        )

    return wallVelocity, higgsWidth, singletWidth, wallOffSet


def findTemperatureProfile(
    c1,
    c2,
    higgsWidth,
    singletWidth,
    wallOffSet,
    outOffEquilDeltas,
    Tplus,
    Tminus,
    Veff,
    grid,
):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of LHS.
    """
    findTemperatureProfile = []
    for z in grid.xiValues:
        h = 0.5 * Veff.higgsVEV(Tminus) * (1 - np.tanh(z / higgsWidth))
        dhdz = (
            -0.5 * Veff.higgsVEV(Tminus) / (higgsWidth * np.cosh(z / higgsWidth) ** 2)
        )

        s = 0.5 * Veff.singletVEV(Tplus) * (1 + np.tanh(z / singletWidth + wallOffSet))
        dsdh = (
            0.5
            * Veff.singletVEV(Tplus)
            / (singletWidth * np.cosh(z / singletWidth + wallOffSet) ** 2)
        )

        T = findTemperaturePoint(
            c1, c2, Veff, h, dhdz, s, dsdz, outOffEquilDeltas, Tplus, Tminus
        )

    return np.array(findTemperatureProfile)


def findTemperaturePoint(
    c1, c2, Veff, h, dhdz, s, dsdz, outOffEquilDeltas, Tplus, Tminus
):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.
    """

    s1 = c1 - outOffEquilDeltas[0, 3]
    s2 = c2 - outOffEquilDeltas[3, 3]

    Tavg = 0.5 * (Tplus + Tminus)

    minRes = minimize(
        lambda T: temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, Veff),
        Tavg,
        bounds=(0, None),
        tol=1e-9,
    )
    # TODO: A fail safe

    if temperatureProfileEqLHS(h, s, dhdz, dsdz, minRes.x, s1, s2, Veff) >= 0:
        return minRes.x

    TLowerBound = minRes.x
    TStep = np.abs(Tplus - TLowerBound)
    if TStep == 0:
        TStep = np.abs(Tminus - TLowerBound)

    TUpperBound = TLowerBound + TStep
    while temperatureProfileEqLHS(h, s, dhdz, dsdz, TUpperBound, s1, s2, Veff) < 0:
        TStep *= 2
        TUpperBound = TLowerBound + TStep

    res = brentq(
        lambda T: temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, Veff),
        TLowerBound,
        TUpperBound,
        xtol=1e-9,
        rtol=1e-9,
    )
    # TODO: Can the function have multiple zeros?

    return res.x


def temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, Veff):
    """
    The LHS of Eq. (20) of arXiv:2204.13120v1
    """
    return (
        0.5 * (dhdz**2 + dsdz**2)
        - Veff.V(h, s, T)
        - 0.5 * Veff.enthalpy(h, s, T)
        + 0.5 * np.sqrt(4 * s1**2 + Veff.enthalpy(h, s, T) ** 2)
        - s2
    )
