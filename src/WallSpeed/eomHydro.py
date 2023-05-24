import numpy as np

from scipy.optimize import minimize, brentq, fsolve
from scipy.integrate import quad


def findWallVelocityLoop(model, TNucl, wallVelocityLTE, hMass, sMass, errTol, grid):
    """
    Finds the wall velocity by solving hydrodynamics, the Boltzmann equation and
    the field equation of motion iteratively.
    """

    # Initial conditions for velocity, hydro boundaries, wall parameters and
    # temperature profile

    if wallVelocityLTE is not None:
        wallVelocity = 0.9 * wallVelocityLTE
        maxWallVelocity = wallVelocityLTE
    else:
        wallVelocity = np.sqrt(1 / 3)
        maxWallVelocity = findJouguetVelocity(model, TNucl)

    offEquilDeltas = 0

    c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)

    higgsWidth = 1 / hMass
    singletWidth = 1 / sMass
    wallOffSet = 0
    higgsWidth, singletWidth, wallOffSet = initialWallParameters(
        higgsWidth, singletWidth, wallOffSet, 0.5 * (Tplus + Tminus), Veff
    )

    initializedWallParameters = [wallVelocity, higgsWidth, singletWidth, wallOffSet]

    wallParameters = [wallVelocity, higgsWidth, singletWidth, wallOffSet]
    error = errTol + 1
    while error > errTol:

        oldWallVelocity = wallParameters[0]
        oldHiggsWidth = wallParameters[1]
        oldSingletWidth = wallParameters[2]
        oldWallOffSet = wallParameters[3]
        oldError = error

        c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallVelocity)

        Tprofile = findTemperatureProfile(
            c1,
            c2,
            higgsWidth,
            singletWidth,
            wallOffSet,
            offEquilDeltas,
            Tplus,
            Tminus,
            model,
            grid,
        )

        offEquilDeltas = solveBoltzmannEquation(
            Tprofile, wallParameters[1], wallParameters[2], wallParameters[3]
        )

        intermediateRes = fsolve(
            momentsOfWallEoM, wallParameters, args=(offEquilDeltas, model)
        )

        wallParameters = intermediateRes.x

        c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallParameters[0])
        Tprofile = findTemperatureProfile(
            c1,
            c2,
            higgsWidth,
            singletWidth,
            wallOffSet,
            offEquilDeltas,
            Tplus,
            Tminus,
            model,
            grid,
        )

        error = np.sqrt(
            ((wallVelocity - oldWallVelocity) / wallVelocity) ** 2
            + ((higgsWidth - oldHiggsWidth) / higgsWidth) ** 2
            + ((singletWidth - oldSingletWidth) / singletWidth) ** 2
            + (wallOffSet - oldWallOffSet) ** 2
        )

    return wallVelocity, higgsWidth, singletWidth, wallOffSet


def momentsOfWallEoM(wallParameters, offEquilDeltas, Veff):
    c1, c2, Tplus, Tminus = findHydroBoundaries(TNucl, wallParameters[0])
    Tprofile = findTemperatureProfile(
        c1,
        c2,
        higgsWidth,
        singletWidth,
        wallOffSet,
        offEquilDeltas,
        Tplus,
        Tminus,
        model,
        grid,
    )

    higgsVEV = Veff.higgsVEV(Tminus)
    singletVEV = Veff.singletVEV(Tplus)

    mom1 = higgsPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        Veff,
        offEquilDeltas,
        Tfunc,
    )
    mom2 = higgsStretchMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        Veff,
        offEquilDeltas,
        Tfunc,
    )
    mom3 = singletPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        Veff,
        offEquilDeltas,
        Tfunc,
    )
    mom4 = singletPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        Veff,
        offEquilDeltas,
        Tfunc,
    )

    return [mom1, mom2, mom3, mom4]


def higgsPressureMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    Veff,
    offEquilDeltas,
    Tfunc,
):
    return quad(
        lambda z: higgsPressureLocal(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * higgsWidth,
        -20 * singletWidth,
    )


def higgsPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    dhdz = -0.5 * higgsVEV / (higgsWidth * np.cosh(z / higgsWidth) ** 2)
    return -dhdz * higgsEquationOfMotion(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        z,
        Veff,
        offEquilDeltas,
        T,
    )


def higgsStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    Veff,
    offEquilDeltas,
    Tfunc,
):
    return quad(
        lambda z: higgsStretchLocal(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * higgsWidth,
        -20 * singletWidth,
    )


def higgsStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    dhdz = -0.5 * higgsVEV / (higgsWidth * np.cosh(z / higgsWidth) ** 2)
    offCenterWeight = -np.tanh(z / higgsWidth)
    return (
        dhdz
        * offCenterWeight
        * higgsEquationOfMotion(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            T,
        )
    )


def higgsEquationOfMotion(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    zLHiggs = z / higgsWidth
    kinetic = -higgsVEV * np.tanh(zLHiggs) / (higgsWidth * np.cosh(zLHiggs)) ** 2
    potential = Veff.higgsDerivative(
        wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z), T
    )
    offEquil = (
        0.5
        * 12
        * Veff.dTopMassdh(
            wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z),
            T,
        )
        * offEquilDeltas[0, 0]
    )
    return kinetic + potential + offEquil


def singletPressureMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    Veff,
    offEquilDeltas,
    Tfunc,
):
    return quad(
        lambda z: singletPressureLocal(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * singletWidth,
        -20 * singletWidth,
    )


def singletPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    dsdz = (
        0.5 * singletVEV / (singletWidth * np.cosh(z / singletWidth + wallOffSet) ** 2)
    )
    return -dsdz * singletEquationOfMotion(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        z,
        Veff,
        offEquilDeltas,
        T,
    )


def singletStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    Veff,
    offEquilDeltas,
    Tfunc,
):
    return quad(
        lambda z: singletStretchLocal(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * singletWidth,
        -20 * singletWidth,
    )


def singletStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    dsdz = (
        0.5 * singletVEV / (singletWidth * np.cosh(z / singletWidth + wallOffSet) ** 2)
    )
    offCenterWeight = np.tanh(z / singletWidth + wallOffSet)
    return (
        dsdz
        * offCenterWeight
        * singletEquationOfMotion(
            higgsVEV,
            higgsWidth,
            singletVEV,
            singletWidth,
            wallOffSet,
            z,
            Veff,
            offEquilDeltas,
            T,
        )
    )


def singletEquationOfMotion(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    Veff,
    offEquilDeltas,
    T,
):
    zLSingletOff = z / singletWidth + wallOffSet
    kinetic = (
        singletVEV * np.tanh(zLSingletOff) / (singletWidth * np.cosh(zLSingletOff)) ** 2
    )
    potential = Veff.singletDerivative(
        wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z), T
    )
    return kinetic + potential


def initialWallParameters(
    higgsWidthGuess, singletWidthGuess, wallOffSetGuess, TGuess, Veff
):
    higgsVEV = Veff.higgsVEV(TGuess)
    singletVEV = Veff.singletVEV(TGuess)

    initRes = minimize(
        lambda higgsWidth, singletWidth, wallOffSet: oneDimAction(
            higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, TGuess, Veff
        ),
        x0=[higgsWidthGuess, singletWidthGuess, wallOffSetGuess],
        bounds=[(0, None), (0, None), (None, None)],
    )

    return initRes.x[0], initRes.x[1], initRes.x[2]


def oneDimAction(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, T, Veff):
    kinetic = (1 / higgsWidth + 1 / singletWidth) * 3 / 2

    integrationLength = (20 + np.abs(wallOffSet)) * max(higgsWidth, singletWidth)

    potential = quad(
        lambda z: Veff.V(
            wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z),
            T,
        ),
        -integrationLength,
        integrationLength,
    ) - integrationLength * (Veff.V(higgsVEV, 0, T) + Veff.V(0, singletVEV, T))

    return kinetic + potential


def wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z):
    h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
    s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))

    return [h, s]


def findTemperatureProfile(
    c1,
    c2,
    higgsWidth,
    singletWidth,
    wallOffSet,
    offEquilDeltas,
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
            c1, c2, Veff, h, dhdz, s, dsdz, offEquilDeltas, Tplus, Tminus
        )

    return np.array(findTemperatureProfile)


def findTemperaturePoint(c1, c2, Veff, h, dhdz, s, dsdz, offEquilDeltas, Tplus, Tminus):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.
    """

    s1 = c1 - offEquilDeltas[0, 3]
    s2 = c2 - offEquilDeltas[3, 3]

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
