import numpy as np

from scipy.optimize import minimize, brentq, root
from scipy.integrate import quad
from .Boltzmann import BoltzmannBackground, BoltzmannSolver


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

    higgsWidthGuess = 1 / hMass
    singletWidthGuess = 1 / sMass
    wallOffSetGuess = 0
    higgsWidth, singletWidth, wallOffSet = initialWallParameters(
        higgsWidthGuess,
        singletWidthGuess,
        wallOffSetGuess,
        0.5 * (Tplus + Tminus),
        model,
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

        wallProfileGrid = wallProfileOnGrid(wallParameters[1:], Tplus, Tminus, grid)

        Tprofile, velocityProfile = findPlasmaProfile(
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

        boltzmannBackground = BoltzmannBackground(wallParameters[0], velocityProfile, wallProfileGrid, Tprofile)

        boltzmannSolver = BoltzmannSolver(grid, boltzmannBackground, model.particles)

        offEquilDeltas = boltzmannSolver.getDeltas()

        intermediateRes = root(
            momentsOfWallEoM, wallParameters, args=(offEquilDeltas, model)
        )

        wallParameters = intermediateRes.x

        error = np.sqrt(
            ((wallVelocity - oldWallVelocity) / wallVelocity) ** 2
            + ((higgsWidth - oldHiggsWidth) / higgsWidth) ** 2
            + ((singletWidth - oldSingletWidth) / singletWidth) ** 2
            + (wallOffSet - oldWallOffSet) ** 2
        )

    return wallParameters


def momentsOfWallEoM(wallParameters, offEquilDeltas, freeEnergy):
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

    higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
    singletVEV = freeEnergy.findPhases(Tplus)[0,1]

    mom1 = higgsPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        offEquilDeltas,
        Tfunc,
    )
    mom2 = higgsStretchMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        offEquilDeltas,
        Tfunc,
    )
    mom3 = singletPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        offEquilDeltas,
        Tfunc,
    )
    mom4 = singletStretchMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
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
    freeEnergy,
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
            freeEnergy,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * higgsWidth,
        20 * higgsWidth,
    )


def higgsPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
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
        freeEnergy,
        offEquilDeltas,
        T,
    )


def higgsStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
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
            freeEnergy,
            offEquilDeltas,
            Tfunc(z),
        ),
        -20 * higgsWidth,
        20 * higgsWidth,
    )


def higgsStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
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
            freeEnergy,
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
    freeEnergy,
    offEquilDeltas,
    T,
):
    zLHiggs = z / higgsWidth
    kinetic = -higgsVEV * np.tanh(zLHiggs) / (higgsWidth * np.cosh(zLHiggs)) ** 2
    [h, s] = wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z)
    [dVdh, dVds] = freeEnergy.derivField([h, s], T)
    offEquil = (
        0.5
        * 12
        * Veff.dTopMassdh([h, s], T) #need to rewrite
        * offEquilDeltas[0, 0]
    )
    return kinetic + dVdh + offEquil


def singletPressureMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
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
            freeEnergy,
            offEquilDeltas,
            Tfunc(z),
        ),
        -(20 + np.abs(wallOffSet)) * singletWidth,
        (20 + np.abs(wallOffSet)) * singletWidth,
    )


def singletPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
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
        freeEnergy,
        offEquilDeltas,
        T,
    )


def singletStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
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
            freeEnergy,
            offEquilDeltas,
            Tfunc(z),
        ),
        -(20 + np.abs(wallOffSet)) * singletWidth,
        (20 + np.abs(wallOffSet)) * singletWidth,
    )


def singletStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
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
            freeEnergy,
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
    freeEnergy,
    offEquilDeltas,
    T,
):
    zLSingletOff = z / singletWidth + wallOffSet
    kinetic = (
        singletVEV * np.tanh(zLSingletOff) / (singletWidth * np.cosh(zLSingletOff)) ** 2
    )
    [h, s] = wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z)
    [dVdh, dVds] = freeEnergy.derivField([h, s], T)
    return kinetic + potential


def initialWallParameters(
    higgsWidthGuess, singletWidthGuess, wallOffSetGuess, TGuess, freeEnergy
):
    higgsVEV = freeEnergy.findPhases(TGuess)[1,0]
    singletVEV = freeEnergy.findPhases(TGuess)[0,1]

    initRes = minimize(
        lambda wallParams: oneDimAction(higgsVEV, singletVEV, wallParams, TGuess, freeEnergy),
        x0=[higgsWidthGuess, singletWidthGuess, wallOffSetGuess],
        bounds=[(0, None), (0, None), (-10, 10)],
    )

    return initRes.x[0], initRes.x[1], initRes.x[2]


def oneDimAction(higgsVEV, singletVEV, wallParams, T, freeEnergy):
    [higgsWidth, singletWidth, wallOffSet] = wallParams

    kinetic = (higgsVEV**2 / higgsWidth + singletVEV**2 / singletWidth) * 3 / 2

    integrationLength = (20 + np.abs(wallOffSet)) * max(higgsWidth, singletWidth)

    integral = quad(
        lambda z: freeEnergy(
            wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z),
            T,
        ),
        -integrationLength,
        integrationLength,
    )

    potential = integral[0] - integrationLength * (
        freeEnergy([higgsVEV, 0], T) + freeEnergy([0, singletVEV], T)
    )

    print(higgsWidth, singletWidth, wallOffSet)

    print(kinetic + potential)

    return kinetic + potential


def wallProfileOnGrid(staticWallParams, Tplus, Tminus, grid,freeEnergy):
    [higgsWidth, singletWidth, wallOffSet] = staticWallParams

    higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
    singletVEV = freeEnergy.findPhases(Tplus)[0,1]

    wallProfileGrid = []
    for z in grid.xiValues:
        wallProfileGrid.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))

    return wallProfileGrid


def wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z):
    h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
    s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))

    return [h, s]



def findPlasmaProfile(
    c1,
    c2,
    higgsWidth,
    singletWidth,
    wallOffSet,
    offEquilDeltas,
    Tplus,
    Tminus,
    freeEnergy,
    grid,
):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 globally. If no solution, the minimum of LHS.
    """
    temperatureProfile = []
    velocityProfile = []
    for z in grid.xiValues:
        higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
        h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
        dhdz = (
            -0.5 * higgsVEV / (higgsWidth * np.cosh(z / higgsWidth) ** 2)
        )

        singletVEV = freeEnergy.findPhases(Tplus)[0,1]
        s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))
        dsdh = (
            0.5
            * singletVEV
            / (singletWidth * np.cosh(z / singletWidth + wallOffSet) ** 2)
        )

        T, vPlasma = findPlasmaProfilePoint(
            c1, c2, freeEnergy, h, dhdz, s, dsdz, offEquilDeltas, Tplus, Tminus
        )

        temperatureProfile.append(T)
        velocityProfile.append(vPlasma)

    return temperatureProfile, velocityProfile


def findPlasmaProfilePoint(
    c1, c2, freeEnergy, h, dhdz, s, dsdz, offEquilDeltas, Tplus, Tminus
):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.
    """

    s1 = c1 - offEquilDeltas[0, 3]
    s2 = c2 - offEquilDeltas[3, 3]

    Tavg = 0.5 * (Tplus + Tminus)

    minRes = minimize(
        lambda T: temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, freeEnergy),
        Tavg,
        bounds=(0, None),
        tol=1e-9,
    )
    # TODO: A fail safe

    if temperatureProfileEqLHS(h, s, dhdz, dsdz, minRes.x, s1, s2, freeEnergy) >= 0:
        T = minRes.x
        vPlasma = plasmaVelocity(h, s, T, s1, freeEnergy)
        return T, vPlasma

    TLowerBound = minRes.x
    TStep = np.abs(Tplus - TLowerBound)
    if TStep == 0:
        TStep = np.abs(Tminus - TLowerBound)

    TUpperBound = TLowerBound + TStep
    while temperatureProfileEqLHS(h, s, dhdz, dsdz, TUpperBound, s1, s2, freeEnergy) < 0:
        TStep *= 2
        TUpperBound = TLowerBound + TStep

    res = brentq(
        lambda T: temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, freeEnergy),
        TLowerBound,
        TUpperBound,
        xtol=1e-9,
        rtol=1e-9,
    )
    # TODO: Can the function have multiple zeros?

    T = res.x
    vPlasma = plasmaVelocity(h, s, T, s1, freeEnergy)
    return T, vPlasma


def temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, freeEnergy):
    """
    The LHS of Eq. (20) of arXiv:2204.13120v1
    """
    return (
        0.5 * (dhdz**2 + dsdz**2)
        - freeEnergy([h, s], T)
        - 0.5 * T*freeEnergy.derivT([h, s], T)
        + 0.5 * np.sqrt(4 * s1**2 + T*freeEnergy.derivT([h, s], T) ** 2)
        - s2
    )



def deltaToTmunu(
    velocityAtCenter,
    Tm,
    offEquilDeltas,
    higgsWidth,
    grid,
    particleList,
    freeEnergy
):

    delta00 = offEquilDeltas["00"]
    delta11 = offEquilDeltas["11"]
    delta02 = offEquilDeltas["02"]
    delta20 = offEquilDeltas["20"]

    u0 = np.sqrt(gammasq(velocityAtCenter))
    u3 = np.sqrt(gammasq(velocityAtCenter))*velocityAtCenter
    ubar0 = u3
    ubar3 = u0

    h = 0.5 * freeEnergy.findPhases(Tm)[1,0]*(1 - np.tanh(grid.xiValues / higgsWidth))
    mTopSquared = 1/2.*model.ytop*h*h #need to update this once model file is ready

    T30 = ((3*delta20 - delta02 - mTopSquared*delta00)*u3*u0+
           (3*delta02 - delta20 + mTopSquared*delta00)*ubar3*ubar0+2*delta11*(u3*ubar0 + ubar3*u0))/2.
    T33 = ((3*delta20 - delta02 - mTopSquared*delta00)*u3*u3+
           (3*delta02 - delta20 + mTopSquared*delta00)*ubar3*ubar3+4*delta11*u3*ubar3)/2.

    return T30, T33

def plasmaVelocity(h, s, T, s1, freeEnergy):
    dVdT = freeEnergy.derivT([h, s], T)
    return (-T * dVdT  + np.sqrt(4 * s1**2 + T * dVdT**2)) / (2 * s1)
