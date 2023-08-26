import numpy as np

from scipy.optimize import minimize, brentq, root
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro
from .model import Particle, FreeEnergy
from .Boltzmann import BoltzmannBackground, BoltzmannSolver
from .helpers import derivative # derivatives for callable functions


def findWallVelocityLoop(particle, freeEnergy, wallVelocityLTE, errTol, grid):
    """
    Finds the wall velocity by solving hydrodynamics, the Boltzmann equation and
    the field equation of motion iteratively.
    """

    # Initial conditions for velocity, hydro boundaries, wall parameters and
    # temperature profile

    thermo = Thermodynamics(freeEnergy)
    hydro = Hydro(thermo)

    if wallVelocityLTE is not None:
        wallVelocity = 0.9 * wallVelocityLTE
        maxWallVelocity = wallVelocityLTE
    else:
        wallVelocity = np.sqrt(1 / 3)
        maxWallVelocity = hydro.vJ

    c1, c2, Tplus, Tminus, velocityAtz0 = hydro.findHydroBoundaries(wallVelocity)

    def ddVddf(freeEnergy,X,whichfield):
        X = np.asanyarray(X)
        h, s = X[..., 0], X[..., 1]
        if whichfield == 0:
            return derivative(
                lambda h: freeEnergy([h,s],freeEnergy.Tnucl),
                h,
                dx = 1e-3,
                n=2,
                order=4,
            )
        else:
            return derivative(
                lambda s: freeEnergy([h,s],freeEnergy.Tnucl),
                s,
                dx = 1e-3,
                n=2,
                order=4,
            )

    hMass = np.sqrt(ddVddf(freeEnergy,freeEnergy.findPhases(freeEnergy.Tnucl)[0],0))
    sMass = np.sqrt(ddVddf(freeEnergy,freeEnergy.findPhases(freeEnergy.Tnucl)[0],1))
    
    higgsWidthGuess = 1 / hMass
    singletWidthGuess = 1 / sMass
    wallOffSetGuess = 0
    higgsWidth, singletWidth, wallOffSet = initialWallParameters(
        higgsWidthGuess,
        singletWidthGuess,
        wallOffSetGuess,
        0.5 * (Tplus + Tminus),
        freeEnergy,
    )

    initializedWallParameters = [wallVelocity, higgsWidth, singletWidth, wallOffSet]

    wallParameters = [wallVelocity, higgsWidth, singletWidth, wallOffSet]

    offEquilDeltas = {"00": np.zeros(grid.M-1), "02": np.zeros(grid.M-1), "20": np.zeros(grid.M-1), "11": np.zeros(grid.M-1)}
    
    error = errTol + 1
    while error > errTol:

        oldWallVelocity = wallParameters[0]
        oldHiggsWidth = wallParameters[1]
        oldSingletWidth = wallParameters[2]
        oldWallOffSet = wallParameters[3]
        oldError = error

        c1, c2, Tplus, Tminus, velocityAtz0 = hydro.findHydroBoundaries(wallVelocity)


        wallProfileGrid = wallProfileOnGrid(wallParameters[1:], Tplus, Tminus, grid,freeEnergy)
        
        Tprofile, velocityProfile = findPlasmaProfile(
            c1,
            c2,
            velocityAtz0,
            higgsWidth,
            singletWidth,
            wallOffSet,
            offEquilDeltas,
            particle,
            Tplus,
            Tminus,
            freeEnergy,
            grid,
        )

        boltzmannBackground = BoltzmannBackground(wallParameters[0], velocityProfile, wallProfileGrid, Tprofile)

        boltzmannSolver = BoltzmannSolver(grid, boltzmannBackground, particle)
        
        # TODO: getDeltas() is not working at the moment (it returns nan), so I turned it off to debug the rest of the loop.
        print('NOTE: offEquilDeltas has been set to 0 to debug the main loop.')
        # offEquilDeltas = boltzmannSolver.getDeltas()

        intermediateRes = root(
            momentsOfWallEoM, wallParameters, args=(offEquilDeltas, freeEnergy, hydro, particle, grid)
        )
        print(intermediateRes)

        wallParameters = intermediateRes.x

        error = np.sqrt(
            ((wallVelocity - oldWallVelocity) / wallVelocity) ** 2
            + ((higgsWidth - oldHiggsWidth) / higgsWidth) ** 2
            + ((singletWidth - oldSingletWidth) / singletWidth) ** 2
            + (wallOffSet - oldWallOffSet) ** 2
        )
    
    return wallParameters


def momentsOfWallEoM(wallParameters, offEquilDeltas, freeEnergy, hydro, particle, grid):
    wallVelocity, higgsWidth, singletWidth, wallOffSet = wallParameters
    c1, c2, Tplus, Tminus, velocityAtz0 = hydro.findHydroBoundaries(wallVelocity)
    Tprofile, vprofile = findPlasmaProfile(
        c1,
        c2,
        velocityAtz0,
        higgsWidth,
        singletWidth,
        wallOffSet,
        offEquilDeltas,
        particle,
        Tplus,
        Tminus,
        freeEnergy,
        grid,
    )

    higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
    singletVEV = freeEnergy.findPhases(Tplus)[0,1]
    
    # Define a function returning the local temparature by interpolating through Tprofile.
    Tfunc = UnivariateSpline(grid.xiValues, Tprofile, k=3, s=0)
    
    # Define a function returning the local Delta00 function by interpolating through offEquilDeltas['00'].
    offEquilDelta00 = UnivariateSpline(grid.xiValues, offEquilDeltas['00'], k=3, s=0)

    mom1 = higgsPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        particle,
        offEquilDelta00,
        Tfunc,  #correct?
    )
    mom2 = higgsStretchMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        particle,
        offEquilDelta00,
        Tfunc,
    )
    mom3 = singletPressureMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        offEquilDelta00,
        Tfunc,
    )
    mom4 = singletStretchMoment(
        higgsVEV,
        higgsWidth,
        singletVEV,
        singletWidth,
        wallOffSet,
        freeEnergy,
        offEquilDelta00,
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
    particle,
    offEquilDelta00,
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
            particle,
            offEquilDelta00(z),
            Tfunc(z),      #this gave an error
            # Tfunc,   # is this ok?
        ),
        -20 * higgsWidth,
        20 * higgsWidth,
    )[0]

def higgsPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
    particle,
    offEquilDelta00,
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
        particle,
        offEquilDelta00,
        T,
    )


def higgsStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
    particle,
    offEquilDelta00,
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
            particle,
            offEquilDelta00(z),
            Tfunc(z),
        ),
        -20 * higgsWidth,
        20 * higgsWidth,
    )[0]


def higgsStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
    particle,
    offEquilDelta00,
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
            particle,
            offEquilDelta00,
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
    particle,
    offEquilDelta00,
    T,
):
    zLHiggs = z / higgsWidth
    kinetic = -higgsVEV * np.tanh(zLHiggs) / (higgsWidth * np.cosh(zLHiggs)) ** 2
    [h, s] = wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z)
    [dVdh, dVds] = freeEnergy.derivField([h, s], T)

    def dmtdh(ptcle,X):
        X = np.asanyarray(X)
        h, s = X[..., 0], X[..., 1]
        return derivative(
            lambda h: ptcle.msqVacuum([h,s]),
            h,
            dx = 1e-3,
            n=1,
            order=4,
        )

    #need to generalize to more than 1 particle.
    offEquil = (
        0.5
        * 12
        * dmtdh(particle,[h,s])
        * offEquilDelta00
    )
    return kinetic + dVdh + offEquil


def singletPressureMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
    offEquilDelta00,
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
            offEquilDelta00(z),
            Tfunc(z),
        ),
        -(20 + np.abs(wallOffSet)) * singletWidth,
        (20 + np.abs(wallOffSet)) * singletWidth,
    )[0]


def singletPressureLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
    offEquilDelta00,
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
        offEquilDelta00,
        T,
    )


def singletStretchMoment(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    freeEnergy,
    offEquilDelta00,
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
            offEquilDelta00(z),
            Tfunc(z),
        ),
        -(20 + np.abs(wallOffSet)) * singletWidth,
        (20 + np.abs(wallOffSet)) * singletWidth,
    )[0]


def singletStretchLocal(
    higgsVEV,
    higgsWidth,
    singletVEV,
    singletWidth,
    wallOffSet,
    z,
    freeEnergy,
    offEquilDelta00,
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
            offEquilDelta00,
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
    offEquilDelta00,
    T,
):
    zLSingletOff = z / singletWidth + wallOffSet
    kinetic = (
        singletVEV * np.tanh(zLSingletOff) / (singletWidth * np.cosh(zLSingletOff)) ** 2
    )
    [h, s] = wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z)
    [dVdh, dVds] = freeEnergy.derivField([h, s], T)
    return kinetic + dVds


def initialWallParameters(
    higgsWidthGuess,
    singletWidthGuess,
    wallOffSetGuess,
    TGuess,
    freeEnergy
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

    # print(higgsWidth, singletWidth, wallOffSet)

    # print(kinetic + potential)

    return kinetic + potential


def wallProfileOnGrid(staticWallParams, Tplus, Tminus, grid,freeEnergy):
    [higgsWidth, singletWidth, wallOffSet] = staticWallParams

    higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
    singletVEV = freeEnergy.findPhases(Tplus)[0,1]

    wallProfileGrid = []
    for z in grid.xiValues:
        wallProfileGrid.append(wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z))

    return np.transpose(wallProfileGrid)


def wallProfile(higgsVEV, singletVEV, higgsWidth, singletWidth, wallOffSet, z):
    h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
    s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))

    return [h, s]



def findPlasmaProfile(
    c1,
    c2,
    velocityAtz0,
    higgsWidth,
    singletWidth,
    wallOffSet,
    offEquilDeltas,
    particle,
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
    for index in range(len(grid.xiValues)):
        z = grid.xiValues[index]
        higgsVEV = freeEnergy.findPhases(Tminus)[1,0]
        h = 0.5 * higgsVEV * (1 - np.tanh(z / higgsWidth))
        dhdz = (
            -0.5 * higgsVEV / (higgsWidth * np.cosh(z / higgsWidth) ** 2)
        )

        singletVEV = freeEnergy.findPhases(Tplus)[0,1]
        s = 0.5 * singletVEV * (1 + np.tanh(z / singletWidth + wallOffSet))
        dsdz = (
            0.5
            * singletVEV
            / (singletWidth * np.cosh(z / singletWidth + wallOffSet) ** 2)
        )

        T, vPlasma = findPlasmaProfilePoint(
            index,c1, c2, velocityAtz0,freeEnergy, h, dhdz, s, dsdz, offEquilDeltas,particle, Tplus, Tminus,grid
        ) 

        temperatureProfile.append(T)
        velocityProfile.append(vPlasma)

    return np.array(temperatureProfile), np.array(velocityProfile)


def findPlasmaProfilePoint(
    index,c1,c2,velocityAtz0,freeEnergy, h, dhdz, s, dsdz, offEquilDeltas,particle, Tplus, Tminus,grid
):
    """
    Solves Eq. (20) of arXiv:2204.13120v1 locally. If no solution, the minimum of LHS.
    """
    
    Tout30, Tout33 = deltaToTmunu(index,h,velocityAtz0,Tminus,offEquilDeltas,grid,particle,freeEnergy)

    s1 = c1 - Tout30 
    s2 = c2 - Tout33

    Tavg = 0.5 * (Tplus + Tminus)
    
    minRes = minimize(
        lambda T: temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, freeEnergy),
        x0 = Tavg,
#        bounds=[(0, None)],
        bounds=[(0,freeEnergy.Tc)],
        tol=1e-9,
    )
    # TODO: A fail safe

    if temperatureProfileEqLHS(h, s, dhdz, dsdz, minRes.x[0], s1, s2, freeEnergy) >= 0:
        T = minRes.x[0]
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

#    T = res.x 
    T = res   #is this okay?
    vPlasma = plasmaVelocity(h, s, T, s1, freeEnergy)
    return T, vPlasma


def temperatureProfileEqLHS(h, s, dhdz, dsdz, T, s1, s2, freeEnergy):
    """
    The LHS of Eq. (20) of arXiv:2204.13120v1
    """
    return (
        0.5 * (dhdz**2 + dsdz**2)
        - freeEnergy([h, s], T)
        + 0.5 * T*freeEnergy.derivT([h, s], T)
        + 0.5 * np.sqrt(4 * s1**2 + (T*freeEnergy.derivT([h, s], T)) ** 2)
        - s2
    )



def deltaToTmunu(
    index,
    h,
    velocityAtCenter,
    Tm,
    offEquilDeltas,
    grid,
    particle,
    freeEnergy
):

    delta00 = offEquilDeltas["00"][index]
    delta11 = offEquilDeltas["11"][index]
    delta02 = offEquilDeltas["02"][index]
    delta20 = offEquilDeltas["20"][index]

    def gammasq(v): #move to helper functions?
        return 1./(1.-v**2)

    u0 = np.sqrt(gammasq(velocityAtCenter))
    u3 = np.sqrt(gammasq(velocityAtCenter))*velocityAtCenter
    ubar0 = u3
    ubar3 = u0


    T30 = ((3*delta20 - delta02 - particle.msqVacuum([h,0])*delta00)*u3*u0+
           (3*delta02 - delta20 + particle.msqVacuum([h,0])*delta00)*ubar3*ubar0+2*delta11*(u3*ubar0 + ubar3*u0))/2.
    T33 = ((3*delta20 - delta02 - particle.msqVacuum([h,0])*delta00)*u3*u3+
           (3*delta02 - delta20 + particle.msqVacuum([h,0])*delta00)*ubar3*ubar3+4*delta11*u3*ubar3)/2.

    return T30, T33

def plasmaVelocity(h, s, T, s1, freeEnergy):
    dVdT = freeEnergy.derivT([h, s], T)
    return (T * dVdT  + np.sqrt(4 * s1**2 + (T * dVdT)**2)) / (2 * s1)
