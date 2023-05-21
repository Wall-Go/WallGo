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


def findJouguetVelocity(Model,Tnucl):
        """
        Finds the Jouguet velocity for a thermal effective potential, defined by Model,
        using that the derivative of v+ wrt T- is zero at the Jouguet velocity
        """
        def vpDerivNum(tm): #the numerator of the derivative of v+^2
            num1 = Model.pSym(Tnucl) - Model.pBrok(tm) #first factor in the numerator of v+^2
            num2 = Model.pSym(Tnucl) + Model.eBrok(tm) 
            den1 = Model.eSym(Tnucl) - Model.eBrok(tm) #first factor in the denominator of v+^2
            den2 = Model.eSym(Tnucl) + Model.pBrok(tm) 
            dnum1 = - Model.dpBrok(tm) #T-derivative of first factor wrt tm
            dnum2 = Model.deBrok(tm)
            dden1 = - Model.deBrok(tm) #T-derivative of second factor wrt tm
            dden2 = Model.dpBrok(tm)
            return(dnum1*num2*den1*den2 + num1*dnum2*den1*den2 - num1*num2*dden1*den2 - num1*num2*den1*dden2)

        tmSol = fsolve(vpDerivNum,Tnucl*1.1)[0]
        vp = np.sqrt((Model.pSym(Tnucl) - Model.pBrok(tmSol))*(Model.pSym(Tnucl) + Model.eBrok(tmSol))/(Model.eSym(Tnucl) - Model.eBrok(tmSol))/(Model.eSym(Tnucl) + Model.pBrok(tmSol)))
        return(vp)

        
        
