import numpy as np

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import WallGoManager, Fields
from SingletStandardModel_Z2 import SingletSM_Z2,EffectivePotentialxSM_Z2

class EffectivePotentialxSMScan(EffectivePotentialxSM_Z2):
    def potentialOneLoop(self, bosons, fermions, checkForImaginary=False):
        m2, n, c, _ = bosons
        m2_0 = self.bosonStuff(Fields([self.modelParameters["v0"],0]), 0)[0]
        y = np.sum(n*(m2*m2 * (np.log(np.abs(m2/np.where(m2_0 == 0, self.modelParameters["mh"]**2,m2_0)) + 1e-50)
                              - 1.5)+2*m2_0*m2), axis=-1)
        m2, n, _, _ = fermions
        m2_0 = self.fermionStuff(Fields([self.modelParameters["v0"],0]), 0)[0]
        y -= np.sum(n*(m2*m2 * (np.log(np.abs(m2/np.where(m2_0 == 0, self.modelParameters["mh"]**2,m2_0)) + 1e-50)
                              - 1.5)+2*m2_0*m2), axis=-1)
        return y/(64*np.pi*np.pi)

class SingletSMScan(SingletSM_Z2):
    def __init__(self, initialInputParameters: dict[str, float]):
        """
        Initialize the SingletSM_Z2 model.

        Args:
        - initialInputParameters: A dictionary of initial input parameters for the model.

        Returns:
        None
        """

        self.modelParameters = self.calculateModelParameters(initialInputParameters)
        self.collisionParameters = self.calculateCollisionParameters(self.modelParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialxSMScan(self.modelParameters, self.fieldCount)
        
        self.defineParticles()


modelsBenoit = np.load('models.npy', allow_pickle=True)
i = 6


WallGo.initialize()

## Modify the config, we use N=5 for this example
WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "11")


# Print WallGo config. This was read by WallGo.initialize()
print("=== WallGo configuration options ===")
print(WallGo.config)

## Guess of the wall thickness
wallThicknessIni = 0.05

# Estimate of the mean free path of the particles in the plasma
meanFreePath = 1

## Create WallGo control object
    # The following 2 parameters are used to estimate the optimal value of dT used 
# for the finite difference derivatives of the potential.
# Temperature scale over which the potential changes by O(1). A good value would be of order Tc-Tn.
temperatureScale = modelsBenoit[i]['Tc'] - modelsBenoit[i]['Tn']
# Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
# Can either be a single float, in which case all the fields have the same scale, or an array.
fieldScale = [modelsBenoit[i]['vn']/10,modelsBenoit[i]['wn']/10]
manager = WallGoManager(wallThicknessIni, meanFreePath, temperatureScale, fieldScale)


"""Initialize your GenericModel instance. 
The constructor currently requires an initial parameter input, but this is likely to change in the future
"""

## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
inputParameters = {
    #"RGScale" : 91.1876,
    "RGScale" : 125., # <- Benoit benchmark
    "v0" : 246.0,
    "MW" : 80.379,
    "MZ" : 91.1876,
    "Mt" : 173.0,
    "g3" : 1.2279920495357861,
    # scalar specific, choose Benoit benchmark values
    "mh1" : 125.0,
    "mh2" : modelsBenoit[i]['ms'],
    "a2" : modelsBenoit[i]['lambdaHS'],
    "b4" : modelsBenoit[i]['lambdaS']
}

model = SingletSMScan(inputParameters)

## ---- collision integration and path specifications

# automatic generation of collision integrals is disabled by default
# set to "False" or comment if collision integrals already exist
# set to "True" to invoke automatic collision integral generation
WallGo.config.config.set("Collisions", "generateCollisionIntegrals", "False")

"""
Register the model with WallGo. This needs to be done only once.
If you need to use multiple models during a single run,
we recommend creating a separate WallGoManager instance for each model. 
"""
manager.registerModel(model)

## Generates or reads collision integrals
manager.generateCollisionFiles()

print("=== WallGo parameter scan ===")
## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----

""" Example mass loop that just does one value of mh2. Note that the WallGoManager class is NOT thread safe internally, 
so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
"""  

modelParameters = model.calculateModelParameters(inputParameters)

"""In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
"""
Tn = modelsBenoit[i]['Tn'] ## nucleation temperature

phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                phaseLocation1 = WallGo.Fields( [0.0, abs(modelsBenoit[i]['wn'])]),
                                phaseLocation2 = WallGo.Fields( [abs(modelsBenoit[i]['vn']), 0.0] ))


"""Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
    1) WallGo needs the PhaseInfo 
    2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
""" 
manager.setParameters(phaseInfo)

vwLTE = manager.wallSpeedLTE()

print(f"LTE wall speed: {vwLTE}")

## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

## This will contain wall widths and offsets for each classical field. Offsets are relative to the first field, so first offset is always 0
wallParams: WallGo.WallParams

# ## Computes the detonation solutions
# wallGoInterpolationResults = manager.solveWallDetonation()
# print(wallGoInterpolationResults.wallVelocities)


bIncludeOffEq = False
print(f"=== Begin EOM with {bIncludeOffEq=} ===")

results = manager.solveWall(bIncludeOffEq)
wallVelocity = results.wallVelocity
widths = results.wallWidths
offsets = results.wallOffsets

print(f"WallGo: {wallVelocity=}; Benoit: wallVelocity={modelsBenoit[i]['vw']}")
print(f"{widths=}")
print(f"{offsets=}")

## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
bIncludeOffEq = True
print(f"=== Begin EOM with {bIncludeOffEq=} ===")

results = manager.solveWall(bIncludeOffEq)
wallVelocity = results.wallVelocity
widths = results.wallWidths
offsets = results.wallOffsets

print(f"WallGo: {wallVelocity=}; Benoit: wallVelocity={modelsBenoit[i]['vw_out2']}")
print(f"{widths=}")
print(f"{offsets=}")

