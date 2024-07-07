import numpy as np
import numpy.typing as npt
import pathlib

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
from WallGo import EffectivePotential
from WallGo import Fields, WallGoResults


class StandardModel(GenericModel):

    particles: list[Particle] = []
    outOfEquilibriumParticles: list[Particle] = []
    modelParameters: dict[str, float] = {}
    collisionParameters: dict[str, float] = {}

    ## Specifying this is REQUIRED
    fieldCount = 1

    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)
        self.collisionParameters = self.calculateCollisionParameters(self.modelParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialSM(self.modelParameters, self.fieldCount)

        self.defineParticles()


    def defineParticles(self) -> None:
        self.clearParticles()

        # NB: particle multiplicity is pretty confusing because some internal
        # DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately
        # define up, down, charm, strange, bottom

        ## === Top quark ===
        topMsqVacuum = (
            lambda fields: 0.5
            * self.modelParameters["yt"] ** 2
            * fields.GetField(0) ** 2
        )
        topMsqDerivative = lambda fields: self.modelParameters[
            "yt"
        ] ** 2 * fields.GetField(0)
        topMsqThermal = lambda T: self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuarkL = Particle(
            "topL",
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=6,
        )
        self.addParticle(topQuarkL)

        topQuarkR = Particle(
            "topR",
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=6,
        )
        self.addParticle(topQuarkR)

        ## === SU(2) gauge boson ===
        WMsqThermal = lambda T: self.modelParameters["g2"] ** 2 * T**2 * 11.0 / 6.0
        WMsqVacuum = lambda fields: fields.GetField(0)
        # The msqDerivative function must take a Fields object and return an array with the same shape as fields.
        WMsqDerivative = lambda fields: fields.GetField(0)

        W = Particle(
            "W",
            msqVacuum=WMsqVacuum,
            msqDerivative=WMsqDerivative,
            msqThermal=WMsqThermal,
            statistics="Boson",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=9,
        )
        self.addParticle(W)

        ## === SU(3) gluon ===
        gluonMsqThermal = lambda T: self.modelParameters["g3"] ** 2 * T**2 * 2.0
        gluonMsqVacuum = lambda fields: fields.GetField(0)
        # The msqDerivative function must take a Fields object and return an array with the same shape as fields.
        gluonMsqDerivative = lambda fields: fields.GetField(0)

        gluon = Particle(
            "gluon",
            msqVacuum=gluonMsqVacuum,
            msqDerivative=gluonMsqDerivative,
            msqThermal=gluonMsqThermal,
            statistics="Boson",
            inEquilibrium=True,
            ultrarelativistic=True,
            totalDOFs=16,
        )
        self.addParticle(gluon)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"] ** 2 * T**2 / 6.0

        lightQuark = Particle(
            "lightQuark",
            msqVacuum=lambda fields: 0.0,
            msqDerivative=lambda fields: 0.0,
            msqThermal=lightQuarkMsqThermal,
            statistics="Fermion",
            inEquilibrium=True,
            ultrarelativistic=True,
            totalDOFs=60,
        )
        self.addParticle(lightQuark)


        ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        # Zero-temperature vev
        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0
        
        # Zero-temperature masses
        mH = inputParameters["mH"]
        mW = inputParameters["mW"] 
        mZ = inputParameters["mZ"]
        mt = inputParameters["mt"]

        modelParameters["mW"] = mW
        modelParameters["mZ"] = mZ
        modelParameters["mt"] = mt 

        # helper
        g0 = 2.*mW / v0

        # Gauge couplings
        modelParameters["g1"] = g0*np.sqrt((mZ/mW)**2 - 1)
        modelParameters["g2"] = g0
        modelParameters["g3"] = inputParameters["g3"]
        modelParameters["yt"] = np.sqrt(1./2.)*g0 * mt/mW
        
        modelParameters["lambda"] = inputParameters["mH"]**2/(2*v0**2)

        bconst = 3/(64*np.pi**2*v0**4)*(2*mW**4 + mZ**4 - 4* mt**4)

        modelParameters["D"] = 1/(8*v0**2)*(2*mW**2 + mZ**2 + 2*mt**2)
        modelParameters["E0"] = 1/(12*np.pi*v0**3)*(4*mW**3 + 2*mZ**3)  

        modelParameters["T0sq"] = 1/4/modelParameters["D"]*(mH**2 -8*bconst*v0**2)
        modelParameters["C0"] = 1/(16*np.pi**2)*(1.42*modelParameters["g2"]**4)

        return modelParameters
        
    def calculateCollisionParameters(self, modelParameters: dict[str, float]) -> dict[str, float]:
        """
        Calculate the collision couplings (Lagrangian parameters) from the input parameters.
        List as they appear in the MatrixElements file
        """
        collisionParameters = {}

        collisionParameters["g3"] = modelParameters["g3"]
        collisionParameters["g2"] = modelParameters["g2"]

        return collisionParameters

class EffectivePotentialSM(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 28 #check if correct 
        self.num_fermion_dof = 90 

    def evaluate(self, fields: Fields, temperature: float, checkForImaginary: bool = False) -> complex:
        # phi ~ 1/sqrt(2) (0, v)
        fields = Fields(fields)
        v = fields.GetField(0) + 0.0000001

        T = temperature+ 0.0000001

        ab = 49.78019250 
        af = 3.111262032

        mW = self.modelParameters["mW"]
        mZ = self.modelParameters["mZ"]
        mt = self.modelParameters["mt"]

        lambdaT = self.modelParameters["lambda"] - 3/(16*np.pi*np.pi*self.modelParameters["v0"]**4)*(2*mW**4*np.log(mW**2/(ab*T**2))+ mZ**4*np.log(mZ**2/(ab*T**2)) -4*mt**4*np.log(mt**2/(af*T**2)))

        cT = self.modelParameters["C0"] + 1/(16* np.pi*np.pi)*(4.8*self.modelParameters["g2"]**2*lambdaT - 6*lambdaT**2)
        eT = self.modelParameters["E0"] + 1/(12*np.pi)*(3 + 3**1.5)*lambdaT**1.5

    
        VT = self.modelParameters["D"]*(T**2 - self.modelParameters["T0sq"])*v**2 - cT*T**2*pow(v,2)*np.log(np.abs(v/T)) - eT*T*pow(v,3) + lambdaT/4*pow(v,4)

        VTotal = np.real(VT + self.constantTerms(T))

        return VTotal


    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## See Eq. (39) in hep-ph/0510375 for general LO formula

        ## How many degrees of freedom we have left. I'm hardcoding the number of DOFs that were done in evaluate(), could be better to pass it from there though
        dofsBoson = self.num_boson_dof 
        dofsFermion = self.num_fermion_dof 

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.


def main():

    WallGo.initialize()

    ## Modify the config, we use N=11 for this example
    WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "5")


    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    ## Guess of the wall thickness
    wallThicknessIni = 0.05
    
    # Estimate of the mean free path of the particles in the plasma
    meanFreePath = 1

    # The following 2 parameters are used to estimate the optimal value of dT used 
    
    # for the finite difference derivatives of the potential.
    # Temperature scale over which the potential changes by O(1). A good value would be of order Tc-Tn.
    temperatureScale = 0.1
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = 50.,
    
    ## Create WallGo control object
    manager = WallGoManager(wallThicknessIni, meanFreePath, temperatureScale, fieldScale)

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "v0" : 246.0,
        "mW" : 80.4,
        "mZ" : 91.2,
        "mt" : 174.0,
        "g3" : 1.2279920495357861,
        "mH" : 50.0
    }

    model = StandardModel(inputParameters)

    ## ---- collision integration and path specifications

    # automatic generation of collision integrals is disabled by default
    # set to "False" or comment if collision integrals already exist
    # set to "True" to invoke automatic collision integral generation
    WallGo.config.config.set("Collisions", "generateCollisionIntegrals", "True")
    # Directory name for collisions integrals defaults to "CollisionOutput/"
    # these can be loaded or generated given the flag "generateCollisionIntegrals"
    WallGo.config.config.set("Collisions", "pathName", "CollisionOutput/")

    # set matrix elements initial specs
    WallGo.config.config.set("MatrixElements", "fileName", "MatrixElements.txt")
    ## Instruct the collision manager to print out
    # symbolic matrix elements as it parses them. Can be useful for debugging
    WallGo.config.config.set("MatrixElements", "verbose", "True")

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## Generates or reads collision integrals
    manager.generateCollisionFiles()

   ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----

    """ Example mass loop that does two value of mH. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """  
    
    values_mH = [0.,50.0, 68.0,79.0,88.0]
    values_Tn = [57.192, 83.426, 100.352,111.480,120.934]

    wallVelocitiesErrors = np.zeros((5,4))

    for i in range(len(values_mH)):
        print(f"=== Begin Bechmark with mH = {values_mH[i]} GeV and Tn = {values_Tn[i]} GeV ====")

        inputParameters["mH"] = values_mH[i]

        """ Register the model with WallGo. This needs to be done only once. TODO What does that mean? It seems we have to do it for every choice of input parameters 
        If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
        """

        manager.changeInputParameters(inputParameters, EffectivePotentialSM)


        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """
        Tn = values_Tn[i]

        phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                        phaseLocation1 = WallGo.Fields( [0.0] ), 
                                        phaseLocation2 = WallGo.Fields( [Tn] ))
        

        """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
        """ 
        manager.setParameters(phaseInfo)

        """WallGo can now be used to compute wall stuff!"""

        ## ---- Solve wall speed in Local Thermal Equilibrium approximation

        vwLTE = manager.wallSpeedLTE()

        print(f"LTE wall speed: {vwLTE}")

        ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

        ## Computes the detonation solutions
        #wallGoInterpolationResults = manager.solveWallDetonation()
        #print(wallGoInterpolationResults.wallVelocities)

        ## This will contain wall widths and offsets for each classical field. Offsets are relative to the first field, so first offset is always 0
        wallParams: WallGo.WallParams

        bIncludeOffEq = False
        print(f"=== Begin EOM with {bIncludeOffEq=} ===")

        results = manager.solveWall(bIncludeOffEq)
        wallVelocity = results.wallVelocity
        wallVelocityError = results.wallVelocityError
        widths = results.wallWidths

        print(f"{wallVelocity=}")
        print(f"{wallVelocityError=}")
        print(f"{widths=}")

        ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
        bIncludeOffEq = True
        print(f"=== Begin EOM with {bIncludeOffEq=} ===")

        results = manager.solveWall(bIncludeOffEq)
        wallVelocity = results.wallVelocity
        wallVelocityError = results.wallVelocityError
        widths = results.wallWidths

        print(f"{wallVelocity=}")
        print(f"{wallVelocityError=}")
        print(f"{widths=}")

        wallVelocitiesErrors[i,:] = np.array([Tn,wallVelocity,wallVelocityError,widths[0]])

    print(wallVelocitiesErrors)


    # end parameter-space loop

# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
