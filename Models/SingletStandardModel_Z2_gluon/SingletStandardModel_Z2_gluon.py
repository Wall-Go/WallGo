import numpy as np
import numpy.typing as npt
import os
import pathlib

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
## For Benoit benchmarks we need the unresummed, non-high-T potential:
from WallGo import EffectivePotentialNoResum
from WallGo import Fields

"""NOTE: the only difference between this file and SingletStandardModel_Z2.py is that we take the gluon to be out-of-eq, and use N=5 instead of N=11.
So this is mostly copy pasted. 
TODO make this smarter with less copy/paste
"""

## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class SingletSM_Z2(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}

    ## Specifying this is REQUIRED
    fieldCount = 2


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialxSM_Z2(self.modelParameters, self.fieldCount)

        self.defineParticles()


    def defineParticles(self) -> None:
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        self.clearParticles()

        ## === Top quark ===
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters["yt"]**2 * fields.GetField(0)**2
        topMsqDerivative = lambda fields: self.modelParameters["yt"]**2 * np.transpose([fields.GetField(0),0*fields.GetField(1)])
        topMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqDerivative = topMsqDerivative,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            totalDOFs = 12
        )
        self.addParticle(topQuark)

        ## === SU(3) gluon ===
        # The msqVacuum function must take a Fields object and return an array of length equal to the number of points in fields.
        gluonMsqVacuum = lambda fields: np.zeros_like(fields.GetField(0))
        # The msqDerivative function must take a Fields object and return an array with the same shape as fields.
        gluonMsqDerivative = lambda fields: np.zeros_like(fields)
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = gluonMsqVacuum,
                            msqDerivative = gluonMsqDerivative,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            totalDOFs = 16
        )
        self.addParticle(gluon)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0
        lightQuark = Particle("lightQuark", 
                            msqVacuum = lambda fields: 0.0,
                            msqDerivative = 0.0,
                            msqThermal = lightQuarkMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            totalDOFs = 60
        )
        self.addParticle(lightQuark)


    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        mh1 = inputParameters["mh1"] # 125 GeV
        mh2 = inputParameters["mh2"]

        ## these are direct inputs:
        modelParameters["RGScale"] = inputParameters["RGScale"]
        modelParameters["a2"] = inputParameters["a2"]
        modelParameters["b4"] = inputParameters["b4"]
        

        modelParameters["lambda"] = 0.5 * mh1**2 / v0**2
        #modelParameters["msq"] = -mh1**2 / 2. # should be same as the following:
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2
        modelParameters["b2"] = mh2**2 - 0.5 * v0**2 * inputParameters["a2"]

        ## Then the gauge/Yukawa sector
        
        Mt = inputParameters["Mt"] 
        MW = inputParameters["MW"]
        MZ = inputParameters["MZ"]

        # helper
        g0 = 2.*MW / v0
        modelParameters["g1"] = g0*np.sqrt((MZ/MW)**2 - 1)
        modelParameters["g2"] = g0
        # Just take QCD coupling as input
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(1./2.)*g0 * Mt/MW

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotentialxSM_Z2(EffectivePotentialNoResum):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 29 
        self.num_fermion_dof = 90 


        """For this benchmark model we do NOT use the default integrals from WallGo.
        This is because the benchmark points we're comparing with were originally done with integrals from CosmoTransitions. 
        In real applications we recommend using the WallGo default implementations.
        """
        self._configureBenchmarkIntegrals()


    def _configureBenchmarkIntegrals(self):
        
        ## Load custom interpolation tables for Jb/Jf. 
        # These should be the same as what CosmoTransitions version 2.0.2 provides by default.
        thisFileDirectory = os.path.dirname(os.path.abspath(__file__))
        self.integrals.Jb.readInterpolationTable(os.path.join(thisFileDirectory, "interpolationTable_Jb_testModel.txt"), bVerbose=False)
        self.integrals.Jf.readInterpolationTable(os.path.join(thisFileDirectory, "interpolationTable_Jf_testModel.txt"), bVerbose=False)
        
        self.integrals.Jb.disableAdaptiveInterpolation()
        self.integrals.Jf.disableAdaptiveInterpolation()

        """And force out-of-bounds constant extrapolation because this is what CosmoTransitions does
        => not really reliable for very negative (m/T)^2 ! 
        Strictly speaking: For x > xmax, CosmoTransitions just returns 0. But a constant extrapolation is OK since the integral is very small 
        at the upper limit.
        """

        from WallGo.InterpolatableFunction import EExtrapolationType
        self.integrals.Jb.setExtrapolationType(extrapolationTypeLower = EExtrapolationType.CONSTANT, 
                                               extrapolationTypeUpper = EExtrapolationType.CONSTANT)
        
        self.integrals.Jf.setExtrapolationType(extrapolationTypeLower = EExtrapolationType.CONSTANT, 
                                               extrapolationTypeUpper = EExtrapolationType.CONSTANT)
        
    

    ## ---------- EffectivePotential overrides. 
    # The user needs to define evaluate(), which has to return value of the effective potential when evaluated at a given field configuration, temperature pair. 
    # Remember to include full T-dependence, including eg. the free energy contribution from photons (which is field-independent!)

    def evaluate(self, fields: Fields, temperature: float, checkForImaginary: bool = False) -> complex:

        # for Benoit benchmark we don't use high-T approx and no resummation: just Coleman-Weinberg with numerically evaluated thermal 1-loop

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        v, x = fields.GetField(0), fields.GetField(1)

        msq = self.modelParameters["msq"]
        b2 = self.modelParameters["b2"]
        lam = self.modelParameters["lambda"]
        b4 = self.modelParameters["b4"]
        a2 = self.modelParameters["a2"]

        # tree level potential
        V0 = 0.5*msq*v**2 + 0.25*lam*v**4 + 0.5*b2*x**2 + 0.25*b4*x**4 + 0.25*a2*v**2 *x**2

        # TODO should probably use the list of defined particles here?
        bosonStuff = self.bosonMassSq(fields, temperature)
        fermionStuff = self.fermionMassSq(fields, temperature)


        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + self.potentialOneLoop(bosonStuff, fermionStuff, checkForImaginary) 
            + self.potentialOneLoopThermal(bosonStuff, fermionStuff, temperature, checkForImaginary)
        )

        return VTotal
    

    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## See Eq. (39) in hep-ph/0510375 for general LO formula

        ## How many degrees of freedom we have left. I'm hardcoding the number of DOFs that were done in evaluate(), could be better to pass it from there though
        dofsBoson = self.num_boson_dof - 14
        dofsFermion = self.num_fermion_dof - 12 ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.

    def bosonMassSq(self, fields: Fields, temperature):

        v, x = fields.GetField(0), fields.GetField(1)

        # TODO: numerical determination of scalar masses from V0

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        
        b2 = self.modelParameters["b2"]
        a2 = self.modelParameters["a2"]
        b4 = self.modelParameters["b4"]

        
        # Scalar masses, just diagonalizing manually. matrix (A C // C B)
        A = msq + 0.5*a2*x**2 + 3.*v**2*lam
        B = b2 + 0.5*a2*v**2 + 3.*b4*x**2
        C = a2 *v*x 
        thingUnderSqrt = A**2 + B**2 - 2.*A*B + 4.*C**2

        msqEig1 = 0.5 * (A + B - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (A + B + np.sqrt(thingUnderSqrt))

        mWsq = g2**2 * v**2 / 4.
        mZsq = (g1**2 + g2**2) * v**2 / 4.
        # "Goldstones"
        mGsq = msq + lam*v**2 + 0.5*a2*x**2

        # this feels error prone:

        # h, s, chi, W, Z
        massSq = np.column_stack( (msqEig1, msqEig2, mGsq, mWsq, mZsq) )
        degreesOfFreedom = np.array([1,1,3,6,3]) 
        c = np.array([3/2,3/2,3/2,5/6,5/6])
        rgScale = self.modelParameters["RGScale"]*np.ones(5)

        return massSq, degreesOfFreedom, c, rgScale
    

    def fermionMassSq(self, fields: Fields, temperature):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2
    
        # @todo include spins for each particle

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])
        c = np.array([3/2])
        rgScale = np.array([self.modelParameters["RGScale"]])

        return massSq, degreesOfFreedom, c, rgScale



def main():

    WallGo.initialize()

    ## Modify the config, we use N=5 for this example
    WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "5")

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
    temperatureScale = 10.
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = [10.,10.]
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
        "mh2" : 120.0,
        "a2" : 0.9,
        "b4" : 1.0
    }

    model = SingletSM_Z2(inputParameters)

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## Create Collision singleton which automatically loads the collision module
    # Use help(Collision.manager) for info about what functionality is available
    collision = WallGo.Collision(model)

    ## ---- Directory name for collisions integrals. Currently we just load these
    scriptLocation = pathlib.Path(__file__).parent.resolve()
    collisionDirectory = scriptLocation / "CollisionOutput/"
    collisionDirectory.mkdir(parents=True, exist_ok=True)
    collision.setOutputDirectory(collisionDirectory)

    manager.loadCollisionFiles(collision)


    ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----
    

    """ Example mass loop that just does one value of mh2. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """  
    values_mh2 = [ 120.0 ]
    for mh2 in values_mh2:

        inputParameters["mh2"] = mh2

        modelParameters = model.calculateModelParameters(inputParameters)

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """
        Tn = 100. ## nucleation temperature
        phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                        phaseLocation1 = WallGo.Fields( [0.0, 200.0] ), 
                                        phaseLocation2 = WallGo.Fields( [246.0, 0.0] ))
        

        """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
        """ 
        manager.setParameters(phaseInfo)

        ## TODO initialize collisions. Either do it here or already in registerModel(). 
        ## But for now it's just hardcoded in Boltzmann.py and __init__.py

        """WallGo can now be used to compute wall stuff!"""

        ## ---- Solve wall speed in Local Thermal Equilibrium approximation

        vwLTE = manager.wallSpeedLTE()

        print(f"LTE wall speed: {vwLTE}")

        ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

        ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
        bIncludeOffEq = True
        print(f"=== Begin EOM with {bIncludeOffEq=} ===")

        results = manager.solveWall(bIncludeOffEq)
        wallVelocity = results.wallVelocity
        widths = results.wallWidths
        offsets = results.wallOffsets

        print(f"{wallVelocity=}")
        print(f"{widths=}")
        print(f"{offsets=}")
        
        


    # end parameter-space loop

# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()