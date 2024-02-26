import numpy as np
import numpy.typing as npt
import os
import pathlib
from dataclasses import dataclass

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
## For Benoit benchmarks we need the unresummed, non-high-T potential:
from WallGo import EffectivePotential_NoResum
from WallGo import Fields, WallGoResults

## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class SingletSM_Z2(GenericModel):

    ## Specifying this is REQUIRED
    fieldCount = 2

    """Define a model-specific dataclass for holding parameters required to evaluate the effective potential etc.
    This should inherit from WallGo.ActionParameters which is the common interface for WallGo model parameters.
    (The init=False means no __init__ function is generated for the dataclass.)
    """
    @dataclass(init=False)
    class SingletParameters(WallGo.ActionParameters):
        RGScale: float # Renormalization scale (in MS-bar scheme for this example)
        yt: float # top Yukawa
        g1: float # U(1) gauge coupling
        g2: float # SU(2) gauge coupling
        g3: float # SU(3) gauge coupling
        msq: float # phi^2
        b2: float # S^2
        lam: float # phi^4 ("lambda", but that keyword is reserved by Python)
        a2: float # phi^2 S^2
        b4: float # S^4


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialxSM_Z2(self.modelParameters, self.fieldCount)

        ## Define particles. this is a lot of clutter, especially if the mass expressions are long, 
        ## so @todo define these in a separate file? 
        
        self.particles: list[Particle] = []
        self.outOfEquilibriumParticles: list[Particle] = []

        ## === Top quark ===
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters.yt**2 * fields.GetField(0)**2
        topMsqThermal = lambda T: self.modelParameters.g3**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(topQuark)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters.g3**2 * T**2 / 6.0

        lightQuark = Particle("lightQuark", 
                            msqVacuum = 0.0,
                            msqThermal = lightQuarkMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 5
        )
        self.addParticle(lightQuark)

        ## === SU(3) gluon ===
        gluonMsqThermal = lambda T: self.modelParameters.g3**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = 0.0,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(gluon)


    @staticmethod
    def calculateModelParameters(inputParameters: dict[str, float]) -> SingletParameters:
        """Converts "physical" input parameters to field-theory parameters (ie. those appearing in the action).
        """

        modelParameters = SingletSM_Z2.SingletParameters()

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        mh1 = inputParameters["mh1"] # 125 GeV
        mh2 = inputParameters["mh2"]

        ## these are direct inputs:
        modelParameters.RGScale = inputParameters["RGScale"]
        modelParameters.a2 = inputParameters["a2"]
        modelParameters.b4 = inputParameters["b4"]
        

        modelParameters.lam = 0.5 * mh1**2 / v0**2
        modelParameters.msq = -modelParameters.lam * v0**2
        modelParameters.b2 = mh2**2 - 0.5 * v0**2 * inputParameters["a2"]

        ## Then the gauge/Yukawa sector

        # helpers
        Mt = inputParameters["Mt"] 
        MW = inputParameters["MW"]
        MZ = inputParameters["MZ"]
        g0 = 2.*MW / v0

        modelParameters.g1 = g0*np.sqrt((MZ/MW)**2 - 1)
        modelParameters.g2 = g0
        # Just take QCD coupling as input
        modelParameters.g3 = inputParameters["g3"]

        modelParameters.yt = np.sqrt(1./2.)*g0 * Mt/MW

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotentialxSM_Z2(EffectivePotential_NoResum):

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

    def evaluate(self, fields: Fields, temperature: float) -> complex:

        # for Benoit benchmark we don't use high-T approx and no resummation: just Coleman-Weinberg with numerically evaluated thermal 1-loop

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        v, x = fields.GetField(0), fields.GetField(1)

        # shorthand reference
        p = self.modelParameters

        # tree level potential
        V0 = 0.5*p.msq*v**2 + 0.25*p.lam*v**4 + 0.5*p.b2*x**2 + 0.25*p.b4*x**4 + 0.25*p.a2*v**2 *x**2

        # From Philipp. @todo should probably use the list of defined particles here?
        bosonStuff = self.boson_massSq(fields, temperature)
        fermionStuff = self.fermion_massSq(fields, temperature)


        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + self.V1(bosonStuff, fermionStuff, p.RGScale)
            + self.V1T(bosonStuff, fermionStuff, temperature)
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


    def boson_massSq(self, fields: Fields, temperature):

        v, x = fields.GetField(0), fields.GetField(1)

        # shorthand reference
        p = self.modelParameters

        # Scalar masses, just diagonalizing manually. matrix (A C // C B)
        A = p.msq + 0.5*p.a2*x**2 + 3.*v**2*p.lam
        B = p.b2 + 0.5*p.a2*v**2 + 3.*p.b4*x**2
        C = p.a2 *v*x
        thingUnderSqrt = A**2 + B**2 - 2.*A*B + 4.*C**2

        msqEig1 = 0.5 * (A + B - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (A + B + np.sqrt(thingUnderSqrt))

        mWsq = p.g2**2 * v**2 / 4.
        mZsq = (p.g1**2 + p.g2**2) * v**2 / 4.
        # "Goldstones"
        mGsq = p.msq + p.lam*v**2 + 0.5*p.a2*x**2

        # h, s, chi, W, Z
        massSq = np.column_stack( (msqEig1, msqEig2, mGsq, mWsq, mZsq) )
        degreesOfFreedom = np.array([1,1,3,6,3]) 
        c = np.array([3/2,3/2,3/2,5/6,5/6])

        return massSq, degreesOfFreedom, c
    

    def fermion_massSq(self, fields: Fields, temperature):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters.yt
        mtsq = yt**2 * v**2 / 2
    
        # @todo include spins for each particle

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])
        
        return massSq, degreesOfFreedom



def main():

    WallGo.initialize()

    ## Create WallGo control object
    manager = WallGoManager()


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

    ## ---- File name for collisions integrals. Currently we just load this
    collisionFileName = pathlib.Path(__file__).parent.resolve() / "Collisions/collisions_top_top_N11.hdf5"
    manager.loadCollisionFile(collisionFileName)


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
        manager.setParameters(modelParameters, phaseInfo)

        ## TODO initialize collisions. Either do it here or already in registerModel(). 
        ## But for now it's just hardcoded in Boltzmann.py and __init__.py

        """WallGo can now be used to compute wall stuff!"""

        ## ---- Solve wall speed in Local Thermal Equilibrium approximation

        vwLTE = manager.wallSpeedLTE()

        print(f"LTE wall speed: {vwLTE}")

        ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

        ## This will contain wall widths and offsets for each classical field. Offsets are relative to the first field, so first offset is always 0
        wallParams: WallGo.WallParams

        bIncludeOffEq = False
        print(f"=== Begin EOM with {bIncludeOffEq=} ===")

        results = manager.solveWall(bIncludeOffEq)
        wallVelocity = results.wallVelocity
        widths = results.wallWidths
        offsets = results.wallOffsets

        print(f"{wallVelocity=}")
        print(f"{widths=}")
        print(f"{offsets=}")

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
