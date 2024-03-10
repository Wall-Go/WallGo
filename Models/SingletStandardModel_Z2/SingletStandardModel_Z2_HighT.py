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


"""Define a model-specific dataclass for holding parameters required to evaluate the effective potential etc.
This should inherit from WallGo.ActionParameters which is the common interface for WallGo model parameters.
Here the init=False decorator means no __init__ function is generated for the dataclass, and slots=True
tells Python to use __slots__ so that new variables are not allowed at runtime.
"""
@dataclass(init=False, slots=True)
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


@dataclass(init=False, slots=True)
class SingletThermalParameters(SingletParameters):
    """Same as above but with extra mD
    """
    mD1sq: float
    mD2sq: float
    mD3sq: float


## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class SingletSM_Z2_HighT(GenericModel):

    ## Specifying this is REQUIRED
    fieldCount = 2

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

        modelParameters = SingletParameters()

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


    def computeThermalParameters(self, temperature: npt.ArrayLike, params: SingletParameters) -> SingletParameters:
        """Override of GenericModel.computeThermalParameters.
        """

        # shorthands
        T = temperature
        p = params

        thermalParams = SingletThermalParameters()

        ## LO matching: only masses get corrected
        thermalParams.RGScale = p.RGScale
        thermalParams.yt = p.yt ## do we need this?!?!
        thermalParams.g1 = p.g1
        thermalParams.g2 = p.g2
        thermalParams.g3 = p.g3

        thermalParams.lam = p.lam
        thermalParams.a2 = p.a2
        thermalParams.b4 = p.b4

        thermalParams.msq = p.msq + T**2 / 16. * (3. * p.g2**2 + p.g1**2 + 4.*p.yt**2 + 8.*p.lam) + T**2 * p.a2 / 24.
        thermalParams.b2 = p.b2 + T**2 * (1./6. *p.a2 + 1./4. *p.b4)

        # how many Higgs doublets / fermion generations
        Nd = 1
        Nf = 3

        ## Debye masses squared (U1, SU2, SU3) 
        thermalParams.mD1sq = p.g1**2 * T**2 * (Nd/6. + 5.*Nf/9.)
        thermalParams.mD2sq = p.g2**2 * T**2 * ( (4. + Nd) / 6. + Nf/3.)
        thermalParams.mD3sq = p.g3**2 * T**2 * (1. + Nf/3.)

        return thermalParams


# end model


class EffectivePotentialxSM_Z2_HighT(WallGo.EffectivePotential):

    ## Store model reference as we need it to get thermal params etc
    def __init__(self, model: SingletSM_Z2_HighT, fieldCount: int):
        
        self.fieldCount = fieldCount
        self.model = model


    ## ---------- EffectivePotential overrides. 
    # The user needs to define evaluate(), which has to return value of the effective potential when evaluated at a given field configuration, temperature pair. 
    # Remember to include full T-dependence, including eg. the free energy contribution from photons (which is field-independent!)

    def evaluate(self, fields: Fields, temperature: npt.ArrayLike) -> npt.ArrayLike:

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        v, x = fields.GetField(0), fields.GetField(1)

        """For this example we compute this to next-to-leading (NLO) order in the high-T approximation m ~gT,
        where g is a small parameter. NLO means g^3 accuracy, which requires thermal masses at 1-loop
        and 1-loop integration over Matsubara zero modes. The T=0 Coleman-Weinberg loops are not included
        as they are ~ m^4 log(m) ~ (gT)^4 log(g).
        """

        # Get resummed masses etc
        p: SingletThermalParameters = self.model.computeThermalParameters(temperature, self.model.modelParameters)

        # tree level potential
        V0 = 0.5*p.msq*v**2 + 0.25*p.lam*v**4 + 0.5*p.b2*x**2 + 0.25*p.b4*x**4 + 0.25*p.a2*v**2 *x**2

        ## Add loops over Matsubara zero modes. The masses here are resummed thermal masses.
        # 3D loop integral, but keep 4D units:
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * temperature

        ## Cheating a bit here and just hardcoding gauge/"goldstone" masses
        mWsq = thermalParameters["g2"]**2 * v**2 / 4.
        mZsq = (thermalParameters["g1"]**2 + thermalParameters["g2"]**2) * v**2 / 4.
        mGsq = msq + lam*v**2 + 0.5*a2*x**2


        ## Scalar mass matrix needs diagonalization, just doing it manually here
        # matrix ( a, b // b, c)

        A = msq + 0.5*a2*x**2 + 3.*v**2*lam
        B = b2 + 0.5*a2*v**2 + 3.*b4*x**2
        C = a2 *v*x 
        thingUnderSqrt = A**2 + B**2 - 2.*A*B + 4.*C**2

        msqEig1 = 0.5 * (A + B - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (A + B + np.sqrt(thingUnderSqrt))
        
    
        # NLO 1-loop correction in Landau gauge. So g^3, Debyes are integrated out by getThermalParameters
        V1 = 2*(3-1) * J3(mWsq) + (3-1) * J3(mZsq) + 3.*J3(mGsq) + J3(msqEig1) + J3(msqEig2)


        VTotal = (
            -self.pressureLO(temperature) # Free energy is -pressure
            + V0
            +

        )

        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + self.V1(bosonStuff, fermionStuff, p.RGScale)
            + self.V1T(bosonStuff, fermionStuff, temperature)
        )

        return VTotal
    

    def pressureLO(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Ideal gas pressure. This contributes a temperature-dependent but field-independent
        term in the effective potential. Obtained from DRalgo, although this could easily be computed
        by just counting degrees of freedom and doing a high-T expansion on loop integrals.
        """
        ## the 3 is from 3 fermion generations
        return (29./90. + 7./24 * 3) * np.pi**2 * temperature**4



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

    model = SingletSM_Z2_HighT(inputParameters)

    params = model.calculateModelParameters(inputParameters)

    thermalParams = model.computeThermalParameters(np.array([123.0, 32]), params)

    print(f"{thermalParams=}")
    input()

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
