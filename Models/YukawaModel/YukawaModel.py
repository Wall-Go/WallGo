import numpy as np
import numpy.typing as npt
import os
import pathlib

from time import process_time

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
## For Benoit benchmarks we need the unresummed, non-high-T potential:
from WallGo import EffectivePotential
from WallGo import Fields

## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class YukawaModel(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}

    ## Specifying this is REQUIRED
    fieldCount = 1


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotential_Yukawa(self.modelParameters, self.fieldCount)

        ## Define particles. this is a lot of clutter, especially if the mass expressions are long, 
        ## so @todo define these in a separate file? 
        
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        ## === Top quark ===
        psiMsqVacuum = lambda fields: self.modelParameters["y"]**2 * fields.GetField(0)**2
        psiMsqThermal = lambda T: 0     #TODO: What is this? 

        psi = Particle("top", 
                            msqVacuum = psiMsqVacuum,
                            msqThermal = psiMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(psi)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["y"]**2 * T**2 / 6.0

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
        gluonMsqThermal = lambda T: self.modelParameters["y"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = 0.0,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(gluon)




    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        modelParameters["msq"] = inputParameters["msq"]
        modelParameters["g"] = inputParameters["g"]
        modelParameters["lambda"] = inputParameters["lambda"]
        modelParameters["y"] = inputParameters["y"]

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotential_Yukawa(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 29 
        self.num_fermion_dof = 90



    ## ---------- EffectivePotential overrides. 
    # The user needs to define evaluate(), which has to return value of the effective potential when evaluated at a given field configuration, temperature pair. 
    # Remember to include full T-dependence, including eg. the free energy contribution from photons (which is field-independent!)

    def evaluate(self, fields: Fields, temperature: float) -> complex:

        # for Benoit benchmark we don't use high-T approx and no resummation: just Coleman-Weinberg with numerically evaluated thermal 1-loop

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        v = fields.GetField(0)

        msq = self.modelParameters["msq"]
        g = self.modelParameters["g"]
        lam = self.modelParameters["lambda"]
        y = self.modelParameters["y"]

        sRes = g * temperature**2 / 24
        msqRes = msq + (lam + 4 * y**2) * temperature**2 / 24

        # resummed potential
        VRes = sRes*v + 0.5*msqRes*v**2 + g*v**3 / 6 + 0.25*lam*v**4

        return VRes
    

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


def main():

    WallGo.initialize()

    ## Create WallGo control object
    manager = WallGoManager()


    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "msq" : 1,
        "g" : -0.79,
        "lambda" : 0.069,
        "y" : 0.24
    }

    model = YukawaModel(inputParameters)

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## ---- File name for collisions integrals. Currently we just load this
    collisionFileName = pathlib.Path(__file__).parent.resolve() / "collisions_top_top_N11.hdf5" 
    manager.loadCollisionFile(collisionFileName)


    ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----
    

    """ Example mass loop that just does one value of mh2. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """  
    modelParameters = model.calculateModelParameters(inputParameters)

    """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
    Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
    """
    Tn = 11. ## nucleation temperature
    phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                    phaseLocation1 = WallGo.Fields( [2.5] ), 
                                    phaseLocation2 = WallGo.Fields( [26.2] ))
    

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

    bIncludeOffEq = True
    print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    startTime = process_time()
    wallVelocity, wallParams = manager.solveWall(bIncludeOffEq)
    endTime = process_time()
    print("Time to complete: ", endTime-startTime)

    print(f"{wallVelocity=}")
    print(f"{wallParams.widths=}")
    print(f"{wallParams.offsets=}")

    exit()

    ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
    bIncludeOffEq = True
    print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    wallVelocity, wallParams = manager.solveWall(bIncludeOffEq)

    print(f"{wallVelocity=}")
    print(f"{wallParams.widths=}")
    print(f"{wallParams.offsets=}")


    # end parameter-space loop

# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()