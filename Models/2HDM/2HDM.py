import numpy as np
import numpy.typing as npt
import pathlib

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
## For Benoit benchmarks we need the unresummed, non-high-T potential:
from WallGo import EffectivePotential
from WallGo import Fields

### LN: This file is very WIP, test with SingletStandardModel_Z2.py instead

## V = msq |phi|^2 + lambda (|phi|^2)^2
class StandardModel(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}

    ## Specifying this is REQUIRED
    fieldCount = 1


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialSM(self.modelParameters, self.fieldCount)

        ## Define particles. this is a lot of clutter, especially if the mass expressions are long, 
        ## so @todo define these in a separate file? 
        
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        ## === Top quark ===
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters["yt"]**2 * fields[0]**2
        topMsqDerivative = lambda fields: self.modelParameters["yt"]**2 * np.transpose([fields.GetField(0)])
        topMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqDerivative = topMsqDerivative
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            multiplicity = 1,
                            DOF = 12
        )
        self.addParticle(topQuark)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        lightQuark = Particle("lightQuark", 
                            msqVacuum = 0.0,
                            msqDerivative = 0.0,
                            msqThermal = lightQuarkMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 5,
                            DOF = 60
        )
        self.addParticle(lightQuark)

        ## === SU(3) gluon ===
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = 0.0,
                            msqDerivative 0.0,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 1,
                            DOF = 16
        )
        self.addParticle(gluon)

        ## Go from whatever input params --> action params
    ## This function was just copied from SingletStandardModel_Z2
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        
        # Higgs parameters
        mh = inputParameters["mh"] 
        
        modelParameters["lambda"] = 0.5 * mh**2 / v0**2
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2

        ## Then the Yukawa sector
        
        Mt = inputParameters["Mt"] 
        modelParameters["yt"] = np.sqrt(2.)*Mt/v0

        ## Then the inert doublet parameters
        mH = inputParameters["mH"]
        mA = inputParameters["mA"]
        mHp = inputParameters["mHp"]

        lambda5 = (mH**2 - mA**2)/v0**2
        lambda4 = -2*(mHp**2- mA**2)/v0**2
        lambda3 = 2*inputParameters["lambdaL"]-lambda4 -lambda5
        msq2 = mHp**2 - lambda3*v0**2/2

        modelParameters["msq2"] = msq2

        modelParameters["lambda3"] = lambda3
        modelParameters["lambda4"] = lambda4
        modelParameters["lambda5"] = lambda5


        ## Some couplings are input parameters
        modelParameters["g1"] = inputParameters["g1"]
        modelParameters["g2"] = inputParameters["g2"]
        modelParameters["g3"] = inputParameters["g3"]
        modelParameters["lambda2"] = inputParameters["lambda2"]
        modelParameters["lambdaL"] = inputParameters["lambdaL"]

        return modelParameters




class EffectivePotentialSM(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 28 #check if correct 
        self.num_fermion_dof = 90 

    def evaluate(self, fields: Fields, temperature: float) -> complex:
        # phi ~ 1/sqrt(2) (0, v)
        v = fields.GetField(0) 

        T = temperature

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)
        
        msq = thermalParameters["msq"]
        lam = thermalParameters["lambda"]

        # tree level potential
        V0 = 0.5 * msq * v**2 + 0.25 * lam * v**4

        ## @todo should have something like a static class just for defining loop integrals. NB: m^2 can be negative for scalars
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * T # keep 4D units

        ## Cheating a bit here and just hardcoding gauge/scalar masses in SM
        mWsq = thermalParameters["g2"]**2 * v**2 / 4.
        mZsq = (thermalParameters["g1"]**2 + thermalParameters["g2"]**2) * v**2 / 4.
        mHsq = msq + 3*lam*v**2
        mGsq = msq + lam*v**2
    
        # NLO 1-loop correction in Landau gauge, so g^3. Debyes are integrated out by getThermalParameters
        V1 = 2*(3-1) * J3(mWsq) + (3-1) * J3(mZsq) + J3(mHsq) + 3.*J3(mGsq)

        VTotal = V0 + V1 + self.constantTerms(T)

        return VTotal


    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## See Eq. (39) in hep-ph/0510375 for general LO formula

        ## How many degrees of freedom we have left. I'm hardcoding the number of DOFs that were done in evaluate(), could be better to pass it from there though
        dofsBoson = self.num_boson_dof - 13 # 13 =  6 + 3 + 4 (W + Z + Higgs)
        dofsFermion = self.num_fermion_dof - 12 ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.

    
    ## Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        ## LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["msq"] = msq + T**2 / 16. * (3. * g2**2 + g1**2 + 4.*yt**2 + 8.*lam)

        # how many Higgs doublets / fermion generations
        Nd = 1
        Nf = 3

        ## Debye masses squared (U1, SU2) 
        mDsq1 = g1**2 * T**2 * (Nd/6. + 5.*Nf/9.)
        mDsq2 = g2**2 * T**2 * ( (4. + Nd) / 6. + Nf/3.)
        mD1 = np.sqrt(mDsq1)
        mD2 = np.sqrt(mDsq2)

        ## Let's also integrate out A0/B0
        h3 = g2**2 / 4.
        h3p = g2**2 / 4.
        h3pp = g2*g1 / 2.

        thermalParameters["msq"] += -1/(4.*np.pi) * T * (3. * h3 * mD2 + h3p * mD1)
        thermalParameters["lambda"] += -1/(4.*np.pi) * T * (3.*h3**2 / mD2 + h3p**2 / mD1 + h3pp**2 / (mD1 + mD2))

        # skipping corrections to gauge couplings because those are not needed at O(g^3)

        return thermalParameters


def main():

    WallGo.initialize()

    ## Create WallGo control object
    manager = WallGoManager()

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "v0" : 246.22,
        "Mt" : 172.76,
        "g1" : 0.35,
        "g2" : 0.65,
        "g3" : 1.2279920495357861,
        "lambda2" : 0.1,
        "lambdaL" : 0.0015,
        "mh" : 125.0,
        "mH" : 65.0,
        "mA" : 300.,
        "mHp" : 300. # We don't use mHm as input parameter, as it is equal to mHp
    }


    model = StandardModel(inputParameters)

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## ---- File name for collisions integrals. Currently we just load this
    collisionFileName = pathlib.Path(__file__).parent.resolve() / "Collisions/collisions_top_top_N11.hdf5"
    manager.loadCollisionFile(collisionFileName)

   ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----

    """ Example mass loop that just does one value of mH. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """  
    values_mH = [ 50.0 ]

    for mH in values_mH:

        inputParameters["mH"] = mH

        modelParameters = model.calculateModelParameters(inputParameters)

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """

        Tn = 63.1 ## nucleation temperature

        phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                        phaseLocation1 = WallGo.Fields( [0.0] ), 
                                        phaseLocation2 = WallGo.Fields( [246.0] ))
        

        """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
        """ 


        ## Wrap everything in a try-except block to check for WallGo specific errors
        try:
            manager.setParameters(modelParameters, phaseInfo)

            """WallGo can now be used to compute wall stuff!"""

            ## ---- Solve wall speed in Local Thermal Equilibrium approximation

            vwLTE = manager.wallSpeedLTE()

            print(f"LTE wall speed: {vwLTE}")

            ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

            ## This will contain wall widths and offsets for each classical field. Offsets are relative to the first field, so first offset is always 0
            wallParams: WallGo.WallParams

            bIncludeOffEq = False
            print(f"=== Begin EOM with {bIncludeOffEq=} ===")

            wallVelocity, wallParams = manager.solveWall(bIncludeOffEq)

            print(f"{wallVelocity=}")
            print(f"{wallParams.widths=}")
            print(f"{wallParams.offsets=}")

            ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
            bIncludeOffEq = True
            print(f"=== Begin EOM with {bIncludeOffEq=} ===")

            wallVelocity, wallParams = manager.solveWall(bIncludeOffEq)

            print(f"{wallVelocity=}")
            print(f"{wallParams.widths=}")
            print(f"{wallParams.offsets=}")

        except WallGo.WallGoError as error:
            ## something went wrong!
            print(error)
            continue


    # end parameter-space loop

# end main()



## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()