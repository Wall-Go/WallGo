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
from WallGo import EffectivePotential_NoResum
from WallGo import Fields

# Inert doublet model, as implemented in 2211.13142
class InertDoubletModel(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}

    ## Specifying this is REQUIRED
    fieldCount = 1


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialIDM(self.modelParameters, self.fieldCount)

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
                            msqDerivative = topMsqDerivative,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            totalDOFs = 12
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
                            totalDOFs = 60
        )
        self.addParticle(lightQuark)

        ## === SU(3) gluon ===
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = 0.0,
                            msqDerivative = 0.0,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            totalDOFs = 16
        )
        self.addParticle(gluon)

        ## Go from whatever input params --> action params
    ## This function was just copied from SingletStandardModel_Z2
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0
        
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
        lambda4 = -2*(mHp**2- mA**2)/v0**2 + lambda5
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


## For this benchmark model we use the 4D potential, implemented as in 2211.13142. We use interpolation tables for Jb/Jf 
class EffectivePotentialIDM(EffectivePotential_NoResum):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 32
        self.num_fermion_dof = 90 


        #JvdV: TODO figure out which integrals to use!
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

        # phi ~ 1/sqrt(2) (0, v)
        v = fields.GetField(0) 
        
        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]


        """
        # Get thermal masses
        thermalParams = self.getThermalParameters(temperature)
        mh1_thermal = msq - thermalParams["msq"] # need to subtract since msq in thermalParams is msq(T=0) + T^2 (...)
        mh2_thermal = b2 - thermalParams["b2"]
        """

        # tree level potential
        V0 = 0.5*msq*v**2 + 0.25*lam*v**4

        # From Philipp. @todo should probably use the list of defined particles here?
        bosonStuff = self.boson_massSq(fields)
        fermionStuff = self.fermion_massSq(fields)

        bosonStuffResummed = self.boson_massSqResummed(fields, temperature)
        fermionMass, _, fermionDOF = fermionStuff
        fermionStuffT = fermionMass, fermionDOF

        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + self.ColemanWeinberg(bosonStuff, fermionStuff) 
            + self.V1T(bosonStuffResummed, fermionStuffT, temperature, checkForImaginary)
        )

        return VTotal
    
    
    def ColemanWeinberg(self,bosons, fermions) -> float:
        c = 3./2.
        m2, m20T, nb = bosons
        Vboson = 1./(64.*np.pi**2)*np.sum(nb*(m2**2*(np.log(np.abs(m2) /m20T )- c) + 2*m2*m20T), axis=-1)
        
        m2, m20T, nf = fermions
        Vfermion = -1./(64.*np.pi**2)*np.sum(nf*(m2**2*(np.log(np.abs(m2) /m20T)- c) + 2*m2*m20T), axis=-1)

        return  Vboson + Vfermion
    
    def fermion_massSq(self, fields: Fields):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2 + 1e-100
        mtsq0T = yt**2 *self.modelParameters["v0"]**2/2
    
        # @todo include spins for each particle

        massSq = np.stack((mtsq,), axis=-1)
        massSq0T = np.stack((mtsq0T,), axis = -1)
        degreesOfFreedom = np.array([12])
        
        return massSq, massSq0T, degreesOfFreedom

    def boson_massSq(self, fields: Fields):

        v = fields.GetField(0) 
        v0 = self.modelParameters["v0"]

        # TODO: numerical determination of scalar masses from V0

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        msq2 = self.modelParameters["msq2"]
        lam3 = self.modelParameters["lambda3"]
        lam4 = self.modelParameters["lambda4"]
        lam5 = self.modelParameters["lambda5"]

        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        
        mhsq = msq + 3*lam*v**2
        mHsq = msq2 + (lam3 + lam4 + lam5)/2*v**2
        mAsq = msq2 + (lam3 + lam4 - lam5)/2*v**2
        mHpmsq = msq2 + lam3/2*v**2

        mhsq0T = msq + 3*lam*v0**2
        mHsq0T = msq2 + (lam3 + lam4 + lam5)/2*v0**2
        mAsq0T = msq2 + (lam3 + lam4 - lam5)/2*v0**2
        mHpmsq0T = msq2 + lam3/2*v0**2

        mWsq = g2**2 * v**2 / 4.+1e-100
        mZsq = (g1**2 + g2**2) * v**2 / 4. + 1e-100

        mWsq0T = g2**2*v0**2/4.
        mZsq0T = (g1**2 + g2**2)*v0**2/4.

        # this feels error prone:

        # W, Z, h, H, A, Hpm
        massSq = np.column_stack( (mWsq, mZsq, mhsq, mHsq, mAsq,mHpmsq ) )
        massSq0 = np.column_stack( ( mWsq0T, mZsq0T, mhsq0T, mHsq0T, mAsq0T,mHpmsq0T  ))
        degreesOfFreedom = np.array([6,3,1,1,1,2]) 

        return massSq, massSq0, degreesOfFreedom
    
    
    def boson_massSqResummed(self, fields: Fields, temperature):

        v = fields.GetField(0) 
        
        # TODO: numerical determination of scalar masses from V0

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        msq2 = self.modelParameters["msq2"]
        lam2 = self.modelParameters["lambda2"]
        lam3 = self.modelParameters["lambda3"]
        lam4 = self.modelParameters["lambda4"]
        lam5 = self.modelParameters["lambda5"]

        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        PiPhi = temperature**2/12.*(6*lam + 2*lam3 + lam4 + 3/4*(3*g2**2 + g1**2) + 3*yt**2) # Eq. (15) of  2211.13142 (note the different normalization of lam)
        PiEta = temperature**2/12.*(6*lam2 + 2*lam3 + lam4 + 3/4*(3*g2**2 + g1**2))# Eq. (16) of 2211.13142 (note the different normalization of lam2)

        mhsq = np.abs(msq + 3*lam*v**2 + PiPhi)
        mGsq = np.abs(msq + lam*v**2 + PiPhi) #Goldstone bosons
        mHsq = msq2 + (lam3 + lam4 + lam5)/2*v**2 + PiEta
        mAsq = msq2 + (lam3 + lam4 - lam5)/2*v**2 + PiEta
        mHpmsq = msq2 + lam3/2*v**2 + PiEta

        mWsq = g2**2 * v**2 / 4.
        mWsqL = g2**2 * v**2 / 4. + 2*g2**2*temperature**2
        mZsq = (g1**2 + g2**2) * v**2 / 4.

        #Eigenvalues of the Z&B-boson mass matrix
        PiB = 2*g1**2*temperature**2
        PiW = 2*g2**2*temperature**2
        m1sq = g1**2*v**2/4
        m2sq = g2**2*v**2/4
        m12sq = -g1*g2*v**2/4

        msqEig1 = (m1sq + m2sq + PiB+ PiW - np.sqrt(4*m12sq**2 + (m2sq - m1sq - PiB +PiW)**2))/2
        msqEig2 = (m1sq + m2sq + PiB+ PiW + np.sqrt(4*m12sq**2 + (m2sq - m1sq - PiB +PiW)**2))/2

        if(mWsq.shape != mWsqL.shape):
            print(f"{mWsq=} {mWsqL=} {mWsq.shape=} {mWsqL.shape=} {v=} {temperature=}") 

        # this feels error prone:

        # W, Wlong, Z,Zlong,photonLong, h, Goldstone H, A, Hpm
        massSq = np.column_stack((mWsq, mWsqL,mZsq, msqEig1, msqEig2, mhsq, mGsq, mHsq, mAsq,mHpmsq ) )
        degreesOfFreedom = np.array([4,2,2,1,1,1,3,1,1,2]) 

        return massSq, degreesOfFreedom, 0

    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## See Eq. (39) in hep-ph/0510375 for general LO formula

        ## How many degrees of freedom we have left. I'm hardcoding the number of DOFs that were done in evaluate(), could be better to pass it from there though
        dofsBoson = self.num_boson_dof - 17
        dofsFermion = self.num_fermion_dof - 12 ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.


def main():

    WallGo.initialize()

    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    ## Length scale determining transform in the xi-direction. See eq (26) in the paper
    Lxi = 0.05

    ## Create WallGo control object
        # The following 2 parameters are used to estimate the optimal value of dT used 
    # for the finite difference derivatives of the potential.
    # Temperature scale over which the potential changes by O(1). A good value would be of order Tc-Tn.
    temperatureScale = 1.
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = 10.
    manager = WallGoManager(Lxi, temperatureScale, fieldScale)

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
        "mH" : 62.66,
        "mA" : 300.,
        "mHp" : 300. # We don't use mHm as input parameter, as it is equal to mHp
    }


    model = InertDoubletModel(inputParameters)

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## ---- File name for collisions integrals. Currently we just load this
    collisionDirectory = pathlib.Path(__file__).parent.resolve() / "collisions_N11"
    manager.loadCollisionFiles(collisionDirectory)

   ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----

    """ Example mass loop that just does one value of mH. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """  
    values_mH = [ 62.66 ]

    for mH in values_mH:

        inputParameters["mH"] = mH

        modelParameters = model.calculateModelParameters(inputParameters)

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """

        Tn = 117.1 ## nucleation temperature

        phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                        phaseLocation1 = WallGo.Fields( [0.0] ), 
                                        phaseLocation2 = WallGo.Fields( [246.0] ))
        

        """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
        """ 
        #print(f"{model.effectivePotential.evaluate(Fields(10.),100.)=} {model.effectivePotential.evaluate(Fields(10.),10.)=} {model.effectivePotential.evaluate(Fields(1.),0.1)=} ")

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