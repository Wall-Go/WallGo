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

## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class ahFermion(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}

    ## Specifying this is REQUIRED
    fieldCount = 1


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialAHFermion(self.modelParameters, self.fieldCount)

        ## Define particles. this is a lot of clutter, especially if the mass expressions are long, 
        ## so @todo define these in a separate file? 
        
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        ## === Top quark ===
        topMsqVacuum = lambda fields: self.modelParameters["mXsq"]**2
        topMsqThermal = lambda T: self.modelParameters["g1"]**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(topQuark)

        ## === U(1) dark photon ===
        photonMsqThermal = lambda T: self.modelParameters["g1"]**2 * T**2 * 2./3.

        photon = Particle("photon", 
                            msqVacuum = 0.0,
                            msqThermal = photonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(photon)




    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        mh1 = inputParameters["mh1"] # 125 GeV

        ## these are direct inputs:
        modelParameters["RGScale"] = inputParameters["RGScale"] 

        modelParameters["lambda"] = 0.5 * mh1**2 / v0**2
        modelParameters["mssq"] = -mh1**2 / 2. # should be same as the following:
        # modelParameters["mssq"] = -modelParameters["lambda"] * v0**2

        ## Then the gauge sector

        # helper
        modelParameters["lambda"] = inputParameters["lambda"]
        modelParameters["g1"] = inputParameters["g1"]
        modelParameters["mXsq"] = inputParameters["mXsq"]

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotentialAHFermion(EffectivePotential_NoResum):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 4 # 1 + 1 + 2*1 
        self.num_fermion_dof = 4 # 2*2*1


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
        v = fields.GetField(0)

        mssq = self.modelParameters["mssq"]
        lam = self.modelParameters["lambda"]

        RGScale = self.modelParameters["RGScale"]

        """
        # Get thermal masses
        thermalParams = self.getThermalParameters(temperature)
        mh1_thermal = msq - thermalParams["msq"] # need to subtract since msq in thermalParams is msq(T=0) + T^2 (...)
        mh2_thermal = b2 - thermalParams["b2"]
        """

        # tree level potential
        V0 = 0.5 * mssq * v**2 + 0.25 * lam * v**4

        # From Philipp. @todo should probably use the list of defined particles here?
        bosonStuff = self.boson_massSq(fields, temperature)
        fermionStuff = self.fermion_massSq(fields, temperature)


        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + self.V1(bosonStuff, fermionStuff, RGScale) 
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
        # dofsBoson = self.num_boson_dof
        # dofsFermion = self.num_fermion_dof ## we only did top quark loops
        dofsBoson = self.num_boson_dof - 4
        dofsFermion = self.num_fermion_dof - 4 ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.


    ## Evaluate the potential in high-T approx (but keep 4D units)
    def evaluateHighT(self, fields: np.ndarray[float], temperature: float) -> complex:

        # S -> S + x
        v = fields.GetField(0)
        T = temperature

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)
        
        mssq = thermalParameters["mssq"]
        lam = thermalParameters["lambda"]

        # tree level potential
        V0 = 0.5 * mssq * v**2 + 0.25 * lam * v**4

        ## @todo should have something like a static class just for defining loop integrals. NB: m^2 can be negative for scalars so make it complex
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * T # keep 4D units

        ## Cheating a bit here and just hardcoding gauge/"goldstone" masses
        mVsq = thermalParameters["g1"]**2 * v**2
        mGsq = mssq + lam*v**2
        mSsq = mssq + 3.*lam*v**2


        ## Scalar mass matrix is already diagonal
        # matrix ( a, b // b, c)
    
        # NLO 1-loop correction in Landau gauge. So g^3, Debyes are integrated out by getThermalParameters
        V1 = 1.*(3-1) * J3(mVsq) + 1.*J3(mGsq) + J3(mSsq)

        VTotal = (
            V0 
            + self.constantTerms(temperature)
            + V1
        )
        return VTotal
    

    ## Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature
        msq = self.modelParameters["mssq"]
        lam = self.modelParameters["lambda"]
        g1 = self.modelParameters["g1"]

        ## LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["mssq"] = msq + T**2 / 12. * (3. * g1**2 + 4.*lam)

        # fermion generations
        nG = 1

        ## Debye masses squared (U1) 
        mDsq1 = g1**2 * T**2 * (1/3. + nG/3.)
        mD1 = np.sqrt(mDsq1)

        ## Let's also integrate out B0
        h3 = 2.*g1**2

        thermalParameters["mssq"] += -1/(4.*np.pi) * T * (+ h3 * mD1/2.)
        thermalParameters["lambda"] += -1/(4.*np.pi) * T**2 * (+ h3**2 / (8.*mD1))

        # skipping corrections to gauge couplings because those are not needed at O(g^3)

        # But adding these as Benoit benchmark needs them explicitly...?
        thermalParameters["mDsq1"] = mDsq1

        return thermalParameters


    def boson_massSq(self, fields: Fields, temperature):

        v = fields.GetField(0)

        # TODO: numerical determination of scalar masses from V0

        mssq = self.modelParameters["mssq"]
        lam = self.modelParameters["lambda"]
        g1 = self.modelParameters["g1"]
        
        # Scalar masses are already diagonal
        mVsq = g1**2 * v**2
        # "Goldstones"
        mGsq = mssq + lam*v**2
        mSsq = mssq + 3.*lam*v**2

        # this feels error prone:

        # h, chi, V
        massSq = np.column_stack( (mSsq, mGsq, mVsq) )
        degreesOfFreedom = np.array([1,1,3]) 
        c = np.array([3/2,3/2,5/6])

        return massSq, degreesOfFreedom, c
    

    def fermion_massSq(self, fields: Fields, temperature):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        mXsq = self.modelParameters["mXsq"]
    
        # @todo include spins for each particle

        massSq = np.stack((mXsq,), axis=-1)
        degreesOfFreedom = np.array([4])
        
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
        "RGScale" : 10.0,
        "v0" : 10.0,
        "g1" : 0.1,
        "mh1" : 10.0,
        "lambda" : 1.0,
        "mXsq" : 1.0
    }

    model = ahFermion(inputParameters)

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
    values_mh1 = [17.0 ]
    for mh1 in values_mh1:

        inputParameters["mh1"] = mh1

        modelParameters = model.calculateModelParameters(inputParameters)

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """
        Tn = 12. ## nucleation temperature
        phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                        phaseLocation1 = WallGo.Fields( [0.0] ), 
                                        phaseLocation2 = WallGo.Fields( [16.0] ))
        

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


    # end parameter-space loop

# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()