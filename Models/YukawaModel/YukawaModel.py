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
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters["yt"]**2 * fields.GetField(0)**2
        topMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

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
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

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
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

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

        modelParameters["m_b"] = inputParameters["m_b"]
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

        msq = self.modelParameters["msq"]
        b2 = self.modelParameters["b2"]
        lam = self.modelParameters["lambda"]
        b4 = self.modelParameters["b4"]
        a2 = self.modelParameters["a2"]

        RGScale = self.modelParameters["RGScale"]

        """
        # Get thermal masses
        thermalParams = self.getThermalParameters(temperature)
        mh1_thermal = msq - thermalParams["msq"] # need to subtract since msq in thermalParams is msq(T=0) + T^2 (...)
        mh2_thermal = b2 - thermalParams["b2"]
        """

        # tree level potential
        V0 = 0.5*msq*v**2 + 0.25*lam*v**4 + 0.5*b2*x**2 + 0.25*b4*x**4 + 0.25*a2*v**2 *x**2

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
        dofsBoson = self.num_boson_dof - 14
        dofsFermion = self.num_fermion_dof - 12 ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7./8. * dofsFermion) * np.pi**2 * temperature**4 / 90.


    ## High-T stuff commented out for now
    """
    ## Evaluate the potential in high-T approx (but keep 4D units)
    def evaluateHighT(self, fields: np.ndarray[float], temperature: float) -> complex:

        v = fields[0] # phi ~ 1/sqrt(2) (0, v)
        x = fields[1] # just S -> S + x 
        T = temperature

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)
        
        msq = thermalParameters["msq"]
        lam = thermalParameters["lambda"]
        b2 = thermalParameters["b2"]
        b4 = thermalParameters["b4"]
        a2 = thermalParameters["a2"]
        

        # tree level potential
        V0 = 0.5 * msq * v**2 + 0.25 * lam * v**4 + 0.5*b2*x**2 + 0.25*b4*x**4 + 0.25*a2*v**2 * x**2

        ## @todo should have something like a static class just for defining loop integrals. NB: m^2 can be negative for scalars so make it complex
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * T # keep 4D units

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

        VTotal = V0 + V1
        return VTotal
    

    ## Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature
        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        
        b2 = self.modelParameters["b2"]
        a2 = self.modelParameters["a2"]
        b4 = self.modelParameters["b4"]

        ## LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["msq"] = msq + T**2 / 16. * (3. * g2**2 + g1**2 + 4.*yt**2 + 8.*lam) + T**2 * a2 / 24.

        thermalParameters["b2"] = b2 + T**2 * (1./6. *a2 + 1./4. *b4)

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

        # But adding these as Benoit benchmark needs them explicitly...?
        thermalParameters["mDsq1"] = mDsq1
        thermalParameters["mDsq2"] = mDsq2

        return thermalParameters
    """

    def boson_massSq(self, fields: Fields, temperature):

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

        return massSq, degreesOfFreedom, c
    

    def fermion_massSq(self, fields: Fields, temperature):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
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
        "m_b" : 1,
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
    values_mh2 = [ 120.0 ]
    for mh2 in values_mh2:

        inputParameters["mh2"] = mh2

        modelParameters = model.calculateModelParameters(inputParameters)

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """
        Tn = 11. ## nucleation temperature
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