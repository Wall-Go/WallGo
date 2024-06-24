import os
import pathlib
import numpy as np
import numpy.typing as npt

## WallGo imports
import WallGo  ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager

## To compare to 2211.13142 we need the unresummed, non-high-T potential:
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

        # Initialize internal Veff with our params dict.
        self.effectivePotential = EffectivePotentialIDM(
            self.modelParameters, self.fieldCount
        )

        ## Define particles. this is a lot of clutter, especially if the mass
        ## expressions are long, so @todo define these in a separate file?

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
    def calculateModelParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
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
        modelParameters["yt"] = np.sqrt(2.0) * Mt / v0

        ## Then the inert doublet parameters
        mH = inputParameters["mH"]
        mA = inputParameters["mA"]
        mHp = inputParameters["mHp"]

        lambda5 = (mH**2 - mA**2) / v0**2
        lambda4 = -2 * (mHp**2 - mA**2) / v0**2 + lambda5
        lambda3 = 2 * inputParameters["lambdaL"] - lambda4 - lambda5
        msq2 = mHp**2 - lambda3 * v0**2 / 2

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


## For this benchmark model we use the 4D potential, implemented as in 2211.13142. 
## We use interpolation tables for Jb/Jf
class EffectivePotentialIDM(EffectivePotential_NoResum):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions 
        ## to ideal gas pressure
        self.num_boson_dof = 32
        self.num_fermion_dof = 90

        """For this benchmark model we do NOT use the default integrals from WallGo.
        This is because the benchmark points we're comparing with were originally done 
        with integrals from CosmoTransitions. 
        In real applications we recommend using the WallGo default implementations.
        """
        self._configureBenchmarkIntegrals()

    def _configureBenchmarkIntegrals(self):

        ## Load custom interpolation tables for Jb/Jf.
        # These should be the same as what CosmoTransitions version 2.0.2 
        # provides by default.
        thisFileDirectory = os.path.dirname(os.path.abspath(__file__))
        self.integrals.Jb.readInterpolationTable(
            os.path.join(thisFileDirectory, "interpolationTable_Jb_testModel.txt"),
            bVerbose=False,
        )
        self.integrals.Jf.readInterpolationTable(
            os.path.join(thisFileDirectory, "interpolationTable_Jf_testModel.txt"),
            bVerbose=False,
        )

        self.integrals.Jb.disableAdaptiveInterpolation()
        self.integrals.Jf.disableAdaptiveInterpolation()

        """And force out-of-bounds constant extrapolation because this is what 
        CosmoTransitions does => not really reliable for very negative (m/T)^2 ! 
        Strictly speaking: For x > xmax, CosmoTransitions just returns 0. But a 
        constant extrapolation is OK since the integral is very small at the upper limit.
        """

        from WallGo.InterpolatableFunction import EExtrapolationType

        self.integrals.Jb.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

        self.integrals.Jf.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

    ## ---------- EffectivePotential overrides.
    # The user needs to define evaluate(), which has to return value of the effective
    # potential when evaluated at a given field configuration, temperature pair.
    # Remember to include full T-dependence, including eg. the free energy contribution
    # from photons (which is field-independent!)

    def evaluate(
        self, fields: Fields, temperature: float, checkForImaginary: bool = False
    ) -> complex:

        # phi ~ 1/sqrt(2) (0, v)
        v = fields.GetField(0)

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        # tree level potential
        V0 = 0.5 * msq * v**2 + 0.25 * lam * v**4

        bosonStuff = self.bosonMassSq(fields)
        fermionStuff = self.fermionMassSq(fields)

        bosonStuffResummed = self.bosonMassSqResummed(fields, temperature)
        fermionMass, fermionDOF,_ ,_ = fermionStuff
        fermionStuffT = fermionMass, fermionDOF, 3/2, 1

        VTotal = (
            V0
            + self.constantTerms(temperature)
            + self.V1(bosonStuff, fermionStuff, checkForImaginary)
            + self.V1T(
                bosonStuffResummed, fermionStuffT, temperature, checkForImaginary
            )
        )

        return VTotal
    
    def Jcw(self, msq: float, degrees_of_freedom: int, c: float, rgScale: float):
        return degrees_of_freedom*(msq*msq * (np.log(np.abs(msq/rgScale**2) + 1e-100) - c) + 2 * msq * rgScale**2)

    def fermionMassSq(self, fields: Fields):

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2 + 1e-100
        mtsq0T = yt**2 * self.modelParameters["v0"] ** 2 / 2

        massSq = np.stack((mtsq,), axis=-1)
        massSq0T = np.stack((mtsq0T,), axis=-1)
        degreesOfFreedom = np.array([12])

        return massSq, degreesOfFreedom, 3/2, np.sqrt(massSq0T)

    def bosonMassSq(self, fields: Fields):

        v = fields.GetField(0)
        v0 = self.modelParameters["v0"]

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        msq2 = self.modelParameters["msq2"]
        lam3 = self.modelParameters["lambda3"]
        lam4 = self.modelParameters["lambda4"]
        lam5 = self.modelParameters["lambda5"]

        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        mhsq = msq + 3 * lam * v**2
        mHsq = msq2 + (lam3 + lam4 + lam5) / 2 * v**2
        mAsq = msq2 + (lam3 + lam4 - lam5) / 2 * v**2
        mHpmsq = msq2 + lam3 / 2 * v**2

        mhsq0T = msq + 3 * lam * v0**2
        mHsq0T = msq2 + (lam3 + lam4 + lam5) / 2 * v0**2
        mAsq0T = msq2 + (lam3 + lam4 - lam5) / 2 * v0**2
        mHpmsq0T = msq2 + lam3 / 2 * v0**2

        mWsq = g2**2 * v**2 / 4.0 + 1e-100
        mZsq = (g1**2 + g2**2) * v**2 / 4.0 + 1e-100

        mWsq0T = g2**2 * v0**2 / 4.0
        mZsq0T = (g1**2 + g2**2) * v0**2 / 4.0

        # this feels error prone:

        # W, Z, h, H, A, Hpm
        massSq = np.column_stack((mWsq, mZsq, mhsq, mHsq, mAsq, mHpmsq))
        massSq0 = np.column_stack((mWsq0T, mZsq0T, mhsq0T, mHsq0T, mAsq0T, mHpmsq0T))
        degreesOfFreedom = np.array([6, 3, 1, 1, 1, 2])
        c = 3/2*np.ones(6)

        return massSq, degreesOfFreedom, c, np.sqrt(massSq0)

    def bosonMassSqResummed(self, fields: Fields, temperature):

        v = fields.GetField(0)

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

        piPhi = (
            temperature**2
            / 12.0
            * (6 * lam + 2 * lam3 + lam4 + 3 / 4 * (3 * g2**2 + g1**2) + 3 * yt**2)
        )  # Eq. (15) of  2211.13142 (note the different normalization of lam)
        piEta = (
            temperature**2
            / 12.0
            * (6 * lam2 + 2 * lam3 + lam4 + 3 / 4 * (3 * g2**2 + g1**2))
        )  # Eq. (16) of 2211.13142 (note the different normalization of lam2)

        mhsq = np.abs(msq + 3 * lam * v**2 + piPhi)
        mGsq = np.abs(msq + lam * v**2 + piPhi)  # Goldstone bosons
        mHsq = msq2 + (lam3 + lam4 + lam5) / 2 * v**2 + piEta
        mAsq = msq2 + (lam3 + lam4 - lam5) / 2 * v**2 + piEta
        mHpmsq = msq2 + lam3 / 2 * v**2 + piEta

        mWsq = g2**2 * v**2 / 4.0
        mWsqL = g2**2 * v**2 / 4.0 + 2 * g2**2 * temperature**2
        mZsq = (g1**2 + g2**2) * v**2 / 4.0

        # Eigenvalues of the Z&B-boson mass matrix
        piB = 2 * g1**2 * temperature**2
        piW = 2 * g2**2 * temperature**2
        m1sq = g1**2 * v**2 / 4
        m2sq = g2**2 * v**2 / 4
        m12sq = -g1 * g2 * v**2 / 4

        msqEig1 = (
            m1sq
            + m2sq
            + piB
            + piW
            - np.sqrt(4 * m12sq**2 + (m2sq - m1sq - piB + piW) ** 2)
        ) / 2
        msqEig2 = (
            m1sq
            + m2sq
            + piB
            + piW
            + np.sqrt(4 * m12sq**2 + (m2sq - m1sq - piB + piW) ** 2)
        ) / 2

        # HACK This is probably not the optimal solution
        if mWsq.shape != mWsqL.shape:
            mWsq = mWsq * np.ones(mWsqL.shape[0])
            mZsq = mZsq * np.ones(mWsqL.shape[0])

        # this feels error prone:

        # W, Wlong, Z,Zlong,photonLong, h, Goldstone H, A, Hpm
        massSq = np.column_stack(
            (mWsq, mWsqL, mZsq, msqEig1, msqEig2, mhsq, mGsq, mHsq, mAsq, mHpmsq)
        )
        degreesOfFreedom = np.array([4, 2, 2, 1, 1, 1, 3, 1, 1, 2])

        # As c and the RG-scale don't enter in V1T, we just set them to 0
        return massSq, degreesOfFreedom, 0, 0

    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## See Eq. (39) in hep-ph/0510375 for general LO formula

        ## How many degrees of freedom we have left. I'm hardcoding the number of DOFs that were done in evaluate(), could be better to pass it from there though
        dofsBoson = self.num_boson_dof - 17
        dofsFermion = self.num_fermion_dof - 12  ## we only did top quark loops

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0


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
    temperatureScale = 1.0
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = 10.0
    manager = WallGoManager(
        wallThicknessIni, meanFreePath, temperatureScale, fieldScale
    )

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "v0": 246.22,
        "Mt": 172.76,
        "g1": 0.35,
        "g2": 0.65,
        "g3": 1.2279920495357861,
        "lambda2": 0.1,
        "lambdaL": 0.0015,
        "mh": 125.0,
        "mH": 62.66,
        "mA": 300.0,
        "mHp": 300.0,  # We don't use mHm as input parameter, as it is equal to mHp
    }

    model = InertDoubletModel(inputParameters)

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    ## collision stuff

    ## Create Collision singleton which automatically loads the collision module
    ## here it will be only invoked in read-only mode if the module is not found
    collision = WallGo.Collision(model)
    # automatic generation of collision integrals is disabled by default
    # comment this line if collision integrals already exist
    collision.generateCollisionIntegrals = True

    """
    Define couplings (Lagrangian parameters)
    list as they appear in the MatrixElements file
    """
    collision.manager.addCoupling(inputParameters["g3"])
    collision.manager.addCoupling(inputParameters["g2"])

    ## ---- Directory name for collisions integrals. Currently we just load these
    scriptLocation = pathlib.Path(__file__).parent.resolve()
    collisionDirectory = scriptLocation / "CollisionOutput/"
    collisionDirectory.mkdir(parents=True, exist_ok=True)

    collision.setOutputDirectory(collisionDirectory)
    collision.manager.setMatrixElementFile(str(scriptLocation / "MatrixElements.txt"))

    manager.loadCollisionFiles(collision)

    ## ---- This is where you'd start an input parameter loop if doing parameter-space scans ----

    """ Example mass loop that just does one value of mH. Note that the WallGoManager class is NOT thread safe internally, 
    so it is NOT safe to parallelize this loop eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """
    values_mH = [62.66]

    for mH in values_mH:

        inputParameters["mH"] = mH

        """In addition to model parameters, WallGo needs info about the phases at nucleation temperature.
        Use the WallGo.PhaseInfo dataclass for this purpose. Transition goes from phase1 to phase2.
        """

        Tn = 117.1  ## nucleation temperature

        phaseInfo = WallGo.PhaseInfo(
            temperature=Tn,
            phaseLocation1=WallGo.Fields([0.0]),
            phaseLocation2=WallGo.Fields([246.0]),
        )

        """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
        """

        ## Wrap everything in a try-except block to check for WallGo specific errors
        try:
            manager.setParameters(phaseInfo)

            """WallGo can now be used to compute wall stuff!"""

            ## ---- Solve wall speed in Local Thermal Equilibrium approximation

            vwLTE = manager.wallSpeedLTE()

            print(f"LTE wall speed: {vwLTE}")

            ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions.
            ## The resulting wall speed should match the LTE result:

            bIncludeOffEq = False
            print(f"=== Begin EOM with {bIncludeOffEq=} ===")

            results = manager.solveWall(bIncludeOffEq)
            print(f"results=")
            wallVelocity = results.wallVelocity
            widths = results.wallWidths
            offsets = results.wallOffsets

            print(f"{wallVelocity=}")
            print(f"{widths=}")
            print(f"{offsets=}")

            ## Repeat with out-of-equilibrium parts included. 
            #This requires solving Boltzmann equations, invoked automatically by solveWall()
            bIncludeOffEq = True
            print(f"=== Begin EOM with {bIncludeOffEq=} ===")

            results = manager.solveWall(bIncludeOffEq)
            wallVelocity = results.wallVelocity
            wallVelocityError = results.wallVelocityError
            widths = results.wallWidths
            offsets = results.wallOffsets

            print(f"{wallVelocity=}")
            print(f"{wallVelocityError=}")
            print(f"{widths=}")
            print(f"{offsets=}")

        except WallGo.WallGoError as error:
            ## something went wrong!
            print(error)
            continue

    # end parameter-space loop


# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
