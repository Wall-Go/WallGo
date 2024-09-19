"""
This Python script, singletStandardModelZ2.py,
implements a minimal Standard Model extension via
a scalar singlet and incorporating a Z2 symmetry.
Only the top quark is out of equilibrium, and only
QCD-interactions are considered in the collisions.

Features:
- Definition of the extended model parameters including the singlet scalar field.
- Definition of the out-of-equilibrium particles.
- Implementation of the one-loop thermal potential, without high-T expansion.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

Note:
This benchmark model was used to compare against the results of
B. Laurent and J. M. Cline, First principles determination
of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
doi:10.1103/PhysRevD.106.023501
As a consequence, we overwrite the default WallGo thermal functions
Jb/Jf. 
"""

import os
import sys
import pathlib
import argparse
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo.initialize()
from WallGo import Fields, GenericModel, Particle, WallGoManager
from WallGo.InterpolatableFunction import EExtrapolationType

# Add the Models folder to the path and import effectivePotentialNoResum.
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))
from effectivePotentialNoResum import (  # pylint: disable=C0411, C0413, E0401
    EffectivePotentialNoResum,
)


class SingletSMZ2(GenericModel):
    r"""
    Z2 symmetric SM + singlet model.

    The potential is given by:
    V = msq |phi|^2 + lam |phi|^4 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 2

    def __init__(
        self,
        initialInputParameters: dict[str, float],
        allowOutOfEquilibriumGluon: bool = False,
    ):
        """
        Initialize the SingletSMZ2 model.

        Parameters
        ----------
        initialInputParameters: dict[str, float]
            A dictionary of initial input parameters for the model.

        Returns
        ----------
        cls: SingletSMZ2
            An object of the SingletSMZ2 class.
        """
        super().__init__()

        self.modelParameters = self.calculateModelParameters(initialInputParameters)
        self.collisionParameters = self.calculateCollisionParameters(
            self.modelParameters
        )

        # Initialize internal effective potential with our params dict.
        self.effectivePotential = EffectivePotentialxSMZ2(
            self.modelParameters, self.fieldCount
        )

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles(allowOutOfEquilibriumGluon)

    def defineParticles(self, includeGluon: bool) -> None:
        """
        Define the particles for the model.
        Note that the particle list only needs to contain the
        particles that are relevant for the Boltzmann equations.
        The particles relevant to the effective potential are
        included independently.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        self.clearParticles()

        # === Top quark ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def topMsqVacuum(fields: Fields) -> Fields:
            return 0.5 * self.modelParameters["yt"] ** 2 * fields.GetField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"] ** 2 * np.transpose(
                [fields.GetField(0), 0 * fields.GetField(1)]
            )

        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuark = Particle(
            "top",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=12,
        )
        self.addParticle(topQuark)

        if includeGluon:

            # === SU(3) gluon ===
            # The msqVacuum function must take a Fields object and return an array of length equal to the number of points in fields.
            def gluonMsqVacuum(fields: Fields) -> Fields:
                return np.zeros_like(fields.GetField(0))

            def gluonMsqDerivative(fields: Fields) -> Fields:
                return np.zeros_like(fields)

            def gluonMsqThermal(T: float) -> float:
                return self.modelParameters["g3"] ** 2 * T**2 * 2.0

            gluon = Particle(
                "gluon",
                index=1,
                msqVacuum=gluonMsqVacuum,
                msqDerivative=gluonMsqDerivative,
                msqThermal=gluonMsqThermal,
                statistics="Boson",
                totalDOFs=16,
            )
            self.addParticle(gluon)

    ## Go from input parameters --> action parameters
    def calculateModelParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate the model parameters based on the input parameters.

        Parameters
        ----------
        inputParameters: dict[str, float]
            A dictionary of input parameters for the model.

        Returns
        ----------
        modelParameters: dict[str, float]
            A dictionary of calculated model parameters.
        """
        super().calculateModelParameters(inputParameters)

        modelParameters = {}

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        massh1 = inputParameters["mh1"]  # 125 GeV
        massh2 = inputParameters["mh2"]

        # these are direct inputs:
        modelParameters["RGScale"] = inputParameters["RGScale"]
        modelParameters["a2"] = inputParameters["a2"]
        modelParameters["b4"] = inputParameters["b4"]

        modelParameters["lambda"] = 0.5 * massh1**2 / v0**2
        # should be same as the following:
        # modelParameters["msq"] = -massh1**2 / 2.
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2
        modelParameters["b2"] = massh2**2 - 0.5 * v0**2 * inputParameters["a2"]

        # Then the gauge and Yukawa sector
        massT = inputParameters["Mt"]
        massW = inputParameters["MW"]
        massZ = inputParameters["MZ"]

        # helper
        g0 = 2.0 * massW / v0

        modelParameters["g1"] = g0 * np.sqrt((massZ / massW) ** 2 - 1)
        modelParameters["g2"] = g0
        # Just take QCD coupling as input
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(0.5) * g0 * massT / massW

        return modelParameters

    def calculateCollisionParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate collision couplings (Lagrangian parameters) from the input parameters.
        List as they appear in the MatrixElements file.

        Parameters
        ----------
        inputParameters: dict[str, float]
            A dictionary of input parameters for the model.

        Returns
        ----------
        collisionParameters: dict[str, float]
            A dictionary of model parameters for the collision.
        """
        collisionParameters = {}

        collisionParameters["g3"] = inputParameters["g3"]

        return collisionParameters


# end model


class EffectivePotentialxSMZ2(EffectivePotentialNoResum):
    """
    Effective potential for the SingletSMZ2 model.

    This class inherits from the EffectivePotentialNoResum class and provides the
    necessary methods for calculating the effective potential.

    For this benchmark model we use the UNRESUMMED 4D potential.
    Furthermore we use customized interpolation tables for Jb/Jf
    """

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        """
        Initialize the EffectivePotentialxSMZ2.

        Parameters
        ----------
        modelParameters: dict[str, float]
            A dictionary of model parameters.
        fieldCount: int
            The number of fields undergoing the phase transition

        Returns
        ----------
        cls: EffectivePotentialxSMZ2
            an object of the EffectivePotentialxSMZ2 class
        """
        super().__init__(modelParameters, fieldCount)
        # The super call already gave us the model params

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 29
        self.numFermionDof = 90

        """For this benchmark model we do NOT use the default integrals from WallGo.
        This is because the benchmark points we're comparing with were originally done
        with integrals from CosmoTransitions. In real applications we recommend 
        using the WallGo default implementations.
        """
        self._configureBenchmarkIntegrals()

    def _configureBenchmarkIntegrals(self) -> None:
        """
        Configure the benchmark integrals.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Load custom interpolation tables for Jb/Jf. These should be
        # the same as what CosmoTransitions version 2.0.2 provides by default.
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

        """Force out-of-bounds constant extrapolation because this is
        what CosmoTransitions does
        => not really reliable for very negative (m/T)^2 ! 
        Strictly speaking: For x > xmax, CosmoTransitions just returns 0. 
        But a constant extrapolation is OK since the integral is very small 
        at the upper limit.
        """

        self.integrals.Jb.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

        self.integrals.Jf.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

    def evaluate(
        self, fields: Fields, temperature: float, checkForImaginary: bool = False
    ) -> complex | np.ndarray:
        """
        Evaluate the effective potential.

        Parameters
        ----------
        fields: Fields
            The field configuration
        temperature: float
            The temperature
        checkForImaginary: bool
            Setting to check for imaginary parts of the potential

        Returns
        ----------
        potentialTotal: complex | np.ndarray
            The value of the effective potential
        """

        # For this benchmark we don't use high-T approx and no resummation
        # just Coleman-Weinberg with numerically evaluated thermal 1-loop

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        fields = Fields(fields)
        v, x = fields.GetField(0), fields.GetField(1)

        msq = self.modelParameters["msq"]
        b2 = self.modelParameters["b2"]
        lam = self.modelParameters["lambda"]
        b4 = self.modelParameters["b4"]
        a2 = self.modelParameters["a2"]

        # tree level potential
        potentialTree = (
            0.5 * msq * v**2
            + 0.25 * lam * v**4
            + 0.5 * b2 * x**2
            + 0.25 * b4 * x**4
            + 0.25 * a2 * v**2 * x**2
        )

        # Particle masses and coefficients for the CW potential
        bosonStuff = self.bosonStuff(fields)
        fermionStuff = self.fermionStuff(fields)

        potentialTotal = (
            potentialTree
            + self.constantTerms(temperature)
            + self.potentialOneLoop(bosonStuff, fermionStuff, checkForImaginary)
            + self.potentialOneLoopThermal(
                bosonStuff, fermionStuff, temperature, checkForImaginary
            )
        )

        return potentialTotal  # TODO: resolve return type.

    def constantTerms(self, temperature: np.ndarray | float) -> np.ndarray | float:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T
        expansion these are just (minus) the ideal gas pressure of light particles that
        were not integrated over in the one-loop part.

        See Eq. (39) in hep-ph/0510375 for general LO formula


        Parameters
        ----------
        temperature: array-like (float)
            The temperature

        Returns
        ----------
        constantTerms: array-like (float)
            The value of the field-independent contribution to the effective potential
        """

        # How many degrees of freedom we have left. The number of DOFs
        # that were included in evaluate() is hardcoded
        dofsBoson = self.numBosonDof - 14
        dofsFermion = self.numFermionDof - 12  # we only included top quark loops

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0

    def bosonStuff(  # pylint: disable=too-many-locals
        self, fields: Fields
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # TODO: fix return type inheritance error
        """
        Computes parameters for the one-loop potential (Coleman-Weinberg and thermal).

        Parameters
        ----------
        fields: Fields
            The field configuration

        Returns
        ----------
        massSq: array_like
            A list of the boson particle masses at each input point `field`.
        degreesOfFreedom: array_like
            The number of degrees of freedom for each particle.
        c: array_like
            A constant used in the one-loop effective potential
        rgScale : array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential
        """
        v, x = fields.GetField(0), fields.GetField(1)

        # Scalar masses, just diagonalizing manually. matrix (A C // C B)
        mass00 = (
            self.modelParameters["msq"]
            + 0.5 * self.modelParameters["a2"] * x**2
            + 3 * self.modelParameters["lambda"] * v**2
        )
        mass11 = (
            self.modelParameters["b2"]
            + 0.5 * self.modelParameters["a2"] * v**2
            + 3 * self.modelParameters["b4"] * x**2
        )
        mass01 = self.modelParameters["a2"] * v * x
        thingUnderSqrt = (mass00 - mass11)**2 + 4 * mass01**2

        msqEig1 = 0.5 * (mass00 + mass11 - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (mass00 + mass11 + np.sqrt(thingUnderSqrt))

        mWsq = self.modelParameters["g2"] ** 2 * v**2 / 4
        mZsq = mWsq + self.modelParameters["g1"] ** 2 * v**2 / 4
        # Goldstones
        mGsq = (
            self.modelParameters["msq"]
            + self.modelParameters["lambda"] * v**2
            + 0.5 * self.modelParameters["a2"] * x**2
        )

        # h, s, chi, W, Z
        massSq = np.column_stack((msqEig1, msqEig2, mGsq, mWsq, mZsq))
        degreesOfFreedom = np.array([1, 1, 3, 6, 3])
        c = np.array([3 / 2, 3 / 2, 3 / 2, 5 / 6, 5 / 6])
        rgScale = self.modelParameters["RGScale"] * np.ones(5)

        return massSq, degreesOfFreedom, c, rgScale

    def fermionStuff(
        self, fields: Fields
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # TODO: fix return type inheritance error
        """
        Computes parameters for the one-loop potential (Coleman-Weinberg and thermal).

        Parameters
        ----------
        fields: Fields
            The field configuration

        Returns
        ----------
        massSq: array_like
            A list of the fermion particle masses at each input point `field`.
        degreesOfFreedom: array_like
            The number of degrees of freedom for each particle.
        c: array_like
            A constant used in the one-loop effective potential
        rgScale : array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential
        """

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])

        c = np.array([3 / 2])
        rgScale = np.array([self.modelParameters["RGScale"]])

        return massSq, degreesOfFreedom, c, rgScale


def main() -> None:
    """Main function for the Standard Model + real singlet example"""

    scriptLocation = pathlib.Path(__file__).parent.resolve()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--outOfEquilibriumGluon",
        help="Treat the SU(3) gluons as out-of-equilibrium particle species",
        action="store_true",
    )
    argParser.add_argument(
        "--regenerateMatrixElements",
        help="Forces full recalculation of the matrix element from DRalgo",
        action="store_true",
    )
    argParser.add_argument(
        "--recalculateCollisions",
        help="""Forces full recalculation of relevant collision integrals instead of loading the provided data files for this example.
                This is very slow and disabled by default.
                The resulting collision data will be written to a directory labeled _UserGenerated; the default provided data will not be overwritten.
                """,
        action="store_true",
    )

    args = argParser.parse_args()

    print(
        f"Running WallGo example model file: Singlet Standard Model (Z2) ; Out-of-equilibrium gluon = {args.outOfEquilibriumGluon}"
    )
    WallGo.initialize()

    # loading in local config file
    WallGo.config.readINI(
        pathlib.Path(__file__).parent.resolve() / "WallGoSettings.ini"
    )

    ##### TEMPORARY, REMOVE ONCE MATRIX ELEMENT STUFF HAS BEEN SETTLED #####
    bUseBenoitCollisionData: bool = (
        not args.outOfEquilibriumGluon and momentumBasisSize == 11
    )

    # Print WallGo config. This was read by WallGo.initialize()
    print("\n=== WallGo configuration options ===")
    print(WallGo.config)

    # Guess of the wall thickness: 5/Tn
    wallThicknessIni = 0.05

    # Estimate of the mean free path of the particles in the plasma: 100/Tn
    meanFreePath = 1.0

    # Create WallGo control object
    # The following 2 parameters are used to estimate the optimal value of dT used
    # for the finite difference derivatives of the potential.
    # Temperature scale (in GeV) over which the potential changes by O(1).
    # A good value would be of order Tc-Tn.
    temperatureScale = 10.0
    # Field scale (in GeV) over which the potential changes by O(1). A good value
    # would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the
    # same scale, or an array.
    fieldScale = [10.0, 10.0]
    manager = WallGoManager(
        wallThicknessIni, meanFreePath, temperatureScale, fieldScale
    )

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input,
    but this is likely to change in the future
    """

    # QFT model input.
    inputParameters = {
        "RGScale": 125.0,
        "v0": 246.0,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        "mh1": 125.0,
        "mh2": 120.0,
        "a2": 0.9,
        "b4": 1.0,
    }

    model = SingletSMZ2(inputParameters, args.outOfEquilibriumGluon)

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    """Setup and run collision integration using the WallGoCollision extension module.
    Note that collision integration is a very expensive operation, which is why this example performs the computation
    only if the --recalculateCollisions command line flag was specified.
    Without this flag we simply load pre-generated collision data files.
    """

    if args.regenerateMatrixElements:
        from WallGo import mathematicaHelpers

        matrixElementModelFile = scriptLocation / "MatrixElements/matrixElements.qcd.m"
        mathematicaHelpers.generateMatrixElementsViaSubprocess(matrixElementModelFile)
        # FIXME need to ensure that the pre-generated "default" matrix elements do not get overriden

    if args.recalculateCollisions:

        # Failsafe, in general you should not worry about the collision module being unavailable as long as it has been properly installed (eg. with pip)
        assert WallGo.isCollisionModuleAvailable(), """WallGoCollision module could not be loaded, cannot proceed with collision integration.
        Please verify you have successfully installed the module ('pip install WallGoCollision')"""

        import WallGoCollision

        # Collision integrations utilize Monte Carlo methods, so RNG is involved. We can set the global seed for collision integrals as follows.
        # This is optional; by default the seed is 0.
        WallGoCollision.setSeed(0)

        # import utility functions for this example. These are not part of core WallGo and are intended to demonstrate how to setup the collision module
        from example_collision_defs import (
            setupCollisionModel_QCD,
            configureCollisionIntegration,
        )

        # Path to matrix elements file, use location of this .py file as base. This model example only includes QCD processes in collision terms.
        # Matrix elements can be generated with the accompanying Mathematica package; here we load a pre-generated file to simplify the example
        matrixElementFile = scriptLocation / "MatrixElements/MatrixElements_QCD.txt"
        collisionModel: WallGoCollision.PhysicsModel = setupCollisionModel_QCD(
            matrixElementFile, args.outOfEquilibriumGluon
        )

        # Create a CollisionTensor object and initialize to the same grid size used elsewhere in WallGo
        collisionTensor: WallGoCollision.CollisionTensor = (
            collisionModel.createCollisionTensor(momentumBasisSize)
        )

        # Use helper function to manually set integration options specific to this example, for example we enable progress tracking
        configureCollisionIntegration(collisionTensor)

        """Run the collision integrator. This is a very long running function: For M out-of-equilibrium particle species and momentum grid size N,
        there are order M^2 x (N-1)^4 integrals to be computed. In your own runs you may want to handle this part in a separate script and offload it eg. to a cluster,
        especially if using N >> 11.
        """
        print("Entering collision integral computation, this may take long", flush=True)
        collisionResults: WallGoCollision.CollisionTensorResult = (
            collisionTensor.computeIntegralsAll()
        )

        """Export the collision integration results to .hdf5. "individual" means that each off-eq particle pair gets its own file.
        This format is currently required for the main WallGo routines to understand the data. 
        """
        collisionDirectory = (
            scriptLocation / f"CollisionOutput_N{momentumBasisSize}_UserGenerated"
        )
        collisionResults.writeToIndividualHDF5(str(collisionDirectory))

        ## TODO we could convert the CollisionTensorResult object from above to CollisionArray directly instead of forcing write hdf5 -> read hdf5

    ## TEMPORARY
    elif bUseBenoitCollisionData:
        collisionDirectory = (
            scriptLocation / f"CollisionOutput_N{momentumBasisSize}_BenoitBenchmark"
        )

    else:
        # Load pre-generated collision files
        collisionDirectory = scriptLocation / f"CollisionOutput_N{momentumBasisSize}"

    try:
        # Load collision files and register them with the manager. They will be used by the internal Boltzmann solver
        manager.loadCollisionFiles(collisionDirectory)
    except Exception:
        print(
            """\nLoad of collision integrals failed! This example files comes with pre-generated collision files for N=5 and N=11,
              so load failure here probably means you've either moved files around or changed the grid size.
              If you were trying to generate your own collision data, make sure you run this example script with the --recalculateCollisions command line flag.
              """
        )
        exit(2)

    print("=== WallGo parameter scan ===")

    """ Example parameter-space loop that just does one value of mh2. Note that the WallGoManager
    class is NOT thread safe internally, so it is NOT safe to parallelize this loop 
    eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale
    parameter scans. 
    """
    valuesMh2 = [120.0]
    for mh2 in valuesMh2:

        inputParameters["mh2"] = mh2

        """In addition to model parameters, WallGo needs info about the phases at
        nucleation temperature. Use the WallGo.PhaseInfo dataclass for this purpose.
        Transition goes from phase1 to phase2.
        """
        Tn = 100.0  # nucleation temperature
        phaseInfo = WallGo.PhaseInfo(
            temperature=Tn,
            phaseLocation1=WallGo.Fields([0.0, 200.0]),
            phaseLocation2=WallGo.Fields([246.0, 0.0]),
        )

        """Give the input to WallGo. It is NOT enough to change parameters
           directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific
               initializations of internal classes
        """
        manager.setParameters(phaseInfo)

        """WallGo can now be used to compute wall stuff!"""

        # ---- Solve wall speed in Local Thermal Equilibrium approximation

        vwLTE = manager.wallSpeedLTE()

        print(f"LTE wall speed:    {vwLTE:.6f}")

        # ---- Solve field EOM. For illustration, first solve it without any
        # out-of-equilibrium contributions. The resulting wall speed should
        # be close to the LTE result

        bIncludeOffEq = False
        print(f"\n=== Begin EOM with {bIncludeOffEq = } ===")

        results = manager.solveWall(bIncludeOffEq)

        print("\n=== Local equilibrium results ===")
        print(f"wallVelocity:      {results.wallVelocity:.6f}")
        print(f"wallVelocityError: {results.wallVelocityError:.6f}")
        print(f"wallWidths:        {results.wallWidths}")
        print(f"wallOffsets:       {results.wallOffsets}")

        # Repeat with out-of-equilibrium parts included. This requires
        # solving Boltzmann equations, invoked automatically by solveWall()
        bIncludeOffEq = True
        print(f"\n=== Begin EOM with {bIncludeOffEq = } ===")

        results = manager.solveWall(bIncludeOffEq)

        print("\n=== Out-of-equilibrium results ===")
        print(f"wallVelocity:      {results.wallVelocity:.6f}")
        print(f"wallVelocityError: {results.wallVelocityError:.6f}")
        print(f"wallWidths:        {results.wallWidths}")
        print(f"wallOffsets:       {results.wallOffsets}")

        print("\n=== Search for detonation solution ===")
        wallGoDetonationResults = manager.solveWallDetonation(onlySmallest=True)[0]
        print("\n=== Detonation results ===")
        print(f"wallVelocity:      {wallGoDetonationResults.wallVelocity}")

    # end parameter-space loop


# end main()


# Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
