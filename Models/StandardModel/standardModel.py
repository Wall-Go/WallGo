"""
This Python script, standardModel.py,
implements the Standard Model, with a light Higgs mass.
This model is ruled out by actual measurements of the Higgs mass, which
show that it is 125 GeV, but this file can be used to compare with 
earlier computations performed in the literature.

Features:
- Definition of the standard model parameters.
- Definition of the out-of-equilibrium particles, in our case the top and W-boson.
- Implementation of the thermal potential, with high-T expansion.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

Note:
This benchmark is used to compare against the results of
G. Moore and T. Prokopec, How fast can the wall move?
A Study of the electroweak phase transition dynamics, Phys.Rev.D 52 (1995) 7182-7204
doi:10.1103/PhysRevD.52.7182
"""

import numpy as np
import pathlib

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo.initialize()
from WallGo import EffectivePotential, Fields, GenericModel, Particle, WallGoManager


class StandardModel(GenericModel):
    r"""
    The Standard model, with a light Higgs mass, such that the
    electroweak phase transition becomes fist order.

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    particles: list[Particle] = []
    outOfEquilibriumParticles: list[Particle] = []
    modelParameters: dict[str, float] = {}
    collisionParameters: dict[str, float] = {}

    ## Specifying this is REQUIRED
    fieldCount = 1

    def __init__(self, initialInputParameters: dict[str, float]):
        """
        Initialize the SM model.

        Parameters
        ----------
        initialInputParameters: dict[str, float]
            A dictionary of initial input parameters for the model.

        Returns
        ----------
        cls: StandardModel
            An object of the StandardModel class.
        """

        self.modelParameters = self.calculateModelParameters(initialInputParameters)
        self.collisionParameters = self.calculateCollisionParameters(
            self.modelParameters
        )

        # Initialize internal effective potential with our params dict.
        self.effectivePotential = EffectivePotentialSM(
            self.modelParameters, self.fieldCount
        )

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    def defineParticles(self) -> None:
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

        ## === Top quark ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def topMsqVacuum(fields: Fields) -> Fields:
            return 0.5 * self.modelParameters["yt"] ** 2 * fields.GetField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"] ** 2 * fields.GetField(0)

        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuarkL = Particle(
            name="topL",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=6,
        )
        self.addParticle(topQuarkL)

        topQuarkR = Particle(
            name="topR",
            index=1,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=6,
        )
        self.addParticle(topQuarkR)

        ## === SU(2) gauge boson ===
        def WMsqVacuum(fields: Fields) -> Fields:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * fields.GetField(0) ** 2 / 4

        def WMsqDerivative(fields: Fields) -> Fields:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * fields.GetField(0) / 2

        def WMsqThermal(T: float) -> float:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * T**2 * 11.0 / 6.0

        wBoson = Particle(
            name="W",
            index=2,
            msqVacuum=WMsqVacuum,
            msqDerivative=WMsqDerivative,
            msqThermal=WMsqThermal,
            statistics="Boson",
            totalDOFs=9,
        )
        self.addParticle(wBoson)

        # TODO create collision model. Backup of in-equilibrium particles:
        """
        ## === SU(3) gluon ===
        def gluonMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 * 2.0

        gluon = Particle(
            "gluon",
            msqVacuum=0.0,
            msqDerivative=0.0,
            msqThermal=gluonMsqThermal,
            statistics="Boson",
            totalDOFs=16,
        )
        self.addParticle(gluon)

        ## === Light quarks, 5 of them ===
        def lightQuarkMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        lightQuark = Particle(
            "lightQuark",
            msqVacuum=0.0,
            msqDerivative=0.0,
            msqThermal=lightQuarkMsqThermal,
            statistics="Fermion",
            totalDOFs=60,
        )
        self.addParticle(lightQuark)
        """

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

        # Zero-temperature vev
        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0

        # Zero-temperature masses
        massH = inputParameters["mH"]
        massW = inputParameters["mW"]
        massZ = inputParameters["mZ"]
        massT = inputParameters["mt"]

        modelParameters["mW"] = massW
        modelParameters["mZ"] = massZ
        modelParameters["mt"] = massT

        # helper
        g0 = 2.0 * massW / v0

        # Gauge couplings
        modelParameters["g1"] = g0 * np.sqrt((massZ / massW) ** 2 - 1)
        modelParameters["g2"] = g0
        modelParameters["g3"] = inputParameters["g3"]
        modelParameters["yt"] = np.sqrt(1.0 / 2.0) * g0 * massT / massW

        modelParameters["lambda"] = inputParameters["mH"] ** 2 / (2 * v0**2)

        # The following parameters are defined on page 6 of hep-ph/9506475
        bconst = 3 / (64 * np.pi**2 * v0**4) * (2 * massW**4 + massZ**4 - 4 * massT**4)

        modelParameters["D"] = (
            1 / (8 * v0**2) * (2 * massW**2 + massZ**2 + 2 * massT**2)
        )
        modelParameters["E0"] = 1 / (12 * np.pi * v0**3) * (4 * massW**3 + 2 * massZ**3)

        modelParameters["T0sq"] = (
            1 / 4 / modelParameters["D"] * (massH**2 - 8 * bconst * v0**2)
        )
        modelParameters["C0"] = (
            1 / (16 * np.pi**2) * (1.42 * modelParameters["g2"] ** 4)
        )

        return modelParameters

    def calculateCollisionParameters(
        self, modelParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate the collision couplings (Lagrangian parameters) from the input
        parameters. List as they appear in the MatrixElements file
        """
        collisionParameters = {}

        collisionParameters["g3"] = modelParameters["g3"]
        collisionParameters["g2"] = modelParameters["g2"]

        return collisionParameters


class EffectivePotentialSM(EffectivePotential):
    """
    Effective potential for the Standard Model.

    This class inherits from the EffectivePotential class.
    """

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        """
        Initialize the EffectivePotentialSM.

        Parameters
        ----------
        modelParameters: dict[str, float]
            A dictionary of model parameters.
        fieldCount: int
            The number of fields undergoing the phase transition

        Returns
        ----------
        cls: EffectivePotentialSM
            an object of the EffectivePotentialSM class
        """
        super().__init__(modelParameters, fieldCount)
        # The super call already gave us the model params

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 28
        self.numFermionDof = 90

    def evaluate(  # pylint: disable=R0914
        self, fields: Fields, temperature: float, checkForImaginary: bool = False
    ) -> float | np.ndarray:
        """
        Evaluate the effective potential. We implement the effective potential
        of eq. (7) of hep-ph/9506475.

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
        # phi ~ 1/sqrt(2) (0, v)
        fields = Fields(fields)
        v = fields.GetField(0) + 0.0000001

        T = temperature + 0.0000001

        ab = 49.78019250
        af = 3.111262032

        mW = self.modelParameters["mW"]
        mZ = self.modelParameters["mZ"]
        mt = self.modelParameters["mt"]

        # FIXME type detection doesn't work here, lambdaT is interpreted as "Any".
        # Hotfix below (specify type hints manually)

        # Implement finite-temperature corrections to the modelParameters lambda,
        # C0 and E0, as on page 6 and 7 of hep-ph/9506475.
        lambdaT: float | np.ndarray = self.modelParameters["lambda"] - 3 / (
            16 * np.pi * np.pi * self.modelParameters["v0"] ** 4
        ) * (
            2 * mW**4 * np.log(mW**2 / (ab * T**2) + 1e-100)
            + mZ**4 * np.log(mZ**2 / (ab * T**2) + 1e-100)
            - 4 * mt**4 * np.log(mt**2 / (af * T**2) + 1e-100)
        )

        cT: float | np.ndarray = self.modelParameters["C0"] + 1 / (
            16 * np.pi * np.pi
        ) * (4.8 * self.modelParameters["g2"] ** 2 * lambdaT - 6 * lambdaT**2)

        # HACK: take the absolute value of lambdaT here,
        # to avoid taking the square root of a negative number
        eT: float | np.ndarray = (
            self.modelParameters["E0"]
            + 1 / (12 * np.pi) * (3 + 3**1.5) * np.abs(lambdaT) ** 1.5
        )

        potentialT: float | np.ndarray = (
            self.modelParameters["D"] * (T**2 - self.modelParameters["T0sq"]) * v**2
            - cT * T**2 * pow(v, 2) * np.log(np.abs(v / T))
            - eT * T * pow(v, 3)
            + lambdaT / 4 * pow(v, 4)
        )

        potentialTotal = np.real(potentialT + self.constantTerms(T))

        return np.asanyarray(potentialTotal)

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
        dofsBoson = self.numBosonDof
        dofsFermion = self.numFermionDof

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0


def main() -> None:  # pylint: disable=R0915, R0914
    """Runs WallGo for the SM, computing bubble wall speed."""

    WallGo.initialize()

    # loading in local config file
    WallGo.config.readINI(
        pathlib.Path(__file__).parent.resolve() / "WallGoSettings.ini"
    )

    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    # Guess of the wall thickness: (approximately) 10/Tn
    wallThicknessIni = 0.1

    # Estimate of the mean free path of the particles in the plasma:
    # (approximately) 100/Tn
    meanFreePath = 1

    # The following 2 parameters are used to estimate the optimal value of dT used
    # for the finite difference derivatives of the potential.
    # Temperature scale (in GeV) over which the potential changes by O(1).
    # A good value would be of order Tc-Tn.
    temperatureScale = 0.5
    # Field scale (in GeV) over which the potential changes by O(1). A good value
    # would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the
    # same scale, or an array.
    fieldScale = (50.0,)

    ## Create WallGo control object
    manager = WallGoManager(
        wallThicknessIni, meanFreePath, temperatureScale, fieldScale
    )

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input,
    but this is likely to change in the future
    """

    # QFT model input. Some of these are probably not intended to change,
    # like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 50.0,
    }

    model = StandardModel(inputParameters)

    ## ---- collision integration and path specifications

    # automatic generation of collision integrals is disabled by default
    # set to "False" or comment if collision integrals already exist
    # set to "True" to invoke automatic collision integral generation
    WallGo.config.config.set("Collisions", "generateCollisionIntegrals", "False")

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    try:
        scriptLocation = pathlib.Path(__file__).parent.resolve()
        manager.loadCollisionFiles(scriptLocation / "CollisionOutput")
    except Exception:
        print(
            """\nLoad of collision integrals failed! This example files comes with pre-generated collision files for N=5 and N=11,
              so load failure here probably means you've either moved files around or changed the grid size.
              If you were trying to generate your own collision data, make sure you run this example script with the --recalculateCollisions command line flag.
              """
        )
        exit(2)

    print("\n=== WallGo parameter scan ===")
    # ---- This is where you'd start an input parameter
    # loop if doing parameter-space scans ----

    """ Example mass loop that does five values of mH. Note that the WallGoManager
    class is NOT thread safe internally, so it is NOT safe to parallelize this loop 
    eg. with OpenMP. We recommend ``embarrassingly parallel`` runs for large-scale
    parameter scans. 
    """

    valuesMH = [0.0, 50.0, 68.0, 79.0, 88.0]
#    valuesMH = [79.0]
    valuesTn = [57.192, 83.426, 100.352, 111.480, 120.934]
#    valuesTn = [111.480]

    for i in range(len(valuesMH)):  # pylint: disable=C0200
        print(
            f"== Begin Benchmark with mH = {valuesMH[i]} GeV, Tn = {valuesTn[i]} GeV =="
        )

        inputParameters["mH"] = valuesMH[i]

        manager.changeInputParameters(inputParameters, EffectivePotentialSM)

        """In addition to model parameters, WallGo needs info about the phases at
        nucleation temperature. Use the WallGo.PhaseInfo dataclass for this purpose.
        Transition goes from phase1 to phase2.
        """
        Tn = valuesTn[i]

        phaseInfo = WallGo.PhaseInfo(
            temperature=Tn,
            phaseLocation1=WallGo.Fields([0.0]),
            phaseLocation2=WallGo.Fields([Tn]),
        )

        """Give the input to WallGo. It is NOT enough to change parameters
           directly in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific
               initializations of internal classes
        """
        manager.setParameters(phaseInfo)

        """WallGo can now be used to compute wall stuff!"""

        ## ---- Solve wall speed in Local Thermal Equilibrium approximation

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

#        print("\n=== Search for detonation solution ===")
#        wallGoDetonationResults = manager.solveWallDetonation(onlySmallest=True)[0]
#        print("\n=== Detonation results ===")
#        print(f"wallVelocity:      {wallGoDetonationResults.wallVelocity}")

    # end parameter-space loop


# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
