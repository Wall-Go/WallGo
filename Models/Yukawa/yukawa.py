"""
A simple example model, of a real scalar field coupled to a Dirac fermion
c.f. 2310.02308
"""

import sys
import pathlib
import numpy as np
from typing import TYPE_CHECKING

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo.initialize()
from WallGo import Fields, GenericModel, Particle

# Add the Models folder to the path; need to import the base
# example template and effectivePotentialNoResum.py
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase
from wallGoExampleBase import ExampleInputPoint

if TYPE_CHECKING:
    import WallGoCollision


class YukawaModel(GenericModel):
    """
    The Yukawa model, inheriting from WallGo.GenericModel.
    """

    def __init__(self) -> None:
        """
        Initialize the Yukawa model.

        Parameters
        ----------

        Returns
        ----------
        cls: YukawaModel
            An object of the YukawaModel class.
        """

        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialYukawa(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def getEffectivePotential(self) -> "EffectivePotentialYukawa":
        return self.effectivePotential

    # ~

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

        # === left fermion ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def psiMsqVacuum(fields: Fields) -> Fields:
            return self.modelParameters["mf"] + self.modelParameters[
                "y"
            ] * fields.getField(0)

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def psiMsqDerivative(fields: Fields) -> Fields:  # pylint: disable = W0613
            return self.modelParameters["y"]

        def psiMsqThermal(T: float) -> float:
            return 1 / 16 * self.modelParameters["y"] ** 2 * T**2

        psiL = Particle(
            "psiL",
            index=1,  # old collision data has top at index 0
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            totalDOFs=4,
        )
        psiR = Particle(
            "psiR",
            index=2,  # old collision data has top at index 0
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            totalDOFs=4,
        )
        self.addParticle(psiL)
        self.addParticle(psiR)

    def calculateLagrangianParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate Lagrangian parameters based on the input parameters.

        Parameters
        ----------
        inputParameters: dict[str, float]
            A dictionary of input parameters for the model.

        Returns
        ----------
        modelParameters: dict[str, float]
            A dictionary of calculated model parameters.
        """
        modelParameters = {}

        # Parameters for "phi" field
        modelParameters = inputParameters

        return modelParameters

    def updateModel(self, newInputParams: dict[str, float]) -> None:
        """Computes new Lagrangian parameters from given input and caches
        them internally. These changes automatically propagate to the
        associated EffectivePotential, particle masses etc.
        """
        newParams = self.calculateLagrangianParameters(newInputParams)
        # Copy to the model dict, do NOT replace the reference.
        # This way the changes propagate to Veff and particles
        self.modelParameters.update(newParams)


class EffectivePotentialYukawa(WallGo.EffectivePotential):
    """
    Effective potential for the Yukawa model.

    This class inherits from the EffectivePotential class and provides the
    necessary methods for calculating the effective potential.
    """

    def __init__(self, owningModel: YukawaModel) -> None:
        """
        Initialize the EffectivePotentialYukawa.
        """

        super().__init__()

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        print(self.modelParameters)

    # ~ EffectivePotential interface
    fieldCount = 1
    """How many classical background fields"""
    # ~

    def evaluate(
        self, fields: Fields, temperature: float, checkForImaginary: bool = False
    ) -> float | np.ndarray:
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
        potentialTotal: float | np.ndarray
            The value of the effective potential
        """
        # getting the field from the list of fields (here just of length 1)
        fields = WallGo.Fields(fields)
        phi = fields.getField(0)

        # the constant term
        f0 = -np.pi**2 / 90 * (1 + 4 * 7 / 8) * temperature**4

        # coefficients of the temperature and field dependent terms
        y = self.modelParameters["y"]
        mf = self.modelParameters["mf"]
        sigmaEff = (
            self.modelParameters["sigma"]
            + 1 / 24 * (self.modelParameters["gamma"] + 4 * y * mf) * temperature**2
        )
        msqEff = (
            self.modelParameters["msq"]
            + 1 / 24 * (self.modelParameters["lam"] + 4 * y**2) * temperature**2
        )

        potentialTotal = (
            f0
            + sigmaEff * phi
            + 1 / 2 * msqEff * phi**2
            + 1 / 6 * self.modelParameters["gamma"] * phi**3
            + 1 / 24 * self.modelParameters["lam"] * phi**4
        )

        return np.array(potentialTotal)


class YukawaModelExample(WallGoExampleBase):
    """
    Sets up the Yukawa model, computes or loads the collison
    integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateCollisions = False

        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory
            / "MatrixElements/MatrixElements_Yukawa.json"
        )

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return YukawaModel()

    def initCollisionModel(
        self, wallGoModel: "YukawaModel"
    ) -> "WallGoCollision.PhysicsModel":
        """Initialize the Collision model and set the seed."""

        import WallGoCollision  # pylint: disable = C0415

        # Collision integrations utilize Monte Carlo methods, so RNG is involved.
        # We can set the global seed for collision integrals as follows.
        # This is optional; by default the seed is 0.
        WallGoCollision.setSeed(0)

        collisionModelDefinition = (
            WallGo.collisionHelpers.generateCollisionModelDefinition(wallGoModel)
        )

        # Add in-equilibrium particles that appear in collision processes
        phiParticle = WallGoCollision.ParticleDescription()
        phiParticle.name = "phi"
        phiParticle.index = 0
        phiParticle.bInEquilibrium = True
        phiParticle.bUltrarelativistic = True
        phiParticle.type = WallGoCollision.EParticleType.eBoson
        # mass-sq function not required or used for UR particles,
        # and it cannot be field-dependent for collisions.
        # Backup of what the vacuum mass was intended to be:
        """
        msqVacuum=lambda fields: (
                msq + g * fields.getField(0) + lam / 2 * fields.getField(0) ** 2
            ),
        """

        parameters = WallGoCollision.ModelParameters()

        parameters.addOrModifyParameter("y", wallGoModel.modelParameters["y"])
        parameters.addOrModifyParameter("gamma", wallGoModel.modelParameters["gamma"])
        parameters.addOrModifyParameter("lam", wallGoModel.modelParameters["lam"])
        parameters.addOrModifyParameter("v", 0.0)

        parameters.addOrModifyParameter(
            "msq[0]", 1 / 16 * wallGoModel.modelParameters["y"] ** 2
        )  # phi thermal mass^2 in units of T
        parameters.addOrModifyParameter(
            "msq[1]",
            wallGoModel.modelParameters["lam"] / 24.0
            + wallGoModel.modelParameters["y"] ** 2.0 / 6.0,
        )  # psi thermal mass^2 in units of T

        collisionModelDefinition.defineParticleSpecies(phiParticle)
        collisionModelDefinition.defineParameters(parameters)

        collisionModel = WallGoCollision.PhysicsModel(collisionModelDefinition)

        return collisionModel

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:
        """Yukawa example uses spatial grid size = 20"""
        super().configureManager(inOutManager)
        inOutManager.config.set("PolynomialGrid", "spatialGridSize", "20")

    def updateModelParameters(
        self, model: "YukawaModel", inputParameters: dict[str, float]
    ) -> None:
        """Update internal model parameters. This example is constructed so
        that the effective potential and particle mass functions refer to
        model.modelParameters, so be careful not to replace that reference here.
        """

        # oldParams = model.modelParameters.copy()
        model.updateModel(inputParameters)

    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        """
        Input parameters, phase info, and settings for the effective potential and
        wall solver for the Yukawa benchmark point.
        """

        output: list[ExampleInputPoint] = []
        output.append(
            ExampleInputPoint(
                {
                    "sigma": 0,
                    "msq": 1,
                    "gamma": -1.28565864794053,
                    "lam": 0.01320208496444000,
                    "y": -0.177421729274665,
                    "mf": 2.0280748307391000,
                },
                WallGo.PhaseInfo(
                    temperature=89.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([30.79]),
                    phaseLocation2=WallGo.Fields([192.35]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureScale=1.0,
                    fieldScale=[
                        100.0,
                    ],
                ),
                WallGo.WallSolverSettings(
                    # we actually do both cases in the common example
                    bIncludeOffEquilibrium=True,
                    meanFreePath=1.0,
                    wallThicknessGuess=0.05,
                ),
            )
        )

        return output

    # ~ End WallGoExampleBase interface


if __name__ == "__main__":

    example = YukawaModelExample()
    example.runExample()
