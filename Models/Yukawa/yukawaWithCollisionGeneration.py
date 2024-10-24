"""
A simple example model, of a real scalar field coupled to a Dirac fermion
c.f. 2310.02308
"""

import sys
import pathlib
from typing import TYPE_CHECKING
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import Fields, GenericModel, Particle

# Add the Yukawa folder to the path to import YukawaModel
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from yukawa import YukawaModel

# Add the Models folder to the path; need to import the base
# example template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint  # pylint: disable=C0411, C0413, E0401

if TYPE_CHECKING:
    import WallGoCollision


class YukawaModelExample(WallGoExampleBase):
    """
    Sets up the Yukawa model, computes or loads the collison
    integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateMatrixElements = False

        self.bShouldRecalculateCollisions = False

        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/MatrixElements_Yukawa.json"
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
        # The out-of-equilibrium particles are taken from the definition in the model file
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

        parameters.add("y", wallGoModel.modelParameters["y"])
        parameters.add("gamma", wallGoModel.modelParameters["gamma"])
        parameters.add("lam", wallGoModel.modelParameters["lam"])
        parameters.add("v", 0.0)

        parameters.add(
            "mf2", 1 / 16 * wallGoModel.modelParameters["y"] ** 2
        )  # phi thermal mass^2 in units of T
        parameters.add(
            "ms2",
            +wallGoModel.modelParameters["lam"] / 24.0
            + wallGoModel.modelParameters["y"] ** 2.0 / 6.0,
        )  # psi thermal mass^2 in units of T

        collisionModelDefinition.defineParticleSpecies(phiParticle)
        collisionModelDefinition.defineParameters(parameters)

        collisionModel = WallGoCollision.PhysicsModel(collisionModelDefinition)

        return collisionModel

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        """Non-abstract override"""

        import WallGoCollision  # pylint: disable = C0415

        """We can also configure various verbosity settings that are useful when
        you want to see what is going on in long-running integrations. These 
        include progress reporting and time estimates, as well as a full result dump
        of each individual integral to stdout. By default these are all disabled. 
        Here we enable some for demonstration purposes.
        """
        verbosity = WallGoCollision.CollisionTensorVerbosity()
        verbosity.bPrintElapsedTime = (
            True  # report total time when finished with all integrals
        )

        """Progress report when this percentage of total integrals (approximately)
        have been computed. Note that this percentage is per-particle-pair, ie. 
        each (particle1, particle2) pair reports when this percentage of their
        own integrals is done. Note also that in multithreaded runs the 
        progress tracking is less precise.
        """
        verbosity.progressReportPercentage = 0.25

        # Print every integral result to stdout? This is very slow and
        # verbose, intended only for debugging purposes
        verbosity.bPrintEveryElement = False

        inOutCollisionTensor.setIntegrationVerbosity(verbosity)

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:

        super().configureManager(inOutManager)

        # Change the amount of grid points in the spatial coordinates
        # for faster computations
        inOutManager.config.configGrid.spatialGridSize = 20
        # Increase the number of iterations in the wall solving to 
        # ensure convergence
        inOutManager.config.configEOM.maxIterations = 25
        # Decrease error tolerance for phase tracing to ensure stability
        inOutManager.config.configThermodynamics.phaseTracerTol = 1e-8

    def updateModelParameters(
        self, model: "YukawaModel", inputParameters: dict[str, float]
    ) -> None:
        """Update internal model parameters. This example is constructed so
        that the effective potential and particle mass functions refer to
        model.modelParameters, so be careful not to replace that reference here.
        """

        newParams = inputParameters
        # Copy to the model dict, do NOT replace the reference.
        # This way the changes propagate to Veff and particles
        model.modelParameters.update(newParams)

    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        """
        Input parameters, phase info, and settings for the effective potential and
        wall solver for the Yukawa benchmark point.
        """

        output: list[ExampleInputPoint] = []
        output.append(
            ExampleInputPoint(
                {
                    "sigma": 0.0,
                    "msq": 1.0,
                    "gamma": -1.2,
                    "lam": 0.10,
                    "y": 0.55,
                    "mf": 0.30,
                },
                WallGo.PhaseInfo(
                    temperature=8.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([0.4]),
                    phaseLocation2=WallGo.Fields([27.0]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=1.0,
                    fieldValueVariationScale=[
                        100.0,
                    ],
                ),
                WallGo.WallSolverSettings(
                    # we actually do both cases in the common example
                    bIncludeOffEquilibrium=True,
                    # meanFreePathScale is determined here by the annihilation channels,
                    # and scales inversely with y^4 or lam^2. This is why
                    # meanFreePathScale has to be so large.
                    meanFreePathScale=10000.0,  # In units of 1/Tnucl
                    wallThicknessGuess=10.0,  # In units of 1/Tnucl
                ),
            )
        )

        return output

    # ~ End WallGoExampleBase interface


if __name__ == "__main__":

    example = YukawaModelExample()
    example.runExample()
