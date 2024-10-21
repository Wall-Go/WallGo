"""
A simple model of a scalar coupled to an out-of-equilibrium fermion. 
The model uses the same matrix elements as the Yukawa model,
but the potential has been modified (the coefficients no longer
correspond to a real physical model).
The couplings entering in the matrix elements are also chosen differently
from the real Yukawa model.
"""

import sys
import pathlib
from typing import TYPE_CHECKING
import numpy as np

# WallGo imports
import WallGo

# Add the SimpleModel folder to the path to import SimpleModel
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from simpleModel import SimpleModel

# Add the Models folder to the path; need to import the base
# example template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint  # pylint: disable=C0411, C0413, E0401

if TYPE_CHECKING:
    import WallGoCollision


class SimpleModelExample(WallGoExampleBase):
    """
    Sets up the model, computes or loads the collison
    integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateMatrixElements = False

        self.bShouldRecalculateCollisions = False

        # We take the matrix elements from the Yukawa model
        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory
            / "MatrixElements/MatrixElements_Yukawa.json"
        )

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return SimpleModel()

    def initCollisionModel(
        self, wallGoModel: "SimpleModel"
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

        parameters = WallGoCollision.ModelParameters()

        # We use a different "y" in the collisions than the Yukawa model,
        # therefore, we rename it to kappa
        parameters.add("y", wallGoModel.modelParameters["kappa"])
        parameters.add("gamma", wallGoModel.modelParameters["gamma"])

        parameters.add(
            "mf2", 1 / 16 * wallGoModel.modelParameters["yf"] ** 2
        )  # phi thermal mass^2 in units of T
        parameters.add(
            "ms2", wallGoModel.modelParameters["msqTh"]
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
        """We use a spatial grid size = 20"""
        super().configureManager(inOutManager)
        inOutManager.config.set("PolynomialGrid", "spatialGridSize", "20")

    def updateModelParameters(
        self, model: "SimpleModel", inputParameters: dict[str, float]
    ) -> None:
        """Update internal model parameters. This example is constructed so
        that the effective potential and particle mass functions refer to
        model.modelParameters, so be careful not to replace that reference here.
        """
        newParams = inputParameters
        #Copy to the model dict, do NOT replace the reference.
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
                    "yf": -0.18,
                    "mf": 2.0,
                    "gamma": -4.0,
                    "kappa": -0.6,
                    "msq": 1,
                    "msqTh": 2./115.,
                    "cubic": -0.77,
                    "quartic": 0.0055,
                },
                WallGo.PhaseInfo(
                    temperature=51.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([0.]),
                    phaseLocation2=WallGo.Fields([82.5223]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=1.0,
                    fieldValueVariationScale=[
                        50.0,
                    ],
                ),
                WallGo.WallSolverSettings(
                    # we actually do both cases in the common example
                    bIncludeOffEquilibrium=True,
                    meanFreePathScale=1000.0, # In units of 1/Tnucl
                    wallThicknessGuess=5.0, # In units of 1/Tnucl
                ),
            )
        )

        return output

    # ~ End WallGoExampleBase interface


if __name__ == "__main__":

    example = SimpleModelExample()
    example.runExample()
