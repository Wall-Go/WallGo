import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing
from pathlib import Path

import WallGo

if typing.TYPE_CHECKING:
    import WallGoCollision


@dataclass
class ExampleInputPoint:
    modelParameters: dict[str, float]
    phaseInfo: WallGo.PhaseInfo


class WallGoExampleBase(ABC):

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        None

    @abstractmethod
    def initWallGoManager(self) -> "WallGo.WallGoManager":
        pass

    @abstractmethod
    def initCollisionModel(self) -> "WallGoCollision.PhysicsModel":
        pass

    @abstractmethod
    def initWallGoModel(self) -> "WallGo.GenericModel":
        pass

    @abstractmethod
    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        return []

    def setupDirectoryPaths(self) -> None:
        self.exampleBaseDir = Path(__file__).resolve()
        self.matrixElementsDir = self.exampleBaseDir / "MatrixElements"

    def initCommandLineArgs(self) -> argparse.ArgumentParser:
        argParser = argparse.ArgumentParser()

        argParser.add_argument(
            "--momentumBasisSize",
            help="Basis size N for momentum grid",
            type=int,
            default=3,
        )

        argParser.add_argument(
            "--recalculateMatrixElements",
            help="Forces full recalculation of matrix elements via DRalgo.",
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

        return argParser

    def runExample(self) -> None:

        WallGo.initialize()

        argParser = self.initCommandLineArgs()
        cmdArgs = argParser.parse_args()

        manager = self.initWallGoManager()
        model = self.initWallGoModel()
        manager.registerModel(model)

        if cmdArgs.recalculateCollisions:
            import WallGoCollision

            collisionModel = self.initCollisionModel()
            collisionModel.readMatrixElements(self.matrixElementsDir, True)

            collisionTensor = collisionModel.createCollisionTensor(3)

            self.configureCollisionIntegration(collisionTensor)

            """Run the collision integrator. This is a very long running function: For M out-of-equilibrium particle species and momentum grid size N,
            there are order M^2 x (N-1)^4 integrals to be computed. In your own runs you may want to handle this part in a separate script and offload it eg. to a cluster,
            especially if using N >> 11.
            """
            print(
                "Entering collision integral computation, this may take long",
                flush=True,
            )
            collisionResults: WallGoCollision.CollisionTensorResult = (
                collisionTensor.computeIntegralsAll()
            )

            """Export the collision integration results to .hdf5. "individual" means that each off-eq particle pair gets its own file.
            This format is currently required for the main WallGo routines to understand the data. 
            """
            collisionDirectory = (
                self.exampleBaseDir
                / f"CollisionOutput_N{cmdArgs.momentumBasisSize}_UserGenerated"
            )
            collisionResults.writeToIndividualHDF5(str(collisionDirectory))

            ## TODO we could convert the CollisionTensorResult object from above to CollisionArray directly instead of forcing write hdf5 -> read hdf5

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

        benchmarkPoints = self.getBenchmarkPoints()

        for benchmark in benchmarkPoints:
            manager.changeInputParameters(benchmark.modelParameters)
            ## FIXME whats up with the name of this function??
            manager.setParameters(benchmark.phaseInfo)

            # ---- Solve wall speed in Local Thermal Equilibrium (LTE) approximation
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
            wallGoInterpolationResults = manager.solveWallDetonation()
            print("\n=== Detonation results ===")
            print(f"wallVelocity:      {wallGoInterpolationResults.wallVelocities}")
