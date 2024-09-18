import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
import typing
from pathlib import Path
import copy

import WallGo

if typing.TYPE_CHECKING:
    import WallGoCollision


@dataclass
class ExampleInputPoint:
    inputParameters: dict[str, float]
    phaseInfo: WallGo.PhaseInfo
    veffDerivativeScales: WallGo.VeffDerivativeScales
    wallSolverSettings: WallGo.WallSolverSettings


class WallGoExampleBase(ABC):

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        None

    @abstractmethod
    def initCollisionModel(self) -> "WallGoCollision.PhysicsModel":
        pass

    @abstractmethod
    def initWallGoModel(self) -> "WallGo.GenericModel":
        pass

    @abstractmethod
    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        return []

    @abstractmethod
    def updateModelParameters(
        self, model: "WallGo.GenericModel", inputParameters: dict[str, float]
    ) -> None:
        """Override with whatever non-WallGo logic your example model needs to run
        when model-specific inputs are changed. Normally this refers to conversion from "physical" input
        (such as electroweak precision observables) to renormalized Lagrangian parameters,
        and/or propagating the changes to the effective potential, particle masses, collision model etc.
        """
        pass

    @property
    @abstractmethod
    def exampleBaseDirectory(self) -> Path:
        """Override to return base directory of this example."""
        pass

    @property
    def defaultMatrixElementPath(self) -> Path:
        """Path to default matrix elements file for the example.
        This is relative to exampleBaseDirectory."""
        return Path("MatrixElements/MatrixElements.txt")

    @property
    def defaultCollisionDirectory(self) -> Path:
        """Path to the directory containing default collision data for the example.
        This is relative to exampleBaseDirectory"""
        return Path("CollisionOutput")

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
        """"""
        WallGo.initialize()

        argParser = self.initCommandLineArgs()
        self.cmdArgs = argParser.parse_args()

        manager = WallGo.WallGoManager()
        model = self.initWallGoModel()
        manager.registerModel(model)

        if self.cmdArgs.recalculateCollisions:
            import WallGoCollision

            collisionModel = self.initCollisionModel()
            collisionModel.readMatrixElements(
                self.exampleBaseDirectory / self.defaultMatrixElementPath, True
            )

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
                self.exampleBaseDirectory
                / f"CollisionOutput_N{self.cmdArgs.momentumBasisSize}_UserGenerated"
            )
            collisionResults.writeToIndividualHDF5(str(collisionDirectory))

            ## TODO we could convert the CollisionTensorResult object from above to CollisionArray directly instead of forcing write hdf5 -> read hdf5
        else:
            collisionDirectory = (
                self.exampleBaseDirectory / self.defaultCollisionDirectory
            )

        # Specify where to load collision files from. The manager will load them when needed by the internal Boltzmann solver
        manager.setPathToCollisionData(collisionDirectory)

        # TODO catch load error nicely, it's less trivial now that the loading is hidden deep in manager
        # print(
        #    """\nLoad of collision integrals failed! WallGo example models come with pre-generated collision files,
        #    so load failure here probably means you've either moved files around or changed to incompatible grid size.
        #    If you were trying to generate your own collision data, make sure you run this example script with the --recalculateCollisions command line flag.
        #    """
        # )
        # exit(42)

        benchmarkPoints = self.getBenchmarkPoints()

        for benchmark in benchmarkPoints:

            """Update model parameters. Our examples store them internally in the model, through which they propagate to the effective potential.
            WallGo is not directly aware of model-specific parameters; it only requires EffectivePotential.evaluate() to be valid at field, temperature input,
            and similarly for particle masses. TODO update collision model
            """
            self.updateModelParameters(model, benchmark.inputParameters)

            # This needs to run before wallSpeedLTE() or solveWall(), as it does a lot of internal caching related to hydrodynamics

            """WallGo needs info about the phases at
            nucleation temperature. Use the WallGo.PhaseInfo dataclass for this purpose.
            Transition goes from phase1 to phase2.
            """
            manager.analyzeHydrodynamics(
                benchmark.phaseInfo,
                benchmark.veffDerivativeScales,
            )

            # ---- Solve wall speed in Local Thermal Equilibrium (LTE) approximation
            vwLTE = manager.wallSpeedLTE()
            print(f"LTE wall speed:    {vwLTE:.6f}")

            # ---- Solve field EOM. For illustration, first solve it without any
            # out-of-equilibrium contributions. The resulting wall speed should
            # be close to the LTE result

            wallSolverSettings = copy.deepcopy(benchmark.wallSolverSettings)

            wallSolverSettings.bIncludeOffEquilibrium = False
            print(f"\n=== Begin EOM with off-eq effects ignored ===")
            results = manager.solveWall(wallSolverSettings)

            print("\n=== Local equilibrium results ===")
            print(f"wallVelocity:      {results.wallVelocity:.6f}")
            print(f"wallVelocityError: {results.wallVelocityError:.6f}")
            print(f"wallWidths:        {results.wallWidths}")
            print(f"wallOffsets:       {results.wallOffsets}")

            # Repeat with out-of-equilibrium parts included. This requires
            # solving Boltzmann equations, invoked automatically by solveWall()
            wallSolverSettings.bIncludeOffEquilibrium = True

            print(f"\n=== Begin EOM with off-eq effects included ===")
            results = manager.solveWall(wallSolverSettings)

            print("\n=== Out-of-equilibrium results ===")
            print(f"wallVelocity:      {results.wallVelocity:.6f}")
            print(f"wallVelocityError: {results.wallVelocityError:.6f}")
            print(f"wallWidths:        {results.wallWidths}")
            print(f"wallOffsets:       {results.wallOffsets}")

            print("\n=== Search for detonation solution ===")
            results = manager.solveWallDetonation(wallSolverSettings)
            print(f"\n=== Detonation results, {len(results)} solutions found ===")
            for res in results:
                print(f"wallVelocity:      {res.wallVelocity}")
