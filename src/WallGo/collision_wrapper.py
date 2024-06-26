import os
import sys
from pathlib import Path
import importlib
from types import ModuleType

# TODO here only need the current config instead of whole wallgo package
import WallGo  ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import Fields

# from .Particle import Particle


class Collision:
    """Thin wrapper around the C++ module. This handles loading of the module, provides Python-readable type hints etc.
    This class is a singleton.
    """

    _instance = None

    def __new__(cls, modelCls: GenericModel) -> "Collision":
        """
        Implement singleton pattern
        Create a new instance of the class if it doesn't already exist.

        Args:
            modelCls (GenericModel): The model class to be wrapped.

        Returns:
            The instance of the class.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, modelCls: GenericModel):
        """
        Initializes the CollisionWrapper object.

        Args:
            modelCls (GenericModel):
            The model class to be used for collision integrations.

        Returns:
            None
        """
        if not hasattr(self, "bInitialized"):
            self.module: ModuleType = None
            self._loadCollisionModule(modelCls)
            self.bInitialized = True
            self.generateCollisionIntegrals = WallGo.config.getboolean("Collisions", "generateCollisionIntegrals")
            print(self.generateCollisionIntegrals)

            scriptLocation = Path(sys.argv[0]).parent.resolve()
            collisionDirectory = scriptLocation / WallGo.config.get("Collisions", "pathName")
            collisionDirectory.mkdir(parents=True, exist_ok=True)

            self.setOutputDirectory(str(collisionDirectory))

    def setSeed(self, seed: int) -> None:
        """Set seed for the Monte Carlo integration. Default is 0.

        Args:
            seed (int): The seed value to set for the Monte Carlo integration.

        Returns:
            None
        """
        # TODO do not need assert here
        if self.generateCollisionIntegrals:
            self._assertLoaded()
            self.module.setSeed(seed)
        else:
            pass

    def _loadCollisionModule(self, modelCls: GenericModel) -> None:
        """Load the collision module.

        Raises:
            ImportError: If the module fails to load.
        """
        try:
            currentDirectory = os.path.dirname(__file__)

            ## path relative to the above
            relativePathToModule = "../../Collision/pybind/lib"

            collisionModulePath = os.path.join(currentDirectory, relativePathToModule)

            sys.path.append(collisionModulePath)

            moduleName = "WallGoCollisionPy"

            self.module = importlib.import_module(moduleName)
            print(f"Loaded module [{moduleName}]")

            ## Construct a "control" object for collision integrations
            # Use help(Collision.manager) for info about what functionality is available
            self.manager = self.module.CollisionManager()

            self.addParticles(modelCls)

        except ImportError:
            print(
                f"Warning: Failed to load [{moduleName}]. "
                "Using read-only mode for collision integrals."
            )
            print("Computation of new collision integrals will NOT be possible.")
        ## Should we assert that the load succeeds?
        # If the user creates this class in the first place, they presumably want to use the module

    def _assertLoaded(self) -> None:
        """Assert that the collision module has been loaded.

        Raises:
            AssertionError: If the collision module has not been loaded.
        """
        assert self.module is not None, "Collision module has not been loaded!"

    def addParticles(self, model: "WallGo.GenericModel", T: float = 1.0) -> None:
        """
        Adds particles to the collision module.

        Args:
            model (WallGo.GenericModel): The model containing the particles.
            T (float, optional): The temperature in GeV units. Defaults to 1.0.

        Returns:
            None

        Notes:
            - Particles need masses in GeV units, i.e., T dependent.
            - Thermal masses are rescaled by the temperature and the default argument of T = 1.
            - This needs to be adapted for non-zero vacuum masses.
            - Register particles with the collision module. This is required for each particle that can appear in matrix elements,
              including particle species that are assumed to stay in equilibrium.
            - The order of registration is the same as the particles are defined in model.particles which
              should be the same as in MatrixElements.txt.
        """
        fieldHack = Fields([0] * model.fieldCount)

        for particle in model.particles:
            self.manager.addParticle(
                self.constructPybindParticle(particle, T, fieldHack)
            )

    def constructPybindParticle(
        self, particle: Particle, T: float, fields: Fields
    ) -> "CollisionModule.ParticleSpecies":
        """
        Converts python 'Particle' object to pybind-bound ParticleSpecies object that the Collision module can understand.
        'Particle' uses masses in GeV^2 units while CollisionModule operates with dimensionless (m/T)^2 etc,
        so the temperature is taken as an input here, and the masses are rescaled accordingly.
        The same should be done for field values since the vacuum mass can depend on that.

        Parameters
        ----------
        particle : Particle
            Particle object with p.msqVacuum and p.msqThermal being in GeV^2 units.
        T : float
            Temperature in GeV units.
        fields : Fields
            Fields object representing the fields in the system.

        Returns
        -------
        CollisionModule.ParticleSpecies
            ParticleSpecies object representing the converted particle.

        """
        ## Convert to correct enum for particle statistics
        particleType = None
        if particle.statistics == "Boson":
            particleType = self.module.EParticleType.BOSON
        elif particle.statistics == "Fermion":
            particleType = self.module.EParticleType.FERMION

        ## Hack vacuum masses are ignored
        return self.module.ParticleSpecies(
            particle.name,
            particleType,
            particle.inEquilibrium,
            particle.msqVacuum(fields) / T**2.0,
            particle.msqThermal(T) / T**2.0,
            particle.ultrarelativistic,
        )

    def calculateCollisionIntegrals(self, bVerbose: bool = False) -> None:
        """
        Calculates the collision integrals.

        Args:
            bVerbose (bool, optional):
            If True, prints verbose output. Defaults to False.
        """
        ## Make sure this is >= 0.
        # The C++ code requires uint so pybind11 will throw TypeError otherwise
        basisSize = WallGo.config.getint("PolynomialGrid", "momentumGridSize")
        self.manager.calculateCollisionIntegrals(basisSize, bVerbose)

    def setOutputDirectory(self, outputDirectory: str) -> None:
        """
        Sets the output directory for the collision integrals.

        Args:
            outputDirectory (str): The output directory to set.

        Returns:
            None
        """
        self.outputDirectory = outputDirectory
        if self.module is not None:
            self.manager.setOutputDirectory(str(self.outputDirectory))
    
    def addCoupling(self, coupling: float) -> None:
        """
        Adds a coupling to the collision module.

        Args:
            coupling (float): The coupling to add.

        Returns:
            None
        """
        if self.generateCollisionIntegrals:
            self._assertLoaded()
            self.manager.addCoupling(coupling)
        else:
            pass