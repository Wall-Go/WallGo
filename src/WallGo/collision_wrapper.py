
import os, sys
import importlib
from types import ModuleType

# TODO here only need the current config instead of whole wallgo package
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import Fields
# from .Particle import Particle

class Collision():
    """Thin wrapper around the C++ module. This handles loading of the module, provides Python-readable type hints etc.
    This class is a singleton.
    """
    _instance = None
    
    def __new__(cls, modelCls: GenericModel):
        # Implement singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
    def __init__(self, modelCls: GenericModel):
        if not hasattr(self, "bInitialized"):
            self.module: ModuleType = None
            self._loadCollisionModule()
            self.bInitialized = True

            ## Construct a "control" object for collision integrations
            # Use help(Collision.manager) for info about what functionality is available
            self.manager = self.module.CollisionManager()

            """
            Register particles with the collision module. This is required for each particle that can appear in matrix elements,
            including particle species that are assumed to stay in equilibrium.
            The order here should be the same as in the matrix elements and how they are introduced in the model file
            """
            self.addParticles(modelCls)
    
    def setSeed(self, seed: int) -> None:
        """Set seed for the Monte Carlo integration. Default is 0.
        """
        self._assertLoaded()
        self.module.setSeed(seed)


    def _loadCollisionModule(self) -> None:

        try: 
            currentDirectory = os.path.dirname(__file__)

            ## path relative to the above
            relativePathToModule = "../../Collision/pybind/lib"

            collisionModulePath = os.path.join(currentDirectory, relativePathToModule)

            sys.path.append(collisionModulePath)

            moduleName = "WallGoCollisionPy"
            
            self.module = importlib.import_module(moduleName)
            print("Loaded module [%s]" % moduleName)

        except ImportError:
            print("Warning: Failed to load [%s]. Using read-only mode for collision integrals." % moduleName)
            print("Computation of new collision integrals will NOT be possible.")
        ## Should we assert that the load succeeds? If the user creates this class in the first place, they presumably want to use the module


    def _assertLoaded(self) -> None:
        assert self.module is not None, "Collision module has not been loaded!"

    """
    Register particles with the collision module. This is required for each particle that can appear in matrix elements,
    including particle species that are assumed to stay in equilibrium.
    The order here should be the same as in the matrix elements and how they are introduced in the model file
    """
    def addParticles(self, model: GenericModel, T = 1.0) -> None:
        """
        Particles need masses in GeV units, ie. T dependent, but for this example we don't really have 
        a temperature. So hacking this by setting T = 1. Also, for this example the vacuum mass = 0
        """
        fieldHack = Fields([0]*model.fieldCount)
        for particle in model.particles:
            self.manager.addParticle( self.constructPybindParticle(particle, T, fieldHack) )

    ## Convert Python 'Particle' object to pybind-bound ParticleSpecies object.
    ## But 'Particle' uses masses in GeV^2 units while we need m^2/T^2, so T is needed as input here.
    ## Should do the same for field values since the vacuum mass can depend on that.
    ## Return value is a ParticleSpecies object
    def constructPybindParticle(self, particle: Particle, T: float, fields: Fields):
        r"""
            Converts 'Particle' object to ParticleSpecies object that the Collision module can understand.
            CollisionModule operates with dimensionless (m/T)^2 etc, so the temperature is taken as an input here. 

            Parameters
            ----------
            particle : Particle
                Particle object with p.msqVacuum and p.msqThermal being in GeV^2 units.
            T : float
                Temperature in GeV units.

            Returns
            -------
            CollisionModule.ParticleSpecies
                ParticleSpecies object
        """


        ## Convert to correct enum for particle statistics
        particleType = None
        if particle.statistics == "Boson":
            particleType = self.module.EParticleType.BOSON
        elif particle.statistics == "Fermion":
            particleType =  self.module.EParticleType.FERMION

        ## Hack vacuum masses are ignored
        return self.module.ParticleSpecies(particle.name, particleType,
                                    particle.inEquilibrium, 
                                    particle.msqVacuum(fields) / T**2.0,
                                    particle.msqThermal(T) / T**2.0,
                                    particle.ultrarelativistic)

    def calculateCollisionIntegrals(self, bVerbose = False):
        ## "N". Make sure this is >= 0. The C++ code requires uint so pybind11 will throw TypeError otherwise
        basisSize = WallGo.config.getint("PolynomialGrid", "momentumGridSize")
        self.manager.calculateCollisionIntegrals(basisSize, bVerbose = False)