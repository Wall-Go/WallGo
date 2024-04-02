
import os, sys
import importlib
from types import ModuleType
from .Particle import Particle

class Collision:
    """Thin wrapper around the C++ module. This handles loading of the module, provides Python-readable type hints etc.
    This class is a singleton.
    """
    _instance = None
    
    def __new__(cls):
        # Implement singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    
    def __init__(self):
        if not hasattr(self, "bInitialized"):
            self.module: ModuleType = None
            self._loadCollisionModule()
            self.bInitialized = True

    
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