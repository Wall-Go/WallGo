"""
CollisionModuleLoader.py : Helpers for loading the Collision module. 
This lazy-loads the module into global CollisionModule variable.
"""

import importlib

CollisionModule = None
collisionModuleLoaded = False

def loadCollisionModule():

    ## keeping these imports scoped on purpose
    import os, sys

    global CollisionModule
    global collisionModuleLoaded
    ## Don't attempt to import if already loaded
    if CollisionModule is None:
        try: 

            currentDirectory = os.path.dirname(__file__)

            ## path relative to the above
            relativePathToModule = "../../Collision/pybind/lib"

            collisionModulePath = os.path.join(currentDirectory, relativePathToModule)

            sys.path.append(collisionModulePath)

            moduleName = "WallGoCollisionPy"
            
            CollisionModule = importlib.import_module(moduleName)
            print("Loaded module [%s]" % moduleName)
            collisionModuleLoaded = True

        except ImportError:
            print("Warning: Failed to load [%s]. Using read-only mode for collision integrals." % moduleName)
            print("Computation of new collision integrals will NOT be possible.")
