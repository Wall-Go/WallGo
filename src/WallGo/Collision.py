## This just loads the C++ collision module

import os, sys
import importlib

def loadCollisionModule():

    collisionModule = None
    try: 
        currentDirectory = os.path.dirname(__file__)

        ## path relative to the above
        relativePathToModule = "../../Collision/pybind/lib"

        collisionModulePath = os.path.join(currentDirectory, relativePathToModule)

        sys.path.append(collisionModulePath)

        moduleName = "WallGoCollisionPy"
        
        collisionModule = importlib.import_module(moduleName)
        print("Loaded module [%s]" % moduleName)

    except ImportError as error:
        print("Warning: Failed to load [%s]. Using read-only mode for collision integrals." % moduleName)
        print("Computation of new collision integrals will NOT be possible.")
        print(error)

    return collisionModule
