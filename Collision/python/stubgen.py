## Helper script that runs pybind11-stubgen with modified PYTHONPATH. 
## This is necessary because 1) pybind11-stubgen requires that 'import PyCollision' works,
## and 2) we want to generate stubs before installing the module to some default location.
## Note that this should also prioritize modules in the specified search path (PYTHONPATH gets priority)
## so that we don't accidentally import some globally installed version of the module

import sys, os
import subprocess
from shutil import which

if len(sys.argv) != 3:
    print("Usage: stubgen.py <module_search_path> <module name (base)>")
    exit(1)

searchDir = sys.argv[1]
moduleName = sys.argv[2]

# Set PYTHONPATH for the subprocess
env = os.environ.copy()
env['PYTHONPATH'] = searchDir

stubgenCommand = "pybind11-stubgen"

if which(stubgenCommand) is None:
    print(f"Error: can't find program: {stubgenCommand}")
    sys.exit(1)

try:
    subprocess.run(['pybind11-stubgen', moduleName], check=True, env=env)
except subprocess.CalledProcessError as e:
    print(f"Error running pybind11-stubgen: {e}")
    sys.exit(2)