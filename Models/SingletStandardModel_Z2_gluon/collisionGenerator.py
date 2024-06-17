"""Simple example file for using the WallGo collision module from Python.
The module is written in C++ but implements Python bindings. There is also a pure Python wrapper class,
WallGo.Collision, which handles dynamic loading of the module and provides complementary functionality.
Note that WallGo.Collision is a singleton class, ie. only one instance of it can exist.
Loading of the module happens when the instance is first created."""

import os
import pathlib

import WallGo
from WallGo import Particle
from WallGo import Fields

from SingletStandardModel_Z2_gluon import (
    SingletSM_Z2,
)

WallGo.initialize()


## Modify the config, we use N=5 for this example
WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "5")

## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
inputParameters = {
    #"RGScale" : 91.1876,
    "RGScale" : 125., # <- Benoit benchmark
    "v0" : 246.0,
    "MW" : 80.379,
    "MZ" : 91.1876,
    "Mt" : 173.0,
    "g3" : 1.2279920495357861,
    # scalar specific, choose Benoit benchmark values
    "mh1" : 125.0,
    "mh2" : 120.0,
    "a2" : 0.9,
    "b4" : 1.0
}

model = SingletSM_Z2(inputParameters)

## Create Collision singleton which automatically loads the collision module
collision = WallGo.Collision(model)

## Optional: set the seed used by Monte Carlo integration. Default is 0
collision.setSeed(0)

"""
Define couplings (Lagrangian parameters)
list as they appear in the MatrixElements file
"""
collision.manager.addCoupling(inputParameters["g3"])

## Set input/output paths
scriptLocation = pathlib.Path(__file__).parent.resolve()

collision.manager.setOutputDirectory(str(scriptLocation / "CollisionOutput/"))
collision.manager.setMatrixElementFile(str(scriptLocation / "MatrixElements.txt"))

## Configure integration. Can skip this step if you're happy with the defaults
integrationOptions = collision.module.IntegrationOptions()
integrationOptions.bVerbose = True
integrationOptions.maxTries = 50
integrationOptions.calls = 50000
integrationOptions.relativeErrorGoal = 1e-1
integrationOptions.absoluteErrorGoal = 1e-8

collision.manager.configureIntegration(integrationOptions)

## Instruct the collision manager to print out symbolic matrix elements as it parses them. Can be useful for debugging
collision.manager.setMatrixElementVerbosity(True)

## "N". Make sure this is >= 0. The C++ code requires uint so pybind11 will throw TypeError otherwise
# basisSize = 5
basisSize = WallGo.config.getint("PolynomialGrid", "momentumGridSize")
# print(WallGo.config)

## Computes collisions for all out-of-eq particles specified above. The last argument is optional and mainly useful for debugging
collision.manager.calculateCollisionIntegrals(basisSize, bVerbose = False)