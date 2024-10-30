# Running Models 

## Overview

This folder contains individual model directories, each equipped with a Python script to calculate the wall velocity specific to that model. Additional files within each model directory support the computation of matrix elements and collision integrations.

## Directory Structure

Each model directory includes the following files and subdirectories:
- `<nameOfModel>.py` - Contains the model definition and the commands for finding the wall velocity.
- `<nameOfModel>Config.ini` - An optional configuration file containing model-specific configuration settings. (Note: This file is not included for the Yukawa model.)
- `exampleCollisionDefs.py` - Defines collision generation settings specific to the model.
- `MatrixElements/` - A subdirectory holding scripts (with .m extensions) used for generating matrix elements, as well as a .json file containing the computed matrix elements.
- `CollisionOutput_N<spatialGridSize>/` - Subdirectories for storing collision data files for each pair of out-of-equilibrium particles, with each folder corresponding to a specific spatial grid size. 

Additionally, the Models folder contains a common utility file:

- `wallGoExampleBase.py` - This script provides a shared template for wall velocity computations. It is used by all models except the Yukawa model.

## Running a model

The examples can be run directly with e.g.

    python3 Models/SingletStandardModel_Z2/singletStandardModelZ2.py
    
For models that use the common `wallGoExampleBase.py', additional command line arguments are available to (re)calculate matrix elements and collisions, and adjust the momentum grid size e.g.

    python3 Models/SingletStandardModel_Z2/singletStandardModelZ2.py --recalculateMatrixElements --recalculateCollisions --momentumGridSize 5

## Requirements

Running the model files requires a valid installation of [**WallGo**](https://github.com/Wall-Go/WallGo) . Generation of new matrix elements and collisions furthermore requires installations of [**WallGoMatrix**](https://github.com/Wall-Go/WallGoMatrix) and  [**WallGoCollision**](https://github.com/Wall-Go/WallGoCollision) respectively.
