## SingletSM_Z2/conftest.py -- Configure singlet model specific tests. These are specifically for the benchmark model that Benoit provided

import numpy as np
import pytest
import math
from typing import Tuple

import WallGo

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel

from .Benchmarks_singlet import BM1

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model


""" NOTE: We run all singlet-specific tests using interpolated Jb/Jf integrals and interpolated FreeEnergy objects. 
The former are automatically loaded with the SingletSM_Z2 model and are DIFFERENT from the default WallGo interpolations
-- this is due to difference in the interpolations used for the original benchmark tests: 
The range is different, and extrapolations are allowed here.

FreeEnergy interpolations are initialized in the singletBenchmarkThermo_interpolate() fixture below.
Interpolations have a huge impact on performance but also affect the results somewhat.
Bottom line is that these tests ARE sensitive to details of the interpolations!
"""


"""----- Define fixtures for the singlet model.
Would be good to make all our singlet-specific tests to use these for easier control.

NOTE: I'm giving these session scope so that their state is preserved between tests (cleared when pytest finishes).
This is helpful as things like FreeEnergy interpolations are slow, however it does make our tests a bit less transparent.
"""

@pytest.fixture(scope="session")
def singletBenchmarkPoint() -> BenchmarkPoint:
    yield BM1


@pytest.fixture(scope="session")
def singletBenchmarkModel(singletBenchmarkPoint: BenchmarkPoint) -> BenchmarkModel:
    inputs = singletBenchmarkPoint.inputParams
    model = SingletSM_Z2(inputs)

    yield BenchmarkModel(model, singletBenchmarkPoint)



"""----- Fixtures for more complicated things that depend on the model/Veff. 
I'm making these return also the original benchmark point so that it's easier to validate results, 
eg. read from BenchmarkPoint.expectedResults"""
 
## This constructs thermodynamics without interpolating anything
@pytest.fixture(scope="session")
def singletBenchmarkThermo(singletBenchmarkModel: BenchmarkModel) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    ## annoyingly Thermo needs Tc in the constructor, even though the class doesn't really use it

    BM = singletBenchmarkModel.benchmarkPoint

    Tc = BM.expectedResults["Tc"]
    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()


    yield thermo, BM



## This is like the singletBenchmarkThermo fixture but interpolates the FreeEnergy objects over the temperature range specified in our BM input 
@pytest.fixture(scope="session")
def singletBenchmarkThermo_interpolate(singletBenchmarkModel: BenchmarkModel) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:
    
    BM = singletBenchmarkModel.benchmarkPoint

    Tc = BM.expectedResults["Tc"]
    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

    ## Let's turn these off so that things are more transparent
    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    """ Then manually interpolate """
    TMin, TMax, dT = BM.config["interpolateTemperatureRange"]
    interpolationPointCount = math.ceil((TMax - TMin) / dT)

    thermo.freeEnergyHigh.newInterpolationTable(TMin, TMax, interpolationPointCount)
    thermo.freeEnergyLow.newInterpolationTable(TMin, TMax, interpolationPointCount)
    
    yield thermo, BM


## Hydro fixture, use the interpolated Thermo fixture because otherwise things get SLOOOW
@pytest.fixture(scope="session")
def singletBenchmarkHydro(singletBenchmarkThermo_interpolate: Tuple[WallGo.Thermodynamics, BenchmarkPoint]) -> Tuple[WallGo.Hydro, BenchmarkPoint]:
    
    thermo, BM = singletBenchmarkThermo_interpolate
    
    ## temperature range guesses
    TMinGuess = BM.config["hydroTMinGuess"]
    TMaxGuess = BM.config["hydroTMaxGuess"]

    ## TODO Should fix rtol, atol here so that our tests don't magically change if the class defaults change !
    yield WallGo.Hydro(thermo, TminGuess=TMinGuess, TmaxGuess=TMaxGuess), BM


## This wouldn't need to be singlet-specific tbh. But it's here for now. And it really needs to get rid of the temperature argument
@pytest.fixture(scope="session")
def singletBenchmarkGrid() -> Tuple[WallGo.Grid, WallGo.Polynomial]:

    M, N = 20, 20
    
    # magic 0.05
    grid = WallGo.Grid(M, N, 0.05, 100)

    return grid, WallGo.Polynomial(grid)



"""Test particle. This is also defined in our main conftest.py, but maybe better to have a separate one dedicated for our singlet benchmark.
TODO fix masses
"""
@pytest.fixture(scope="session")
def singletBenchmarkParticle():
    return WallGo.Particle(
        name="top",
        msqVacuum=lambda phi: 0.5 * phi**2,
        msqThermal=lambda T: 0.1 * T**2,
        statistics="Fermion",
        inEquilibrium=False,
        ultrarelativistic=False,
        multiplicity=1,
    )



"""EOM object for the singlet model, no out-of-equilibrium contributions.
This still needs a particle input though (intended??) so I'm using the particle fixture defined in our main conftest.py
"""
@pytest.fixture(scope="session")
def singletBenchmarkEOM_equilibrium(singletBenchmarkParticle, singletBenchmarkThermo_interpolate, singletBenchmarkHydro, singletBenchmarkGrid) -> Tuple[WallGo.EOM, BenchmarkPoint]:
    
    thermo, BM = singletBenchmarkThermo_interpolate
    hydro, _ = singletBenchmarkHydro
    grid, _ = singletBenchmarkGrid

    fieldCount = 2

    ## TODO fix error tolerance?
    eom = WallGo.EOM(singletBenchmarkParticle, thermo, hydro, grid, fieldCount, includeOffEq=False)

    return eom, BM


@pytest.fixture(scope="session")
def singletBenchmarkBoltzmannSolver(singletBenchmarkModel: BenchmarkModel, grid: WallGo.Grid):

    singletBenchmarkModel

    boltzmannSolver = WallGo.BoltzmannSolver(grid, basisM = "Cardinal", basisN = "Chebyshev")
    boltzmannSolver.updateParticleList( singletBenchmarkModel.model.outOfEquilibriumParticles )