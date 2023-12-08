import numpy as np
import pytest
import math
from typing import Tuple

import WallSpeed

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel

from .Benchmarks_singlet import BM1

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model



"""----- Define some fixtures. Currently these are used for hydro boundaries test only.
Would be good to make all our singlet-specific tests to use these for easier control.
"""

@pytest.fixture
def singletBenchmarkPoint() -> BenchmarkPoint:
    yield BM1


@pytest.fixture
def singletBenchmarkModel(singletBenchmarkPoint: BenchmarkPoint) -> BenchmarkModel:
    inputs = singletBenchmarkPoint.inputParams
    model = SingletSM_Z2(inputs)

    yield BenchmarkModel(model, singletBenchmarkPoint)



"""----- Fixtures for more complicated things that depend on the model/Veff. 
I'm making these return also the original benchmark point so that it's easier to validate results, 
eg. read from BenchmarkPoint.expectedResults"""
 
## This constructs thermodynamics without interpolating anything
@pytest.fixture
def singletBenchmarkThermo(singletBenchmarkModel: BenchmarkModel) -> Tuple[WallSpeed.Thermodynamics, BenchmarkPoint]:

    ## annoyingly Thermo needs Tc in the constructor, even though the class doesn't really use it

    BM = singletBenchmarkModel.benchmarkPoint

    Tc = BM.expectedResults["Tc"]
    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallSpeed.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()


    yield thermo, BM



## This is like the singletBenchmarkThermo fixture but interpolates the FreeEnergy objects over the temperature range specified in our BM input 
@pytest.fixture
def singletBenchmarkThermo_interpolate(singletBenchmarkModel: BenchmarkModel) -> Tuple[WallSpeed.Thermodynamics, BenchmarkPoint]:
    
    BM = singletBenchmarkModel.benchmarkPoint

    Tc = BM.expectedResults["Tc"]
    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallSpeed.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

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
@pytest.fixture
def singletBenchmarkHydro(singletBenchmarkThermo_interpolate: Tuple[WallSpeed.Thermodynamics, BenchmarkPoint]) -> Tuple[WallSpeed.Hydro, BenchmarkPoint]:
    
    thermo, BM = singletBenchmarkThermo_interpolate

    ## TODO Should fix rtol, atol here so that our tests don't magically change if the class defaults change !
    yield WallSpeed.Hydro(thermo), BM


