## StandardModel/conftest.py -- Configure singlet model specific tests.
import pytest
from typing import Tuple

import WallGo

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel

from .Benchmarks_SM import standardModelBenchmarks

from Models.StandardModel.standardModel import (
    StandardModel,
)


""" 
FreeEnergy interpolations are initialized in the standardModelBenchmarkThermo_interpolate() fixture below.
Interpolations have a huge impact on performance but also affect the results somewhat.
Bottom line is that these tests ARE sensitive to details of the interpolations!
"""


"""----- Define fixtures for the standard model.

NOTE: I'm giving these session scope so that their state is preserved between tests (cleared when pytest finishes).
This is helpful as things like FreeEnergy interpolations are slow, however it does make our tests a bit less transparent.
"""


@pytest.fixture(scope="session", params=standardModelBenchmarks, ids=["BM1", "BM2", "BM3", "BM4", "BM5"])
def standardModelBenchmarkPoint(request: pytest.FixtureRequest) -> BenchmarkPoint:
    yield request.param


@pytest.fixture(scope="session")
def standardModelBenchmarkModel(standardModelBenchmarkPoint: BenchmarkPoint) -> BenchmarkModel:
    inputs = standardModelBenchmarkPoint.inputParams
    model = StandardModel()
    model.updateModel(inputs)

    yield BenchmarkModel(model, standardModelBenchmarkPoint)



"""----- Fixtures for more complicated things that depend on the model/Veff. 
I'm making these return also the original benchmark point so that it's easier to validate results, 
eg. read from BenchmarkPoint.expectedResults"""


## This constructs thermodynamics without interpolating anything
@pytest.fixture(scope="session")
def standardModelBenchmarkThermo(
    standardModelBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    BM = standardModelBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    # I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though,
    # so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(
        standardModelBenchmarkModel.model.getEffectivePotential(),
        Tn,
        phase2,
        phase1,
    )

    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    thermo.freeEnergyHigh.minPossibleTemperature = 40.0
    thermo.freeEnergyHigh.maxPossibleTemperature = 60.0
    thermo.freeEnergyLow.minPossibleTemperature = 40.0
    thermo.freeEnergyLow.maxPossibleTemperature = 60.0

    thermo.setExtrapolate()

    yield thermo, BM


## This is like the standardModelBenchmarkThermo fixture but interpolates the FreeEnergy objects over the temperature range specified in our BM input
@pytest.fixture(scope="session")
def standardModelBenchmarkThermo_interpolate(
    standardModelBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    BM = standardModelBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though,
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(
        standardModelBenchmarkModel.model.getEffectivePotential(), Tn, phase2, phase1
    )

    ## Let's turn these off so that things are more transparent
    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    """ Then manually interpolate """
    TMin, TMax, dT = BM.config["interpolateTemperatureRangeHighTPhase"]

    thermo.freeEnergyHigh.tracePhase(TMin, TMax, dT)

    TMin, TMax, dT = BM.config["interpolateTemperatureRangeLowTPhase"]
    thermo.freeEnergyLow.tracePhase(TMin, TMax, dT)

    thermo.setExtrapolate()

    yield thermo, BM



## Test for following minimum
@pytest.fixture(scope="session")
def standardModelBenchmarkFreeEnergy(
    standardModelBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint]:

    BM = standardModelBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    # free energies for both phases
    freeEnergy1 = WallGo.FreeEnergy(
        standardModelBenchmarkModel.model.getEffectivePotential(), Tn, phase1
    )
    freeEnergy2 = WallGo.FreeEnergy(
        standardModelBenchmarkModel.model.getEffectivePotential(), Tn, phase2
    )


    """ Then manually interpolate """
    TMin, TMax, dT = BM.config["interpolateTemperatureRangeHighTPhase"]
    freeEnergy1.tracePhase(TMin, TMax, dT, rTol=1e-6, paranoid=False)

    TMin, TMax, dT = BM.config["interpolateTemperatureRangeLowTPhase"]
    freeEnergy2.tracePhase(TMin, TMax, dT, rTol=1e-6, paranoid=False)

    yield freeEnergy1, freeEnergy2, BM