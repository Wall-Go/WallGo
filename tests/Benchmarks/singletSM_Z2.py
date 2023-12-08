import numpy as np
import pytest

import WallSpeed

from .BenchmarkPoint import BenchmarkPoint

from Models.SingletStandardModel_Z2.SingletStandardModel_Z2 import SingletSM_Z2 # Benoit benchmark model



class BenchmarkModel:
    """This just holds a model instance + BenchmarkPoint.
    """

    model: WallSpeed.GenericModel
    benchmarkPoint: BenchmarkPoint

    def __init__(self, model: WallSpeed.GenericModel, benchmarkPoint: BenchmarkPoint):

        self.model = model
        self.benchmarkPoint = BenchmarkPoint



## BM1 = _the_ Benoit benchmark point 
BM1 = BenchmarkPoint( 
    inputParams =
    {
    "RGScale" : 125., # <- Benoit's value
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
    },

    phaseInfo = 
    {
        "Tn" : 100.,
        ## Guesses for phase locations
        "phaseLocation1" : [ 0.0, 200.0 ],
        "phaseLocation2" : [ 246.0, 0.0]
    },

    config = {
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRange" : tuple([0.0, 1.2*108.22, 1.0]) ## upper bound is 1.2 * Tc
    },

    expectedResults = 
    {
        "Tc" : 108.22,
        ## Phase locations at nucleation temperature
        "phaseLocation1" : [0.0, 104.85563975],
        "phaseLocation2" : [195.01844099, 0.0]
    } 
)


BM2 = BenchmarkPoint(
    {
        "RGScale" : 125.,
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        "mh1" : 125.0,
        "mh2" : 160.0,
        "a2" : 1.2,
        "b4" : 1.0
    },
    {}
)


BM3 = BenchmarkPoint(
    {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 160.0,
        "a2" : 1.6,
        "b4" : 1.0
    },
    {}
)

BM4 = BenchmarkPoint(
    {
        "RGScale" : 125.,
        "v0" : 246.22,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        "mh1" : 125.0,
        "mh2" : 160.0,
        "a2" : 1.2,
        "b4" : 1.0
    },
    {}
)


singletBenchmarks = [ BM1, BM2, BM3, BM4 ]


""" Define some fixtures. Currently these are used for hydro boundaries test only
"""

@pytest.fixture
def singletBenchmarkPoint() -> BenchmarkPoint:
    yield BM1


@pytest.fixture
def singletBenchmarkModel(singletBenchmarkPoint: BenchmarkPoint) -> BenchmarkModel:
    inputs = singletBenchmarkPoint.inputParams
    model = SingletSM_Z2(inputs)

    yield BenchmarkModel(model, singletBenchmarkPoint)

 
## This constructs thermodynamics without interpolating anything
@pytest.fixture
def singletBenchmarkThermo(singletBenchmarkModel: BenchmarkModel) -> WallSpeed.Thermodynamics:

    ## annoyingly Thermo needs Tc in the constructor, even though the class doesn't really use it

    Tc = singletBenchmarkModel.benchmarkPoint.expectedResults["Tc"]
    Tn = singletBenchmarkModel.benchmarkPoint.phaseInfo["Tn"]
    phase1 = singletBenchmarkModel.benchmarkPoint.expectedResults["phaseLocation1"]
    phase2 = singletBenchmarkModel.benchmarkPoint.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallSpeed.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()


    yield thermo


## This is like the singletBenchmarkThermo fixture but interpolates the FreeEnergy objects over the temperature range specified in our BM input 
@pytest.fixture
def singletBenchmarkThermo_interpolate(singletBenchmarkModel: BenchmarkModel) -> WallSpeed.Thermodynamics:
    
    Tc = singletBenchmarkModel.benchmarkPoint.expectedResults["Tc"]
    Tn = singletBenchmarkModel.benchmarkPoint.phaseInfo["Tn"]
    phase1 = singletBenchmarkModel.benchmarkPoint.expectedResults["phaseLocation1"]
    phase2 = singletBenchmarkModel.benchmarkPoint.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though, 
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallSpeed.Thermodynamics(singletBenchmarkModel.model.effectivePotential, Tc, Tn, phase2, phase1)

    ## Let's turn these off so that things are more transparent
    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    """ Then manually interpolate """
    TMin, TMax, dT = singletBenchmarkModel.benchmarkPoint.config["interpolateTemperatureRange"]
    interpolationPointCount = int((TMax - TMin) / dT)

    thermo.freeEnergyHigh.newInterpolationTable(TMin, TMax, interpolationPointCount)
    thermo.freeEnergyLow.newInterpolationTable(TMin, TMax, interpolationPointCount)

    yield thermo


## Hydro fixture, use the interpolated Thermo fixture because otherwise things get SLOOOW
@pytest.fixture
def singletBenchmarkHydro(singletBenchmarkThermo_interpolate: WallSpeed.Thermodynamics) -> WallSpeed.Hydro:
    
    ## TODO Should fix rtol, atol here so that our tests don't magically change if the class defaults change !
    yield WallSpeed.Hydro(singletBenchmarkThermo_interpolate)


