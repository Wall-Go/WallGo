## SingletSM_Z2/conftest.py -- Configure singlet model specific tests. These are specifically for the benchmark model that Benoit provided
import pytest
from typing import Tuple

import WallGo

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel

from .Benchmarks_singlet import BM1

from Models.SingletStandardModel_Z2.singletStandardModelZ2 import (
    SingletSMZ2,
)  # Benoit benchmark model
from Models.SingletStandardModel_Z2.SingletStandardModel_Z2_Simple import (
    SingletSM_Z2_Simple,
)  # just O(g^2T^4) bits


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
    model = SingletSMZ2()
    model.updateModel(inputs)

    yield BenchmarkModel(model, singletBenchmarkPoint)


@pytest.fixture(scope="session")
def singletSimpleBenchmarkModel(
    singletBenchmarkPoint: BenchmarkPoint,
) -> BenchmarkModel:
    inputs = singletBenchmarkPoint.inputParams
    model = SingletSM_Z2_Simple(inputs)

    yield BenchmarkModel(model, singletBenchmarkPoint)


"""----- Fixtures for more complicated things that depend on the model/Veff. 
I'm making these return also the original benchmark point so that it's easier to validate results, 
eg. read from BenchmarkPoint.expectedResults"""


## This constructs thermodynamics without interpolating anything
@pytest.fixture(scope="session")
def singletBenchmarkThermo(
    singletBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    BM = singletBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    # I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though,
    # so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(
        singletBenchmarkModel.model.getEffectivePotential(),
        Tn,
        phase2,
        phase1,
    )

    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    thermo.freeEnergyHigh.minPossibleTemperature = 50.0
    thermo.freeEnergyHigh.maxPossibleTemperature = 200.0
    thermo.freeEnergyLow.minPossibleTemperature = 50.0
    thermo.freeEnergyLow.maxPossibleTemperature = 200.0

    thermo.setExtrapolate()

    yield thermo, BM


## This is like the singletBenchmarkThermo fixture but interpolates the FreeEnergy objects over the temperature range specified in our BM input
@pytest.fixture(scope="session")
def singletBenchmarkThermo_interpolate(
    singletBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    BM = singletBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    ## I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though,
    ## so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(
        singletBenchmarkModel.model.getEffectivePotential(), Tn, phase2, phase1
    )

    ## Let's turn these off so that things are more transparent
    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    """ Then manually interpolate """
    TMin, TMax, dT = BM.config["interpolateTemperatureRange"]

    thermo.freeEnergyHigh.tracePhase(TMin, TMax, dT)
    thermo.freeEnergyLow.tracePhase(TMin, TMax, dT)

    thermo.setExtrapolate()

    yield thermo, BM


## Test for derivatives of potential
@pytest.fixture(scope="session")
def singletSimpleBenchmarkEffectivePotential(
    singletSimpleBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.EffectivePotential, BenchmarkPoint]:

    # shorthand
    BM = singletSimpleBenchmarkModel.benchmarkPoint
    Veff = singletSimpleBenchmarkModel.model.getEffectivePotential()

    yield Veff, BM


## Test for following minimum
@pytest.fixture(scope="session")
def singletSimpleBenchmarkFreeEnergy(
    singletSimpleBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint]:

    BM = singletSimpleBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    # free energies for both phases
    freeEnergy1 = WallGo.FreeEnergy(
        singletSimpleBenchmarkModel.model.getEffectivePotential(), Tn, phase1
    )
    freeEnergy2 = WallGo.FreeEnergy(
        singletSimpleBenchmarkModel.model.getEffectivePotential(), Tn, phase2
    )

    # interpolation range
    TMin = 50.0
    TMax = 150.0
    dT = 0.1
    BM.config["interpolateTemperatureRange"] = TMin, TMax, dT

    freeEnergy1.tracePhase(TMin, TMax, dT, rTol=1e-6, paranoid=False)
    freeEnergy2.tracePhase(TMin, TMax, dT, rTol=1e-6, paranoid=False)

    yield freeEnergy1, freeEnergy2, BM


# Test for thermodynamics of simple model
@pytest.fixture(scope="session")
def singletSimpleBenchmarkThermodynamics(
    singletSimpleBenchmarkModel: BenchmarkModel,
) -> Tuple[WallGo.Thermodynamics, BenchmarkPoint]:

    BM = singletSimpleBenchmarkModel.benchmarkPoint

    Tn = BM.phaseInfo["Tn"]
    phase1 = BM.expectedResults["phaseLocation1"]
    phase2 = BM.expectedResults["phaseLocation2"]

    # interpolation range
    TMin = 50.0
    TMax = 150.0
    dT = 0.1
    BM.config["interpolateTemperatureRange"] = TMin, TMax, dT

    # I assume phase1 = high-T, phase2 = low-T. Would prefer to drop these labels though,
    # so WallGo could safely assume that the transition is always phase1 -> phase2
    thermo = WallGo.Thermodynamics(
        singletSimpleBenchmarkModel.model.getEffectivePotential(),
        Tn,
        phase2,
        phase1,
    )

    # Let's turn these off so that things are more transparent
    thermo.freeEnergyHigh.disableAdaptiveInterpolation()
    thermo.freeEnergyLow.disableAdaptiveInterpolation()

    thermo.freeEnergyHigh.tracePhase(TMin, TMax, dT)
    thermo.freeEnergyLow.tracePhase(TMin, TMax, dT)

    thermo.setExtrapolate()

    yield thermo, BM


## Hydro fixture, use the interpolated Thermo fixture because otherwise things get SLOOOW
@pytest.fixture(scope="session")
def singletBenchmarkHydrodynamics(
    singletBenchmarkThermo_interpolate: Tuple[WallGo.Thermodynamics, BenchmarkPoint]
) -> Tuple[WallGo.Hydrodynamics, BenchmarkPoint]:

    thermo, BM = singletBenchmarkThermo_interpolate

    yield WallGo.Hydrodynamics(thermo, 10.0, 0.01, 1e-6, 1e-6), BM


## This wouldn't need to be singlet-specific tbh. But it's here for now
@pytest.fixture(scope="session")
def singletBenchmarkGrid() -> Tuple[WallGo.Grid, WallGo.Polynomial]:

    M, N = 22, 11

    tailInside = 0.2
    tailOutside = 0.2
    wallThickness = 0.05
    momentumFalloffT = 100

    grid = WallGo.grid3Scales.Grid3Scales(
        M, N, tailInside, tailOutside, wallThickness, momentumFalloffT
    )

    return grid


@pytest.fixture(scope="session")
def singletBenchmarkCollisionArray(
    singletBenchmarkModel: BenchmarkModel, singletBenchmarkGrid: WallGo.Grid
) -> WallGo.CollisionArray:

    particles = singletBenchmarkModel.model.outOfEquilibriumParticles
    ## TODO better file path
    import pathlib

    fileDir = pathlib.Path(__file__).parent.resolve()
    collisionPath = fileDir / "../../TestData/N11/"

    return WallGo.CollisionArray.newFromDirectory(
        collisionPath, singletBenchmarkGrid, "Chebyshev", particles, bInterpolate=False
    )


@pytest.fixture(scope="session")
def singletBenchmarkBoltzmannSolver(
    singletBenchmarkModel: BenchmarkModel,
    singletBenchmarkGrid: WallGo.Grid,
    singletBenchmarkCollisionArray: WallGo.CollisionArray,
) -> WallGo.BoltzmannSolver:

    boltzmannSolver = WallGo.BoltzmannSolver(
        singletBenchmarkGrid, basisM="Cardinal", basisN="Chebyshev"
    )
    boltzmannSolver.updateParticleList(
        singletBenchmarkModel.model.outOfEquilibriumParticles
    )
    boltzmannSolver.setCollisionArray(singletBenchmarkCollisionArray)
    return boltzmannSolver


## EOM object for the singlet model, no out-of-equilibrium contributions.
@pytest.fixture(scope="session")
def singletBenchmarkEOM_equilibrium(
    singletBenchmarkBoltzmannSolver,
    singletBenchmarkThermo_interpolate,
    singletBenchmarkHydrodynamics,
    singletBenchmarkGrid: WallGo.Grid,
) -> Tuple[WallGo.EOM, BenchmarkPoint]:

    thermo, BM = singletBenchmarkThermo_interpolate
    hydrodynamics, _ = singletBenchmarkHydrodynamics
    grid = singletBenchmarkGrid
    boltzmannSolver = singletBenchmarkBoltzmannSolver
    meanFreePathScale = 0

    fieldCount = 2

    ## TODO fix error tolerance?
    eom = WallGo.EOM(
        boltzmannSolver,
        thermo,
        hydrodynamics,
        grid,
        fieldCount,
        meanFreePathScale,
        (0.1, 100.0),
        (-10.0, 10.0),
        includeOffEq=False,
    )

    return eom, BM
