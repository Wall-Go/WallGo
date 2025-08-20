"""
This Python script, exampleOutputThermodynamics.py,
uses the implementation of the minimal Standard Model
extension in singletStandardModelZ2.py and gives methods
for saving the thermodynamics of the model for later use
in e.g. PTTools.
"""

import sys
import pathlib
import numpy as np
import h5py

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()

# Add the SingletStandardModelZ2 folder to the path to import SingletSMZ2
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from singletStandardModelZ2 import SingletSMZ2


def main() -> None:

    manager = WallGo.WallGoManager()

    # Model definition is done in the SingletSMZ2 class
    model = SingletSMZ2()
    manager.registerModel(model)

    inputParameters = {
        "RGScale": 125.0,
        "v0": 246.0,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        "mh1": 125.0,
        "mh2": 120.0,
        "lHS": 0.9,
        "lSS": 1.0,
    }

    model.updateModel(inputParameters)

    # Creates a Thermodynamics object, containing all thermodynamic functions
    # by tracing the high- and low-temperature phases.
    # For temperatures outside of the range of existence of the phases, the
    # thermodynamic functions are extrapolated using the template model.
    manager.setupThermodynamicsHydrodynamics(
        WallGo.PhaseInfo(
            temperature=100.0,  # nucleation temperature
            phaseLocation1=WallGo.Fields([0.0, 200.0]),
            phaseLocation2=WallGo.Fields([246.0, 0.0]),
        ),
        WallGo.VeffDerivativeSettings(
            temperatureVariationScale=10.0, fieldValueVariationScale=[10.0, 10.0]
        ),
    )

    Tcrit = manager.thermodynamics.findCriticalTemperature(dT = 0.1)

    # Generate tables for the thermodynamics functions of both phases.
    # Note that if the minimum or maximum temperature chosen is outside of
    # the range of existence of the phases, the extrapolation is used.

    # Temperature range and step for high-temperature phase
    highTPhaseRange = (80.0, 140.0, 0.1)
    # Temperature range and step for low-temperature phase
    lowTPhaseRange = (80.0, 110.0, 0.1)

    # Create temperature arrays
    temp_high = np.arange(highTPhaseRange[0], highTPhaseRange[1], highTPhaseRange[2])
    temp_low = np.arange(lowTPhaseRange[0], lowTPhaseRange[1], lowTPhaseRange[2])

    # Evaluate thermodynamic functions for high-temperature phase
    p_high = np.array([manager.thermodynamics.pHighT(T) for T in temp_high])
    e_high = np.array([manager.thermodynamics.eHighT(T) for T in temp_high])
    csq_high = np.array([manager.thermodynamics.csqHighT(T) for T in temp_high])

    # Evaluate thermodynamic functions for low-temperature phase
    p_low = np.array([manager.thermodynamics.pLowT(T) for T in temp_low])
    e_low = np.array([manager.thermodynamics.eLowT(T) for T in temp_low])
    csq_low = np.array([manager.thermodynamics.csqLowT(T) for T in temp_low])

    # Get temperature limits; outside of these limits, extrapolation has been used
    max_temp_high = manager.thermodynamics.freeEnergyHigh.maxPossibleTemperature[0]
    min_temp_high = manager.thermodynamics.freeEnergyHigh.minPossibleTemperature[0]
    max_temp_low = manager.thermodynamics.freeEnergyLow.maxPossibleTemperature[0]
    min_temp_low = manager.thermodynamics.freeEnergyLow.minPossibleTemperature[0]

    # Create HDF5 file
    filename = pathlib.Path(__file__).resolve().parent/"thermodynamics_data.h5"

    with h5py.File(filename, 'w') as f:
        # Global attributes
        f.attrs["model_label"] = "singletSMZ2"
        f.attrs["critical_temperature"] = Tcrit
        f.attrs["nucleation_temperature"] = 100.
        
        # High-temperature phase group
        high_group = f.create_group("high_temperature_phase")
        high_group.create_dataset("temperature", data=temp_high)
        high_group.create_dataset("pressure", data=p_high)
        high_group.create_dataset("energy_density", data=e_high)
        high_group.create_dataset("sound_speed_squared", data=csq_high)
        high_group.attrs["max_possible_temperature"] = max_temp_high
        high_group.attrs["min_possible_temperature"] = min_temp_high

        # Low-temperature phase group
        low_group = f.create_group("low_temperature_phase")
        low_group.create_dataset("temperature", data=temp_low)
        low_group.create_dataset("pressure", data=p_low)
        low_group.create_dataset("energy_density", data=e_low)
        low_group.create_dataset("sound_speed_squared", data=csq_low)
        low_group.attrs["max_possible_temperature"] = max_temp_low
        low_group.attrs["min_possible_temperature"] = min_temp_low

    print(f"Thermodynamics data saved to {filename}")
    print("Model: singletSMZ2")
    print(f"Critical temperature: {Tcrit:.2f} GeV")
    print(f"High-T phase: {len(temp_high)} temperature points from {temp_high[0]:.1f} to {temp_high[-1]:.1f} GeV")
    print(f"Low-T phase: {len(temp_low)} temperature points from {temp_low[0]:.1f} to {temp_low[-1]:.1f} GeV")
    print(f"High-T phase valid range: {min_temp_high:.1f} to {max_temp_high:.1f} GeV")
    print(f"Low-T phase valid range: {min_temp_low:.1f} to {max_temp_low:.1f} GeV")


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
