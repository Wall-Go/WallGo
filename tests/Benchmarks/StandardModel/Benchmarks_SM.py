import WallGo.fields

from tests.BenchmarkPoint import BenchmarkPoint

BM1 = BenchmarkPoint(
    inputParams={
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 0.0,
    },
    phaseInfo={
        "Tn":57.1958,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([55.]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRangeHighTPhase": (
            46,
            58.,
            0.01,
        ),  ##Set by hand such that the lowest temperature is (slightly) below the actual minimum temperature
        "interpolateTemperatureRangeLowTPhase": (
            56.,
            58.,
            0.01,
        ),  ##Set by hand such that the highest temperature is (slightly) above the actual maximum temperature
        "derivativeSettings": {
            "temperatureVariationScale": 0.75,
            "fieldValueVariationScale": 50.0,
        },  ## Optimized settings for Standard Model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 57.4104,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([55.806]),
        "minimumTemperaturePhase1": 51.8199,
        "maximumTemperaturePhase2": 57.583,
    },
)

BM2 = BenchmarkPoint(
    inputParams={
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 34.0,
    },
    phaseInfo={
        "Tn": 70.5793,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([64.6294]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRangeHighTPhase": (
            60.,
            75.,
            0.01,
        ),  ##Set by hand such that the lowest temperature is (slightly) below the actual minimum temperature
        "interpolateTemperatureRangeLowTPhase": (
            66.,
            72.5,
            0.01,
        ),  ##Set by hand such that the highest temperature is (slightly) above the actual maximum temperature
        "derivativeSettings": {
            "temperatureVariationScale": 0.75,
            "fieldValueVariationScale": 50.0,
        },  ## Optimized settings for Standard Model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 70.8238,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([64.9522]),
        "minimumTemperaturePhase1": 63.8464,
        "maximumTemperaturePhase2": 71.025,
    },
)

BM3 = BenchmarkPoint(
    inputParams={
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 50.0,
    },
    phaseInfo={
        "Tn": 83.4251,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([67.2538]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRangeHighTPhase": (
            70.,
            90.,
            0.01,
        ),  ##Set by hand such that the lowest temperature is (slightly) below the actual minimum temperature
        "interpolateTemperatureRangeLowTPhase": (
            75.,
            85.,
            0.01,
        ),  ##Set by hand such that the highest temperature is (slightly) above the actual maximum temperature
        "derivativeSettings": {
            "temperatureVariationScale": 0.75,
            "fieldValueVariationScale": 50.0,
        },  ## Optimized settings for Standard Model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 83.668,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([67.2538]),
        "minimumTemperaturePhase1": 75.291,
        "maximumTemperaturePhase2": 83.879,
    },
)

BM4 = BenchmarkPoint(
    inputParams={
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 70.0,
    },
    phaseInfo={
        "Tn": 102.344,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([65.8969]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRangeHighTPhase": (
            95.,
            110.,
            0.01,
        ),  ##Set by hand such that the lowest temperature is (slightly) below the actual minimum temperature
        "interpolateTemperatureRangeLowTPhase": (
            90.,
            105.,
            0.01,
        ),  ##Set by hand such that the highest temperature is (slightly) above the actual maximum temperature
        "derivativeSettings": {
            "temperatureVariationScale": 0.75,
            "fieldValueVariationScale": 50.0,
        },  ## Optimized settings for Standard Model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 102.57,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([65.8969]),
        "minimumTemperaturePhase1": 91.9013,
        "maximumTemperaturePhase2": 102.844,
    },
)

BM5 = BenchmarkPoint(
    inputParams={
        "v0": 246.0,
        "mW": 80.4,
        "mZ": 91.2,
        "mt": 174.0,
        "g3": 1.2279920495357861,
        "mH": 81.0,
    },
    phaseInfo={
        "Tn": 113.575,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([64.8777]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRangeHighTPhase": (
            105.,
            118.,
            0.01,
        ),  ##Set by hand such that the lowest temperature is (slightly) below the actual minimum temperature
        "interpolateTemperatureRangeLowTPhase": (
            105.,
            116.,
            0.01,
        ),  ##Set by hand such that the highest temperature is (slightly) above the actual maximum temperature
        "derivativeSettings": {
            "temperatureVariationScale": 0.75,
            "fieldValueVariationScale": 50.0,
        },  ## Optimized settings for Standard Model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 113.795,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0]),
        "phaseLocation2": WallGo.Fields([64.8777]),
        "minimumTemperaturePhase1": 101.602,
        "maximumTemperaturePhase2": 114.105,
    },
)


##
standardModelBenchmarks = [BM1, BM2, BM3, BM4, BM5]
