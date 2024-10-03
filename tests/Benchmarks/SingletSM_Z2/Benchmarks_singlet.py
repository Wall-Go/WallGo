import WallGo.fields

from tests.BenchmarkPoint import BenchmarkPoint

## BM1 = _the_ Benoit benchmark point
BM1 = BenchmarkPoint(
    inputParams={
        "RGScale": 125.0,  # <- Benoit's value
        "v0": 246.0,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1": 125.0,
        "mh2": 120.0,
        "a2": 0.9,
        "b4": 1.0,
    },
    phaseInfo={
        "Tn": 100.0,
        ## Guesses for phase locations
        "phaseLocation1": WallGo.Fields([0.0, 200.0]),
        "phaseLocation2": WallGo.Fields([246.0, 0.0]),
    },
    config={
        ## Give TMin, TMax, dT as a tuple
        "interpolateTemperatureRange": (
            79.85352365651234,
            122.82493098783233,
            0.1,
        ),  ##Set by hand to appropriate values, obtained from Template model
        "hydroTMinGuess": 79.85352365651234,  ##Set by hand to appropriate value, obtained from Template model
        "hydroTMaxGuess": 122.82493098783233,  ##Set by hand to appropriate value, obtained from Template model
    },
    ## Will probs need to adjust these once we decide on what our final implementation of the benchmark model is
    expectedResults={
        "Tc": 108.22,
        ## Phase locations at nucleation temperature
        "phaseLocation1": WallGo.Fields([0.0, 104.86827943]),
        "phaseLocation2": WallGo.Fields([195.03213609, 0.0]),
        ## Jouguet velocity
        "vJ": 0.644230065637426,
        ## LTE wall velocity
        "vwLTE": 0.6203233205259607,
        ## Hydro boundaries stuff
        "c1": -3331587978,
        "c2": 2976953742,
        "Tplus": 103.1,
        "Tminus": 100.1,
    },
)


BM2 = BenchmarkPoint(
    {
        "RGScale": 125.0,
        "v0": 246.0,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        "mh1": 125.0,
        "mh2": 160.0,
        "a2": 1.2,
        "b4": 1.0,
    },
    {},
)


BM3 = BenchmarkPoint(
    {
        # "RGScale" : 91.1876,
        "RGScale": 125.0,  # <- Benoit benchmark
        "v0": 246.0,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1": 125.0,
        "mh2": 160.0,
        "a2": 1.6,
        "b4": 1.0,
    },
    {},
)

BM4 = BenchmarkPoint(
    {
        "RGScale": 125.0,
        "v0": 246.22,
        "MW": 80.379,
        "MZ": 91.1876,
        "Mt": 173.0,
        "g3": 1.2279920495357861,
        "mh1": 125.0,
        "mh2": 160.0,
        "a2": 1.2,
        "b4": 1.0,
    },
    {},
)

##
singletBenchmarks = [BM1, BM2, BM3, BM4]
