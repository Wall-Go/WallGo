import numpy as np
from .BenchmarkPoint import BenchmarkPoint

BM1 = BenchmarkPoint( 
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
    # otherData:
    {
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
