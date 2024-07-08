import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallGo


## This can be somewhat slow as this is often the first test that uses Hydro fixtures
def test_Jouguet(singletBenchmarkHydrodynamics: Tuple[WallGo.Hydrodynamics, BenchmarkPoint]):

    hydrodynamics, BM = singletBenchmarkHydrodynamics

    vJ_expected = BM.expectedResults["vJ"]
    vJ_result = hydrodynamics.vJ

    assert vJ_result == pytest.approx(vJ_expected, rel=1e-3)
    

## This can be slow if Jb/Jf need to be evaluated at very negative (m/T)^2
@pytest.mark.slow
def test_hydroBoundaries(singletBenchmarkHydrodynamics: Tuple[WallGo.Hydrodynamics, BenchmarkPoint]):

    hydrodynamics, BM = singletBenchmarkHydrodynamics

    vw_in = 0.5229
    res = hydrodynamics.findHydroBoundaries(vw_in)

    ## Goal values for hydrodynamics boundaries. These are the first 4 return values from findHydroBoundaries so check those only
    c1 = BM.expectedResults["c1"]
    c2 = BM.expectedResults["c2"]
    Tplus = BM.expectedResults["Tplus"]
    Tminus = BM.expectedResults["Tminus"]

    np.testing.assert_allclose(res[:4], (c1, c2, Tplus, Tminus), rtol=1e-3)


## Wall velocity in the Local Thermal Equilibrium approximation
def test_vwLTE(singletBenchmarkHydrodynamics: Tuple[WallGo.Hydrodynamics, BenchmarkPoint]):

    hydrodynamics, BM = singletBenchmarkHydrodynamics

    vwLTE_expected = BM.expectedResults["vwLTE"]
    vwLTE_result = hydrodynamics.findvwLTE()

    assert vwLTE_result == pytest.approx(vwLTE_expected, rel=1e-3)

    