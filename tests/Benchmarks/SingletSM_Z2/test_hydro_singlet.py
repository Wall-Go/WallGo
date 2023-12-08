import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint

import WallSpeed


## This tends to be slow as it's often the first test to use the hydro fixture, so initialization takes long. Hence the marker
@pytest.mark.slow
def test_hydroBoundaries(singletBenchmarkHydro: Tuple[WallSpeed.Hydro, BenchmarkPoint]):

    hydro, BM = singletBenchmarkHydro

    ## Jouguet velocity
    vJ_expected = 0.6444
    res = hydro.vJ
    assert res == pytest.approx(vJ_expected, rel=1e-2)

    vw_in = 0.5229
    res = hydro.findHydroBoundaries(vw_in)

    ## Goal values for hydro boundaries. These are the first 4 return values from findHydroBoundaries so check those only
    c1 = BM.expectedResults["c1"]
    c2 = BM.expectedResults["c2"]
    Tplus = BM.expectedResults["Tplus"]
    Tminus = BM.expectedResults["Tminus"]

    np.testing.assert_allclose(res[:4], (c1, c2, Tplus, Tminus), rtol=1e-2)
