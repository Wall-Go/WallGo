import pytest
import numpy as np
import WallGo


def test_BM1():
    Tc = 108.22
    Tn = 100
    vJ = 0.6444
    vw = 0.5229
    c1 = 3331587978
    c2 = 2976953742
    Tplus = 103.1
    Tminus = 100.1
    mod = WallGo.Model(125, 120, 1.0, 0.9)
    params = mod.params
    res = mod.Vtot([[110, 130]], 100)
    assert res == pytest.approx(-1.19018205e09, rel=1e-2)
    free = WallGo.FreeEnergy(mod.Vtot, Tc, Tnucl=Tn, params=params)
    res = free.findPhases(100)
    np.testing.assert_allclose(
        res, [[195.03215146, 0.0], [0.0, 104.86914171]], rtol=1e-2
    )
    # res = mod.find_Tc()
    res = 108.22
    assert res == pytest.approx(Tc, rel=1e-2)
    free.interpolateMinima(0, 1.2 * Tc, 1)
    thermo = WallGo.Thermodynamics(free)
    hydro = WallGo.Hydro(thermo)
    res = hydro.vJ
    assert res == pytest.approx(vJ, rel=1e-2)
    res = hydro.findHydroBoundaries(vw)
    np.testing.assert_allclose(res[:4], (c1, c2, Tplus, Tminus), rtol=1e-2)


def test_BM2():
    mod = WallGo.Model(125, 160.0, 1.0, 1.2)
    res = mod.Vtot([[110, 130]], 100)
    assert res == pytest.approx(-1.15450678e09, rel=1e-2)


def test_BM3():
    mod = WallGo.Model(125, 160, 1.0, 1.6)
    res = mod.Vtot([[110, 130]], 100)
    assert res == pytest.approx(-1.23684861e09, rel=1e-2)


def test_BM4():
    mod = WallGo.Model(125, 80, 1.0, 0.5)
    res = mod.Vtot([[100, 100]], 100)
    assert res == pytest.approx(-1210419844, rel=1e-2)
