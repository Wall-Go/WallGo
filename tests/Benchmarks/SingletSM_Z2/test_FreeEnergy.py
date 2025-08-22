import pytest
import numpy as np
from typing import Tuple

from tests.BenchmarkPoint import BenchmarkPoint, BenchmarkModel

import WallGo


@pytest.mark.parametrize("T", [90, 110])
def test_freeEnergy_singletSimple(
    singletSimpleBenchmarkFreeEnergy: Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint],
    T: float,
) -> None:
    """
    Testing numerics of FreeEnergy
    """
    freeEnergy1, freeEnergy2, BM = singletSimpleBenchmarkFreeEnergy

    # exact results
    thermalParameters = freeEnergy1.effectivePotential.getThermalParameters(T)
    f0 = -107.75 * np.pi ** 2 / 90 * T ** 4
    vExact = np.sqrt(-thermalParameters["muHsq"] / thermalParameters["lHH"])
    VvExact = -0.25 * thermalParameters["muHsq"] ** 2 / thermalParameters["lHH"]
    xExact = np.sqrt(-thermalParameters["muSsq"] / thermalParameters["lSS"])
    VxExact = -0.25 * thermalParameters["muSsq"] ** 2 / thermalParameters["lSS"]

    # tolerance
    rTol = 1e-5
    aTol = rTol * T

    # evaluate the free energy objects
    f1: WallGo.FreeEnergyValueType = freeEnergy1(T)
    f2: WallGo.FreeEnergyValueType = freeEnergy2(T)

    ## We get two fields (v = Higgs, x = singlet) and the Veff value at this field configuration
    fields, veffValue = f1.fieldsAtMinimum, f1.veffValue

    v, x = fields.getField(0), fields.getField(1)
    assert 0 == pytest.approx(v, abs=aTol)
    assert xExact == pytest.approx(x, rel=rTol)
    assert f0 + VxExact == pytest.approx(veffValue, rel=rTol)

    fields, veffValue = f2.fieldsAtMinimum, f2.veffValue

    v, x = fields.getField(0), fields.getField(1)
    assert vExact == pytest.approx(v, rel=rTol)
    assert 0 == pytest.approx(x, abs=aTol)
    assert f0 + VvExact == pytest.approx(veffValue, rel=rTol)


def test_freeEnergy_singletSimple_passingArrays(
    singletSimpleBenchmarkModel: BenchmarkModel,
    singletSimpleBenchmarkFreeEnergy: Tuple[WallGo.FreeEnergy, WallGo.FreeEnergy, BenchmarkPoint],
) -> None:
    """
    Testing building FreeEnergy from passing arrays
    """
    freeEnergy1, freeEnergy2, BM = singletSimpleBenchmarkFreeEnergy

    # temperature range
    temperatureRange = np.linspace(90, 110, num=50)
    nT = len(temperatureRange)

    vList = np.zeros((nT, 2))
    vVeffList = np.zeros(nT)
    xList = np.zeros((nT, 2))
    xVeffList = np.zeros(nT)

    # tolerance
    tol = 1e-15

    for iT, T in enumerate(temperatureRange):
        # exact results
        thermalParameters = freeEnergy1.effectivePotential.getThermalParameters(T)
        f0 = -107.75 * np.pi ** 2 / 90 * T ** 4

        vExact = np.sqrt(-thermalParameters["muHsq"] / thermalParameters["lHH"])
        VvExact = -0.25 * thermalParameters["muHsq"] ** 2 / thermalParameters["lHH"]
        vList[iT, 0] = vExact
        vVeffList[iT] = f0 + VvExact

        xExact = np.sqrt(-thermalParameters["muSsq"] / thermalParameters["lSS"])
        VxExact = -0.25 * thermalParameters["muSsq"] ** 2 / thermalParameters["lSS"]

        xList[iT, 1] = xExact
        xVeffList[iT] = f0 + VxExact

    freeEnergyHighT = WallGo.FreeEnergyArrays(
        temperatures=temperatureRange,
        minimumList=xList,
        potentialEffList=xVeffList,
        allowedDiscrepancy=tol,
    )

    freeEnergyLowT = WallGo.FreeEnergyArrays(
        temperatures=temperatureRange,
        minimumList=vList,
        potentialEffList=vVeffList,
        allowedDiscrepancy=tol,
    )

    # free energies for both phases
    freeEnergy1 = WallGo.FreeEnergy(
        singletSimpleBenchmarkModel.model.getEffectivePotential(),
        temperatureRange[10],
        WallGo.Fields(vList[10]),
    )
    freeEnergy2 = WallGo.FreeEnergy(
        singletSimpleBenchmarkModel.model.getEffectivePotential(),
        temperatureRange[10],
        WallGo.Fields(xList[10]),
    )

    freeEnergy1.constructInterpolationFromArray(
        freeEnergyArrays=freeEnergyLowT,
        dT=abs(temperatureRange[1]-temperatureRange[0])
    )

    freeEnergy2.constructInterpolationFromArray(
        freeEnergyArrays=freeEnergyHighT,
        dT=abs(temperatureRange[1]-temperatureRange[0])
    )
