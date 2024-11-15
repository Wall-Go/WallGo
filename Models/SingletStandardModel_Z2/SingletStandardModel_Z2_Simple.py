"""
This is a simplified version of the xSM, with an effective potential obtained in the
high-temperature expansion.
This file is used for running tests, and can not be used as a standalone model
file for computing the wall velocity.
"""

import pathlib
import numpy as np

# WallGo imports
from WallGo import Fields
from WallGo import EffectivePotential
from .singletStandardModelZ2 import SingletSMZ2, EffectivePotentialxSMZ2


# Z2 symmetric SM + singlet model. V = muHsq |phi|^2 + lHH (|phi|^2)^2 + 1/2 muSsq S^2 + 1/4 lSS S^4 + 1/2 lHS |phi|^2 S^2
class SingletSM_Z2_Simple(SingletSMZ2):

    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateLagrangianParameters(
            initialInputParameters
        )

        self.effectivePotential = EffectivePotentialxSM_Z2_Simple(self)


# Overwrite more complicated effective potential keeping only O(g^2T^4) bits
class EffectivePotentialxSM_Z2_Simple(EffectivePotential):

    fieldCount = 2

    def __init__(self, owningModel: SingletSMZ2) -> None:
        self.owner = owningModel
        self.modelParameters = owningModel.modelParameters

    def evaluate(
        self,
        fields: Fields,
        temperature: float,
    ) -> float:

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        fields = Fields(fields)
        v, x = fields.getField(0), fields.getField(1)

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)

        muHsq = thermalParameters["muHsq"]
        lHH = thermalParameters["lHH"]
        muSsq = thermalParameters["muSsq"]
        lSS = thermalParameters["lSS"]
        lHS = thermalParameters["lHS"]

        # tree level potential
        V0 = (
            0.5 * muHsq * v**2
            + 0.25 * lHH * v**4
            + 0.5 * muSsq * x**2
            + 0.25 * lSS * x**4
            + 0.25 * lHS * v**2 * x**2
        )

        return V0 + self.constantTerms(temperature)

    def constantTerms(self, temperature: float) -> float:
        return -107.75 * np.pi**2 / 90 * temperature**4

    # Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature
        muHsq = self.modelParameters["muHsq"]
        lHH = self.modelParameters["lHH"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        muSsq = self.modelParameters["muSsq"]
        lHS = self.modelParameters["lHS"]
        lSS = self.modelParameters["lSS"]

        # LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["muHsq"] = (
            muHsq
            + T**2 / 16.0 * (3.0 * g2**2 + g1**2 + 4.0 * yt**2 + 8.0 * lHH)
            + T**2 * lHS / 24.0
        )

        thermalParameters["muSsq"] = muSsq + T**2 * (1.0 / 6.0 * lHS + 1.0 / 4.0 * lSS)

        return thermalParameters
