import pathlib
import numpy as np

# WallGo imports
from WallGo import Fields
from .SingletStandardModel_Z2 import SingletSM_Z2, EffectivePotentialxSM_Z2


# Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class SingletSM_Z2_Simple(SingletSM_Z2):

    def __init__(self, initialInputParameters: dict[str, float]):

        super().__init__(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialxSM_Z2_Simple(
            self.modelParameters, self.fieldCount
        )


# Overwrite more complicated effective potential keeping only O(g^2T^4) bits
class EffectivePotentialxSM_Z2_Simple(EffectivePotentialxSM_Z2):

    def evaluate(
        self, fields: Fields, temperature: float, checkForImaginary=False,
    ) -> float:

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        v, x = fields.GetField(0), fields.GetField(1)

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)

        msq = thermalParameters["msq"]
        lam = thermalParameters["lambda"]
        b2 = thermalParameters["b2"]
        b4 = thermalParameters["b4"]
        a2 = thermalParameters["a2"]

        # tree level potential
        V0 = (
            0.5 * msq * v**2
            + 0.25 * lam * v**4
            + 0.5 * b2 * x**2
            + 0.25 * b4 * x**4
            + 0.25 * a2 * v**2 * x**2
        )

        return V0 + self.constantTerms(temperature)

    def constantTerms(self, temperature: float) -> float:
        return -107.75 * np.pi ** 2 / 90 * temperature ** 4

    # Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature
        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        b2 = self.modelParameters["b2"]
        a2 = self.modelParameters["a2"]
        b4 = self.modelParameters["b4"]

        # LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["msq"] = (
            msq
            + T**2 / 16.0 * (3.0 * g2**2 + g1**2 + 4.0 * yt**2 + 8.0 * lam)
            + T**2 * a2 / 24.0
        )

        thermalParameters["b2"] = b2 + T**2 * (1.0 / 6.0 * a2 + 1.0 / 4.0 * b4)

        return thermalParameters