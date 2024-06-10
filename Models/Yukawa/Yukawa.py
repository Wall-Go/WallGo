"""
A simple example model, of a real scalar field coupled to a Dirac fermion
"""
import numpy as np
import WallGo


class Yukawa(WallGo.GenericModel):
    """
    The Yukawa model, inheriting from WallGo.GenericModel.
    """

    # specifying this is necessary
    fieldCount = 1

    def __init__(self, inputParameters: dict[str, float]):
        """
        Initialisation of the Yukawa model:
            - stores modelParameters
            - initialises effectivePotential
            - constructs list of Particles
        """

        # must initialise and store the model parameters with this variable name
        self.modelParameters = inputParameters

        # must do the same for the effective potential
        self.effectivePotential = EffectivePotentialYukawa(
            self.modelParameters, self.fieldCount
        )

        # constructing the list of particles, starting with psi
        # taking its fluctuations out of equilibrium
        y = self.modelParameters["y"]
        psiMsqVacuum = lambda fields: self.modelParameters["mf"] + y * fields.GetField(0)
        psiMsqDerivative = lambda fields: y
        psiMsqThermal = lambda T: 1 / 16 * y**2 * T**2
        psi = WallGo.Particle(
            "psi",
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=4,
        )
        self.addParticle(psi)

        # now adding the phi field, assuming fluctuations in equilibrium
        msq = self.modelParameters["msq"]
        g = self.modelParameters["g"]
        lam = self.modelParameters["lam"]
        phiMsqVacuum = lambda fields: (
            msq + g * fields.GetField(0) + lam / 2 * fields.GetField(0) ** 2
        )
        phiMsqDerivative = lambda fields: g + lam * fields.GetField(0)
        phiMsqThermal = lambda T: 1 / 24 * (lam + 4 * y**2) * T**2
        phi = WallGo.Particle(
            "phi",
            msqVacuum=phiMsqVacuum,
            msqDerivative=phiMsqDerivative,
            msqThermal=phiMsqThermal,
            statistics="Boson",
            inEquilibrium=True,
            ultrarelativistic=True,
            totalDOFs=1,
        )
        self.addParticle(phi)


class EffectivePotentialYukawa(WallGo.EffectivePotential):
    """
    The effective potential for a specific model inherits from the
    WallGo class EffectivePotential.
    """

    def evaluate(self, fields: WallGo.Fields, temperature: float) -> float:
        """
        It is necessary to define a member function called 'evaluate'
        which returns the value of the potential.
        """
        # getting the field from the list of fields (here just of length 1)
        fields = WallGo.Fields(fields)
        phi = fields.GetField(0)

        # the constant term
        f_0 = -np.pi**2 / 90 * (1 + 4 * 7 / 8) * temperature**4

        # coefficients of the temperature and field dependent terms
        y = self.modelParameters["y"]
        mf = self.modelParameters["mf"]
        sigma_eff = (
            self.modelParameters["sigma"]
            + (self.modelParameters["g"] + 4 * y * mf) * temperature**2
        )
        msq_eff = (
            self.modelParameters["msq"]
            + (self.modelParameters["lam"] + 4 * y**2) * temperature**2
        )

        # the combined result
        return (
            f_0
            + sigma_eff * phi
            + 1 / 2 * msq_eff * phi**2
            + 1 / 6 * self.modelParameters["g"] * phi**3
            + 1 / 24 * self.modelParameters["lam"] * phi**4
        )
