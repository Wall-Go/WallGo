"""
A simple model of a scalar coupled to an out-of-equilibrium fermion. 
The model uses the same matrix elements as the Yukawa model,
but the potential has been modified (the coefficients no longer
correspond to a real physical model).
The couplings entering in the matrix elements are also chosen differently
from the real Yukawa model.

This model file only computes the wall velocity if the collision terms are
provided. The corresponding collisions can be obtained with 
simpleModelWithCollisionGeneration.py.
"""

import pathlib
import numpy as np

# WallGo imports
import WallGo
from WallGo import Fields, GenericModel, Particle


class SimpleModel(GenericModel):
    """
    The simple model, inheriting from WallGo.GenericModel.
    """

    def __init__(self) -> None:
        """
        Initialize the simple model.
        """
        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialSimple(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def getEffectivePotential(self) -> "EffectivePotentialSimple":
        return self.effectivePotential

    # ~

    def defineParticles(self) -> None:
        """
        Define the out-of-equilibrium particles for the model.
        """
        self.clearParticles()

        # Vacuum mass squared
        def psiMsqVacuum(fields: Fields) -> Fields:
            return (
                self.modelParameters["mf"]
                + self.modelParameters["yf"] * fields.getField(0)
            ) ** 2

        # Field-derivative of the vacuum mass squared
        def psiMsqDerivative(fields: Fields) -> Fields:  # pylint: disable = W0613
            return 2 * (
                self.modelParameters["yf"] * self.modelParameters["mf"]
                + self.modelParameters["yf"] ** 2 * fields.getField(0)
            )

        # Thermal lass
        def psiMsqThermal(T: float) -> float:
            return 1 / 16 * self.modelParameters["yf"] ** 2 * T**2

        psiL = Particle(
            "psiL",
            index=1,
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            totalDOFs=2,
        )
        psiR = Particle(
            "psiR",
            index=2,
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            totalDOFs=2,
        )
        self.addParticle(psiL)
        self.addParticle(psiR)


class EffectivePotentialSimple(WallGo.EffectivePotential):
    """
    Effective potential for the simple model.
    """

    def __init__(self, owningModel: SimpleModel) -> None:
        """
        Initialize the EffectivePotentialSimple.
        """
        super().__init__()

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

    # ~ EffectivePotential interface
    fieldCount = 1
    """How many classical background fields"""
    # ~

    def evaluate(self, fields: Fields, temperature: float) -> float | np.ndarray:
        """
        Evaluate the effective potential as a function of the fields and temperature.
        """
        # getting the field from the list of fields (here just of length 1)
        fields = WallGo.Fields(fields)
        phi = fields.getField(0)

        # the constant term
        f0 = -np.pi**2 / 90 * (1 + 4 * 7 / 8) * temperature**4

        # coefficients of the temperature and field dependent terms
        msq = self.modelParameters["msq"]
        msqTh = self.modelParameters["msqTh"]
        cubic = self.modelParameters["cubic"]
        quartic = self.modelParameters["quartic"]

        potentialTotal = (
            f0
            + 1 / 2 * (msq + msqTh * temperature**2) * phi**2
            + cubic * phi**3
            + quartic * phi**4
        )

        return np.array(potentialTotal)


def main() -> None:

    manager = WallGo.WallGoManager()
    manager.config.set("PolynomialGrid", "spatialGridSize", "20")

    pathtoCollisions = pathlib.Path(__file__).resolve().parent / pathlib.Path(
        f"CollisionOutput_N11"
    )
    manager.setPathToCollisionData(pathtoCollisions)

    model = SimpleModel()
    manager.registerModel(model)

    inputParameters = {
        "yf": -0.18,
        "mf": 2.0,
        "gamma": -4.0,
        "kappa": -0.6,
        "msq": -1.0,
        "msqTh": 2.0 / 115.0,
        "cubic": -0.77,
        "quartic": 0.0055,
    }

    model.modelParameters.update(inputParameters)

    manager.setupThermodynamicsHydrodynamics(
        WallGo.PhaseInfo(
            temperature=51.0,  # nucleation temperature
            phaseLocation1=WallGo.Fields([0.0]),
            phaseLocation2=WallGo.Fields([82.5223]),
        ),
        WallGo.VeffDerivativeSettings(
            temperatureVariationScale=1.0,
            fieldValueVariationScale=[
                50.0,
            ],
        ),
    )

    # ---- Solve wall speed in Local Thermal Equilibrium (LTE) approximation
    vwLTE = manager.wallSpeedLTE()
    print(f"LTE wall speed:    {vwLTE:.6f}")

    solverSettings = WallGo.WallSolverSettings(
        bIncludeOffEquilibrium=False,
        meanFreePathScale=1000.0,  # In units of 1/Tnucl
        wallThicknessGuess=5.0,  # In units of 1/Tnucl
    )

    results = manager.solveWall(solverSettings)

    print(
        f"Wall velocity without out-of-equilibrium contributions {results.wallVelocity:.6f}"
    )

    solverSettings.bIncludeOffEquilibrium = True

    results = manager.solveWall(solverSettings)

    print(
        f"Wall velocity with out-of-equilibrium contributions {results.wallVelocity:.6f}"
    )


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
