"""
A simple model of a scalar coupled to an out-of-equilibrium
fermion. The model is based on the Yukawa model, but the masses
and interactions are treated as independent (this is not
physical, but makes the computation simpler).
"""
import pathlib
from pathlib import Path
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
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

    def defineParticles(self) -> None:
        """
        Define the out-of-equilibrium particles for the model.
        """
        self.clearParticles()

        # Vacuum mass squared
        def psiMsqVacuum(fields: Fields) -> Fields:
            return self.modelParameters["mf"] + self.modelParameters[
                "yf"
            ] * fields.getField(0)
        
        # Field-derivative of the vacuum mass squared
        def psiMsqDerivative(fields: Fields) -> Fields:  # pylint: disable = W0613
            return self.modelParameters["yf"]

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
            totalDOFs=4,
        )
        psiR = Particle(
            "psiR",
            index=2,
            msqVacuum=psiMsqVacuum,
            msqDerivative=psiMsqDerivative,
            msqThermal=psiMsqThermal,
            statistics="Fermion",
            totalDOFs=4,
        )
        self.addParticle(psiL)
        self.addParticle(psiR)

    def calculateLagrangianParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate parameters for potential and collisions, based on the input
        parameters. Here the model parameters are direct input.
        """
        modelParameters = {}

        modelParameters = inputParameters

        return modelParameters
    
    def updateModel(self, newInputParams: dict[str, float]) -> None:
        """Computes new Lagrangian parameters from given input and caches
        them internally. These changes automatically propagate to the
        associated EffectivePotential, particle masses etc.
        """
        newParams = self.calculateLagrangianParameters(newInputParams)
        # Copy to the model dict, do NOT replace the reference.
        # This way the changes propagate to Veff and particles
        self.modelParameters.update(newParams)


class EffectivePotentialSimple(WallGo.EffectivePotential):
    """
    Effective potential for the simple model.

    This class inherits from the EffectivePotential class and provides the
    necessary methods for calculating the effective potential.
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

    def evaluate(
        self, fields: Fields, temperature: float
    ) -> float | np.ndarray:
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
    model = SimpleModel()
    manager.registerModel(model)

    inputParameters = {
                    "yf": -0.18,
                    "mf": 2.0,
                    "gamma": -4.0,
                    "kappa": -0.6,
                    "msq": 1,
                    "msqTh": 2./115.,
                    "cubic": -0.77,
                    "quartic": 0.0055,
                    }
    
    model.updateModel(inputParameters)
    
    manager.setupThermodynamicsHydrodynamics(
                WallGo.PhaseInfo(
                    temperature=51.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([0.]),
                    phaseLocation2=WallGo.Fields([82.5223]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=1.0,
                    fieldValueVariationScale=[
                        50.0,
                    ],
                )
            )
    
    manager.config.set("PolynomialGrid", "spatialGridSize", "20")
    
    pathtoCollisions = pathlib.Path(__file__).resolve().parent/ Path(f"CollisionOutput_N11")
    manager.setPathToCollisionData(pathtoCollisions)
    
    # ---- Solve wall speed in Local Thermal Equilibrium (LTE) approximation
    vwLTE = manager.wallSpeedLTE()
    print(f"LTE wall speed:    {vwLTE:.6f}")

    solverSettings = WallGo.WallSolverSettings(
                    bIncludeOffEquilibrium=False,
                    meanFreePathScale=1000.0, # In units of 1/Tnucl
                    wallThicknessGuess=5.0, # In units of 1/Tnucl
                )

    results = manager.solveWall(
                    solverSettings
    )

    print(f"Wall velocity without out-of-equilibrium contributions {results.wallVelocity:.6f}")

    solverSettings.bIncludeOffEquilibrium = True

    results = manager.solveWall(
                    solverSettings
    )

    print(f"Wall velocity with out-of-equilibrium contributions {results.wallVelocity:.6f}")



## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()