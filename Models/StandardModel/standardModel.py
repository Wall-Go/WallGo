"""
This Python script, standardModel.py,
implements the Standard Model, with a light Higgs mass.
This model is ruled out by actual measurements of the Higgs mass, which
show that it is 125 GeV, but this file can be used to compare with 
earlier computations performed in the literature.

Features:
- Definition of the standard model parameters.
- Definition of the out-of-equilibrium particles, in our case the top and W-boson.
- Implementation of the thermal potential, with high-T expansion.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

Note:
This benchmark is used to compare against the results of
G. Moore and T. Prokopec, How fast can the wall move?
A Study of the electroweak phase transition dynamics, Phys.Rev.D 52 (1995) 7182-7204
doi:10.1103/PhysRevD.52.7182
"""

import sys
import pathlib
import numpy as np
from typing import TYPE_CHECKING

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import EffectivePotential, Fields, GenericModel, Particle

# Add the Models folder to the path; need to import the base
# example template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint  # pylint: disable=C0411, C0413, E0401

if TYPE_CHECKING:
    import WallGoCollision


class StandardModel(GenericModel):
    r"""
    The Standard model, with a light Higgs mass, such that the
    electroweak phase transition becomes fist order.

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    def __init__(self) -> None:
        """
        Initialize the SM model.

        Parameters
        ----------

        Returns
        ----------
        cls: StandardModel
            An object of the StandardModel class.
        """

        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialSM(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

        # ~ GenericModel interface

    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def getEffectivePotential(self) -> "EffectivePotentialSM":
        return self.effectivePotential

    # ~

    def defineParticles(self) -> None:
        """
        Define the particles for the model.
        Note that the particle list only needs to contain the
        particles that are relevant for the Boltzmann equations.
        The particles relevant to the effective potential are
        included independently.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        self.clearParticles()

        ## === Top quark ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def topMsqVacuum(fields: Fields) -> Fields:
            return 0.5 * self.modelParameters["yt"] ** 2 * fields.getField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"] ** 2 * fields.getField(0)

        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuarkL = Particle(
            name="TopL",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=6,
        )
        self.addParticle(topQuarkL)

        topQuarkR = Particle(
            name="TopR",
            index=1,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=6,
        )
        self.addParticle(topQuarkR)

        ## === SU(2) gauge boson ===
        def WMsqVacuum(fields: Fields) -> Fields:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * fields.getField(0) ** 2 / 4

        def WMsqDerivative(fields: Fields) -> Fields:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * fields.getField(0) / 2

        def WMsqThermal(T: float) -> float:  # pylint: disable=invalid-name
            return self.modelParameters["g2"] ** 2 * T**2 * 11.0 / 6.0

        wBoson = Particle(
            name="W",
            index=2,
            msqVacuum=WMsqVacuum,
            msqDerivative=WMsqDerivative,
            msqThermal=WMsqThermal,
            statistics="Boson",
            totalDOFs=9,
        )
        self.addParticle(wBoson)

    def calculateLagrangianParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate the model parameters based on the input parameters.

        Parameters
        ----------
        inputParameters: dict[str, float]
            A dictionary of input parameters for the model.

        Returns
        ----------
        modelParameters: dict[str, float]
            A dictionary of calculated model parameters.
        """
        modelParameters = {}

        # Zero-temperature vev
        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0

        # Zero-temperature masses
        massH = inputParameters["mH"]
        massW = inputParameters["mW"]
        massZ = inputParameters["mZ"]
        massT = inputParameters["mt"]

        modelParameters["mW"] = massW
        modelParameters["mZ"] = massZ
        modelParameters["mt"] = massT

        # helper
        g0 = 2.0 * massW / v0

        # Gauge couplings
        modelParameters["g1"] = g0 * np.sqrt((massZ / massW) ** 2 - 1)
        modelParameters["g2"] = g0
        modelParameters["g3"] = inputParameters["g3"]
        modelParameters["yt"] = np.sqrt(1.0 / 2.0) * g0 * massT / massW

        modelParameters["lambda"] = inputParameters["mH"] ** 2 / (2 * v0**2)

        # The following parameters are defined on page 6 of hep-ph/9506475
        bconst = 3 / (64 * np.pi**2 * v0**4) * (2 * massW**4 + massZ**4 - 4 * massT**4)

        modelParameters["D"] = (
            1 / (8 * v0**2) * (2 * massW**2 + massZ**2 + 2 * massT**2)
        )
        modelParameters["E0"] = 1 / (12 * np.pi * v0**3) * (4 * massW**3 + 2 * massZ**3)

        modelParameters["T0sq"] = (
            1 / 4 / modelParameters["D"] * (massH**2 - 8 * bconst * v0**2)
        )
        modelParameters["C0"] = (
            1 / (16 * np.pi**2) * (1.42 * modelParameters["g2"] ** 4)
        )

        return modelParameters

    def updateModel(self, newInputParams: dict[str, float]) -> None:
        """Computes new Lagrangian parameters from given input and caches
        them internally. These changes automatically propagate to the associated
        EffectivePotential, particle masses etc.
        """
        newParams = self.calculateLagrangianParameters(newInputParams)
        # Copy to the model dict, do NOT replace the reference.
        # This way the changes propagate to Veff and particles
        self.modelParameters.update(newParams)


class EffectivePotentialSM(EffectivePotential):
    """
    Effective potential for the Standard Model.

    This class inherits from the EffectivePotential class.
    """

    # ~ EffectivePotential interface
    fieldCount = 1
    """How many classical background fields"""

    effectivePotentialError = 1e-15
    """
    Relative accuracy at which the potential can be computed. Here the potential is
    polynomial so we can set it to the machine precision.
    """

    def __init__(self, owningModel: StandardModel) -> None:
        """
        Initialize the EffectivePotentialSM.
        """
        super().__init__()

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 28
        self.numFermionDof = 90

    def evaluate(  # pylint: disable=R0914
        self,
        fields: Fields,
        temperature: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Evaluate the effective potential. We implement the effective potential
        of eq. (7) of hep-ph/9506475.

        Parameters
        ----------
        fields: Fields
            The field configuration
        temperature: float
            The temperature

        Returns
        ----------
        potentialTotal: complex | np.ndarray
            The value of the effective potential
        """
        # phi ~ 1/sqrt(2) (0, v)
        fields = Fields(fields)
        v = fields.getField(0) + 0.0000001

        T = temperature + 0.0000001

        ab = 49.78019250
        af = 3.111262032

        mW = self.modelParameters["mW"]
        mZ = self.modelParameters["mZ"]
        mt = self.modelParameters["mt"]

        # Implement finite-temperature corrections to the modelParameters lambda,
        # C0 and E0, as on page 6 and 7 of hep-ph/9506475.
        lambdaT = self.modelParameters["lambda"] - 3 / (
            16 * np.pi * np.pi * self.modelParameters["v0"] ** 4
        ) * (
            2 * mW**4 * np.log(mW**2 / (ab * T**2) + 1e-100)
            + mZ**4 * np.log(mZ**2 / (ab * T**2) + 1e-100)
            - 4 * mt**4 * np.log(mt**2 / (af * T**2) + 1e-100)
        )

        cT: float | np.ndarray = self.modelParameters["C0"] + 1 / (
            16 * np.pi * np.pi
        ) * (4.8 * self.modelParameters["g2"] ** 2 * lambdaT - 6 * lambdaT**2)

        # HACK: take the absolute value of lambdaT here,
        # to avoid taking the square root of a negative number
        eT: float | np.ndarray = (
            self.modelParameters["E0"]
            + 1 / (12 * np.pi) * (3 + 3**1.5) * np.abs(lambdaT) ** 1.5
        )

        potentialT: float | np.ndarray = (
            self.modelParameters["D"] * (T**2 - self.modelParameters["T0sq"]) * v**2
            - cT * T**2 * pow(v, 2) * np.log(np.abs(v / T))
            - eT * T * pow(v, 3)
            + lambdaT / 4 * pow(v, 4)
        )

        potentialTotal = np.real(potentialT + self.constantTerms(T))

        return np.asanyarray(potentialTotal)

    def constantTerms(self, temperature: np.ndarray | float) -> np.ndarray | float:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T
        expansion these are just (minus) the ideal gas pressure of light particles that
        were not integrated over in the one-loop part.

        See Eq. (39) in hep-ph/0510375 for general LO formula

        Parameters
        ----------
        temperature: array-like (float)
            The temperature

        Returns
        ----------
        constantTerms: array-like (float)
            The value of the field-independent contribution to the effective potential
        """

        # How many degrees of freedom we have left. The number of DOFs
        # that were included in evaluate() is hardcoded
        dofsBoson = self.numBosonDof
        dofsFermion = self.numFermionDof

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0


class StandardModelExample(WallGoExampleBase):
    """
    Sets up the standard model, computes or loads the collision
    integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateMatrixElements = False
        self.bShouldRecalculateCollisions = False
        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/matrixElements.ew.json"
        )

    # ~ Begin WallGoExampleBase interface

    def getDefaultCollisionDirectory( # pylint: disable = W0246
        self, momentumGridSize: int
    ) -> pathlib.Path:
        """Returns the path to the directory with collisions. Does not
        have to be overwritten for this example, but included for completeness"""
        return super().getDefaultCollisionDirectory(momentumGridSize)

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return StandardModel()

    def initCollisionModel(
        self, wallGoModel: "StandardModel"
    ) -> "WallGoCollision.PhysicsModel":
        """Initialize the Collision model and set the seed."""

        import WallGoCollision  # pylint: disable = C0415

        # This example comes with a very explicit example function on how to setup and
        # configure the collision module. It is located in a separate module
        # (same directory) to avoid bloating this file. Import and use it here.
        from exampleCollisionDefs import (  # pylint: disable = W0246
            setupCollisionModel_QCDEW,
        )  # pylint: disable = C0415

        collisionModel = setupCollisionModel_QCDEW(
            wallGoModel.modelParameters,
        )

        return collisionModel

    def updateCollisionModel(
        self,
        inWallGoModel: "StandardModel",
        inOutCollisionModel: "WallGoCollision.PhysicsModel",
    ) -> None:
        """Propagate changes in WallGo model to the collision model."""
        import WallGoCollision  # pylint: disable = C0415

        changedParams = WallGoCollision.ModelParameters()

        gs = inWallGoModel.modelParameters["g3"]  # names differ for historical reasons
        gw = inWallGoModel.modelParameters["g2"]  # names differ for historical reasons
        
        
        # Note that the particular values of masses here are for a comparison with arXiv:hep-ph/9506475.
        # For proceeding beyond the leading-log approximation one should use the asymptotic masses.
        # For quarks we include the thermal mass only
        changedParams.addOrModifyParameter("gs", gs)
        changedParams.addOrModifyParameter("gw", gw)
        changedParams.addOrModifyParameter(
            "mq2", gs**2 / 6.0
        )  # quark thermal mass^2 in units of T
        changedParams.addOrModifyParameter(
            "mg2", 2.0 * gs**2
        )  # gluon thermal mass^2 in units of T
        changedParams.addOrModifyParameter(
            "mw2", 3.0 * gw**2 / 5.0
        )  # W boson thermal mass^2 in units of T

        inOutCollisionModel.updateParameters(changedParams)

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        """Non-abstract override"""

        import WallGoCollision  # pylint: disable = C0415

        """Configure the integrator. Default settings should be reasonably OK so
        you can modify only what you need, or skip this step entirely. Here we
        set everything manually to show how it's done.
        """
        integrationOptions = WallGoCollision.IntegrationOptions()
        integrationOptions.calls = 50000
        integrationOptions.maxTries = 10
        # collision integration momentum goes from 0 to maxIntegrationMomentum.
        # This is in units of temperature
        integrationOptions.maxIntegrationMomentum = 20
        integrationOptions.absoluteErrorGoal = 1e-5
        integrationOptions.relativeErrorGoal = 1e-1

        inOutCollisionTensor.setIntegrationOptions(integrationOptions)

        """We can also configure various verbosity settings that are useful when you
        want to see what is going on in long-running integrations. These include 
        progress reporting and time estimates, as well as a full result dump of each
        individual integral to stdout. By default these are all disabled. 
        Here we enable some for demonstration purposes.
        """
        verbosity = WallGoCollision.CollisionTensorVerbosity()
        verbosity.bPrintElapsedTime = (
            True  # report total time when finished with all integrals
        )

        """Progress report when this percentage of total integrals (approximately) have
        been computed. Note that this percentage is per-particle-pair, ie. each
        (particle1, particle2) pair reports when this percentage of their own integrals
        is done. Note also that in multithreaded runs the progress tracking is less 
        precise.
        """
        verbosity.progressReportPercentage = 0.25

        # Print every integral result to stdout? This is very slow and verbose,
        # intended only for debugging purposes
        verbosity.bPrintEveryElement = False

        inOutCollisionTensor.setIntegrationVerbosity(verbosity)

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:
        inOutManager.config.loadConfigFromFile(
            pathlib.Path(self.exampleBaseDirectory / "standardModelConfig.ini")
        )
        super().configureManager(inOutManager)

    def updateModelParameters(
        self, model: "StandardModel", inputParameters: dict[str, float]
    ) -> None:
        """Convert SM inputs to Lagrangian params and update internal model parameters.
        This example is constructed so that the effective potential and particle mass
        functions refer to model.modelParameters, so be careful not to replace that
        reference here.
        """

        # oldParams = model.modelParameters.copy()
        
        model.updateModel(inputParameters)

        """Collisions integrals for this example depend on the QCD and Electroweak
        coupling, so if these change we should mark the collision data as outdated
        by setting self.bNeedsNewCollisions = True. But since we want to keep
        the example simple, we skip this check and assume the existing data is OK.
        (FIXME?)
        """
        self.bNeedsNewCollisions = False  # pylint: disable = W0201

        """
        if (
            not oldParams
            or newParams["g3"] != oldParams["g3"]
            or newParams["g2"] != oldParams["g2"]
        ):
            self.bNeedsNewCollisions = True
        """

    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        """
        Input parameters, phase info, and settings for the effective potential and
        wall solver for the standard model benchmark points, with different values
        of the Higgs mass.
        """
        valuesMH = [0.0, 34.0, 50.0, 70.0, 81.0]
        valuesTn = [57.192, 70.579, 83.426, 102.344, 113.575]

        output: list[ExampleInputPoint] = []

        for i in range(len(valuesMH)):  # pylint: disable=C0200
            output.append(
                ExampleInputPoint(
                    {
                        "v0": 246.0,
                        "mW": 80.4,
                        "mZ": 91.2,
                        "mt": 174.0,
                        "g3": 1.2279920495357861,
                        "mH": valuesMH[i],
                    },
                    WallGo.PhaseInfo(
                        temperature=valuesTn[i],
                        phaseLocation1=WallGo.Fields([0.0]),
                        phaseLocation2=WallGo.Fields([valuesTn[i]]),
                    ),
                    WallGo.VeffDerivativeSettings(
                        temperatureVariationScale=1., fieldValueVariationScale=[50.0]
                    ),
                    WallGo.WallSolverSettings(
                        # we actually do both cases in the common example
                        bIncludeOffEquilibrium=True,
                        meanFreePathScale=100.0, # In units of 1/Tnucl
                        wallThicknessGuess=5.0, # In units of 1/Tnucl
                    ),
                )
            )

        return output

    # ~ End WallGoExampleBase interface


if __name__ == "__main__":

    example = StandardModelExample()
    example.runExample()
