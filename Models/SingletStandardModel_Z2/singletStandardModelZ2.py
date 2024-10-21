"""
This Python script, singletStandardModelZ2.py,
implements a minimal Standard Model extension via
a scalar singlet and incorporating a Z2 symmetry.
Only the top quark is out of equilibrium, and only
QCD-interactions are considered in the collisions.

Features:
- Definition of the extended model parameters including the singlet scalar field.
- Definition of the out-of-equilibrium particles.
- Implementation of the one-loop thermal potential, without high-T expansion.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

Note:
This benchmark model was used to compare against the results of
B. Laurent and J. M. Cline, First principles determination
of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
doi:10.1103/PhysRevD.106.023501
As a consequence, we overwrite the default WallGo thermal functions
Jb/Jf. 
"""

import os
import sys
import pathlib
import argparse
from typing import TYPE_CHECKING
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import Fields, GenericModel, Particle
from WallGo.interpolatableFunction import EExtrapolationType

from WallGo.PotentialTools import EffectivePotentialNoResum, EImaginaryOption

# Add the Models folder to the path; need to import the base example
# template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint  # pylint: disable=C0411, C0413, E0401

if TYPE_CHECKING:
    import WallGoCollision


class SingletSMZ2(GenericModel):
    r"""
    Z2 symmetric SM + singlet model.

    The potential is given by:
    V = msq |phi|^2 + lam |phi|^4 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    def __init__(self, allowOutOfEquilibriumGluon: bool = False):
        """
        Initialize the SingletSMZ2 model.

        Parameters
        ----------
            FIXME
        Returns
        ----------
        cls: SingletSMZ2
            An object of the SingletSMZ2 class.
        """

        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialxSMZ2(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles(allowOutOfEquilibriumGluon)
        self.bIsGluonOffEq = allowOutOfEquilibriumGluon

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 2

    def getEffectivePotential(self) -> "EffectivePotentialxSMZ2":
        return self.effectivePotential

    # ~

    def defineParticles(self, includeGluon: bool) -> None:
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

        # === Top quark ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def topMsqVacuum(fields: Fields) -> Fields:
            return 0.5 * self.modelParameters["yt"] ** 2 * fields.getField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"] ** 2 * np.transpose(
                [fields.getField(0), 0 * fields.getField(1)]
            )

        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuark = Particle(
            "top",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=12,
        )
        self.addParticle(topQuark)

        if includeGluon:

            # === SU(3) gluon ===
            # The msqVacuum function must take a Fields object and return an
            # array of length equal to the number of points in fields.
            def gluonMsqVacuum(fields: Fields) -> Fields:
                return np.zeros_like(fields.getField(0))

            def gluonMsqDerivative(fields: Fields) -> Fields:
                return np.zeros_like(fields)

            def gluonMsqThermal(T: float) -> float:
                return self.modelParameters["g3"] ** 2 * T**2 * 2.0

            gluon = Particle(
                "gluon",
                index=1,
                msqVacuum=gluonMsqVacuum,
                msqDerivative=gluonMsqDerivative,
                msqThermal=gluonMsqThermal,
                statistics="Boson",
                totalDOFs=16,
            )
            self.addParticle(gluon)

    def calculateLagrangianParameters(
        self, inputParameters: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate Lagrangian parameters based on the input parameters.

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

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        massh1 = inputParameters["mh1"]  # 125 GeV
        massh2 = inputParameters["mh2"]

        # these are direct inputs:
        modelParameters["RGScale"] = inputParameters["RGScale"]
        modelParameters["a2"] = inputParameters["a2"]
        modelParameters["b4"] = inputParameters["b4"]

        modelParameters["lambda"] = 0.5 * massh1**2 / v0**2
        # should be same as the following:
        # modelParameters["msq"] = -massh1**2 / 2.
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2
        modelParameters["b2"] = massh2**2 - 0.5 * v0**2 * inputParameters["a2"]

        # Then the gauge and Yukawa sector
        massT = inputParameters["Mt"]
        massW = inputParameters["MW"]
        massZ = inputParameters["MZ"]

        # helper
        g0 = 2.0 * massW / v0

        modelParameters["g1"] = g0 * np.sqrt((massZ / massW) ** 2 - 1)
        modelParameters["g2"] = g0
        # Just take QCD coupling as input
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(0.5) * g0 * massT / massW

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


# end model


class EffectivePotentialxSMZ2(EffectivePotentialNoResum):
    """
    Effective potential for the SingletSMZ2 model.

    This class inherits from the EffectivePotentialNoResum class and provides the
    necessary methods for calculating the effective potential.

    For this benchmark model we use the UNRESUMMED 4D potential.
    Furthermore we use customized interpolation tables for Jb/Jf
    """

    # ~ EffectivePotential interface
    fieldCount = 2
    """How many classical background fields"""

    effectivePotentialError = 1e-8
    """
    Relative accuracy at which the potential can be computed. Here it is set by the
    error tolerance of the thermal integrals Jf/Jb.
    """

    def __init__(self, owningModel: SingletSMZ2) -> None:
        """
        Initialize the EffectivePotentialxSMZ2.
        """

        # Not using default Jb/Jf interpolation tables here
        super().__init__(
            imaginaryOption=EImaginaryOption.PRINCIPAL_PART,
            useDefaultInterpolation=False,
        )

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 29
        self.numFermionDof = 90

        """For this benchmark model we do NOT use the default integrals from WallGo.
        This is because the benchmark points we're comparing with were originally done
        with integrals from CosmoTransitions. In real applications we recommend using the WallGo default implementations.
        """
        self._configureBenchmarkIntegrals()

    def _configureBenchmarkIntegrals(self) -> None:
        """
        Configure the benchmark integrals.

        Parameters
        ----------
        None

        Returns
        ----------
        None
        """
        # Load custom interpolation tables for Jb/Jf. These should be
        # the same as what CosmoTransitions version 2.0.2 provides by default.
        thisFileDirectory = os.path.dirname(os.path.abspath(__file__))
        self.integrals.Jb.readInterpolationTable(
            os.path.join(thisFileDirectory, "interpolationTable_Jb_testModel.txt"),
        )
        self.integrals.Jf.readInterpolationTable(
            os.path.join(thisFileDirectory, "interpolationTable_Jf_testModel.txt"),
        )

        self.integrals.Jb.disableAdaptiveInterpolation()
        self.integrals.Jf.disableAdaptiveInterpolation()

        """Force out-of-bounds constant extrapolation because this is
        what CosmoTransitions does
        => not really reliable for very negative (m/T)^2 ! 
        Strictly speaking: For x > xmax, CosmoTransitions just returns 0. 
        But a constant extrapolation is OK since the integral is very small 
        at the upper limit.
        """

        self.integrals.Jb.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

        self.integrals.Jf.setExtrapolationType(
            extrapolationTypeLower=EExtrapolationType.CONSTANT,
            extrapolationTypeUpper=EExtrapolationType.CONSTANT,
        )

    def evaluate(
        self, fields: Fields, temperature: float
    ) -> float | np.ndarray:
        """
        Evaluate the effective potential.

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

        # For this benchmark we don't use high-T approx and no resummation
        # just Coleman-Weinberg with numerically evaluated thermal 1-loop

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        fields = Fields(fields)
        v, x = fields.getField(0), fields.getField(1)

        msq = self.modelParameters["msq"]
        b2 = self.modelParameters["b2"]
        lam = self.modelParameters["lambda"]
        b4 = self.modelParameters["b4"]
        a2 = self.modelParameters["a2"]

        # tree level potential
        potentialTree = (
            0.5 * msq * v**2
            + 0.25 * lam * v**4
            + 0.5 * b2 * x**2
            + 0.25 * b4 * x**4
            + 0.25 * a2 * v**2 * x**2
        )

        # Particle masses and coefficients for the CW potential
        bosonInformation = self.bosonInformation(fields)
        fermionInformation = self.fermionInformation(fields)

        potentialTotal = (
            potentialTree
            + self.constantTerms(temperature)
            + self.potentialOneLoop(bosonInformation, fermionInformation)
            + self.potentialOneLoopThermal(
                bosonInformation, fermionInformation, temperature
            )
        )

        return np.array(potentialTotal)

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
        dofsBoson = self.numBosonDof - 14
        dofsFermion = self.numFermionDof - 12  # we only included top quark loops

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0

    def bosonInformation(  # pylint: disable=too-many-locals
        self, fields: Fields
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes parameters for the one-loop potential (Coleman-Weinberg and thermal).

        Parameters
        ----------
        fields: Fields
            The field configuration

        Returns
        ----------
        massSq: array_like
            A list of the boson particle masses at each input point `field`.
        degreesOfFreedom: array_like
            The number of degrees of freedom for each particle.
        c: array_like
            A constant used in the one-loop effective potential
        rgScale : array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential
        """
        v, x = fields.getField(0), fields.getField(1)

        # Scalar masses, just diagonalizing manually. matrix (A C // C B)
        mass00 = (
            self.modelParameters["msq"]
            + 0.5 * self.modelParameters["a2"] * x**2
            + 3 * self.modelParameters["lambda"] * v**2
        )
        mass11 = (
            self.modelParameters["b2"]
            + 0.5 * self.modelParameters["a2"] * v**2
            + 3 * self.modelParameters["b4"] * x**2
        )
        mass01 = self.modelParameters["a2"] * v * x
        thingUnderSqrt = (mass00 - mass11) ** 2 + 4 * mass01**2

        msqEig1 = 0.5 * (mass00 + mass11 - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (mass00 + mass11 + np.sqrt(thingUnderSqrt))

        mWsq = self.modelParameters["g2"] ** 2 * v**2 / 4
        mZsq = mWsq + self.modelParameters["g1"] ** 2 * v**2 / 4
        # Goldstones
        mGsq = (
            self.modelParameters["msq"]
            + self.modelParameters["lambda"] * v**2
            + 0.5 * self.modelParameters["a2"] * x**2
        )

        # h, s, chi, W, Z
        massSq = np.column_stack((msqEig1, msqEig2, mGsq, mWsq, mZsq))
        degreesOfFreedom = np.array([1, 1, 3, 6, 3])
        c = np.array([3 / 2, 3 / 2, 3 / 2, 5 / 6, 5 / 6])
        rgScale = self.modelParameters["RGScale"] * np.ones(5)

        return massSq, degreesOfFreedom, c, rgScale

    def fermionInformation(
        self, fields: Fields
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes parameters for the one-loop potential (Coleman-Weinberg and thermal).

        Parameters
        ----------
        fields: Fields
            The field configuration

        Returns
        ----------
        massSq: array_like
            A list of the fermion particle masses at each input point `field`.
        degreesOfFreedom: array_like
            The number of degrees of freedom for each particle.
        c: array_like
            A constant used in the one-loop effective potential
        rgScale : array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential
        """

        v = fields.getField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])

        c = np.array([3 / 2])
        rgScale = np.array([self.modelParameters["RGScale"]])

        return massSq, degreesOfFreedom, c, rgScale


class SingletStandardModelExample(WallGoExampleBase):
    """
    Sets up the Standard Model + singlet, computes or loads the collison
    integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateCollisions = False

        self.bShouldRecalculateMatrixElements = False

        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/MatrixElements_QCD.json"
        )
        self.matrixElementInput = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/qcd.m"
        )

    # ~ Begin WallGoExampleBase interface
    def initCommandLineArgs(self) -> argparse.ArgumentParser:
        """Non-abstract override to add a SM + singlet specific command line option"""

        argParser: argparse.ArgumentParser = super().initCommandLineArgs()
        argParser.add_argument(
            "--outOfEquilibriumGluon",
            help="Treat the SU(3) gluons as out-of-equilibrium particle species",
            action="store_true",
        )
        return argParser
        
    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return SingletSMZ2(self.cmdArgs.outOfEquilibriumGluon)

    def initCollisionModel(
        self, wallGoModel: "SingletSMZ2"
    ) -> "WallGoCollision.PhysicsModel":
        """Initialize the Collision model and set the seed."""

        import WallGoCollision  # pylint: disable = C0415

        # Collision integrations utilize Monte Carlo methods, so RNG is involved.
        # We can set the global seed for collision integrals as follows.
        # This is optional; by default the seed is 0.
        WallGoCollision.setSeed(0)

        # This example comes with a very explicit example function on how to setup and
        # configure the collision module. It is located in a separate module
        # (same directory) to avoid bloating this file. Import and use it here.
        from exampleCollisionDefs import (
            setupCollisionModel_QCD,
        )  # pylint: disable = C0415

        collisionModel = setupCollisionModel_QCD(
            wallGoModel.modelParameters,
            wallGoModel.bIsGluonOffEq,
        )

        return collisionModel
    
    def updateCollisionModel(
        self,
        inWallGoModel: "SingletSMZ2",
        inOutCollisionModel: "WallGoCollision.PhysicsModel",
    ) -> None:
        """Propagate changes in WallGo model to the collision model.
        For this example we just need to update the QCD coupling and
        fermion/gluon thermal masses.
        """
        import WallGoCollision  # pylint: disable = C0415

        changedParams = WallGoCollision.ModelParameters()

        gs = inWallGoModel.modelParameters["g3"]  # names differ for historical reasons
        changedParams.addOrModifyParameter("gs", gs)
        changedParams.addOrModifyParameter(
            "mq2", gs**2 / 6.0
        )  # quark thermal mass^2 in units of T
        changedParams.addOrModifyParameter(
            "mg2", 2.0 * gs**2
        )  # gluon thermal mass^2 in units of T

        inOutCollisionModel.updateParameters(changedParams)

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        """Non-abstract override"""

        import WallGoCollision  # pylint: disable = C0415

        """Configure the integrator. Default settings should be reasonably OK so you
        can modify only what you need, or skip this step entirely. Here we set
        everything manually to show how it's done.
        """
        integrationOptions = WallGoCollision.IntegrationOptions()
        integrationOptions.calls = 50000
        integrationOptions.maxTries = 50
        # collision integration momentum goes from 0 to maxIntegrationMomentum.
        # This is in units of temperature
        integrationOptions.maxIntegrationMomentum = 20
        integrationOptions.absoluteErrorGoal = 1e-8
        integrationOptions.relativeErrorGoal = 1e-1

        inOutCollisionTensor.setIntegrationOptions(integrationOptions)

        """We can also configure various verbosity settings that are useful when
        you want to see what is going on in long-running integrations. These 
        include progress reporting and time estimates, as well as a full result dump
        of each individual integral to stdout. By default these are all disabled. 
        Here we enable some for demonstration purposes.
        """
        verbosity = WallGoCollision.CollisionTensorVerbosity()
        verbosity.bPrintElapsedTime = (
            True  # report total time when finished with all integrals
        )

        """Progress report when this percentage of total integrals (approximately)
        have been computed. Note that this percentage is per-particle-pair, ie. 
        each (particle1, particle2) pair reports when this percentage of their
        own integrals is done. Note also that in multithreaded runs the 
        progress tracking is less precise.
        """
        verbosity.progressReportPercentage = 0.25

        # Print every integral result to stdout? This is very slow and
        # verbose, intended only for debugging purposes
        verbosity.bPrintEveryElement = False

        inOutCollisionTensor.setIntegrationVerbosity(verbosity)

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:
        """We load the configs from a file for this example."""
        inOutManager.config.loadConfigFromFile(
            pathlib.Path(self.exampleBaseDirectory / "singletStandardModelZ2Config.ini")
        )
        super().configureManager(inOutManager)

    def updateModelParameters(
        self, model: "SingletSMZ2", inputParameters: dict[str, float]
    ) -> None:
        """Convert SM + singlet inputs to Lagrangian params and update internal
        model parameters. This example is constructed so that the effective
        potential and particle mass functions refer to model.modelParameters,
        so be careful not to replace that reference here.
        """

        # oldParams = model.modelParameters.copy()

        model.updateModel(inputParameters)

        """Collisions integrals for this example depend only on the QCD coupling,
        if it changes we must recompute collisions before running the wall solver.
        The bool flag here is inherited from WallGoExampleBase and temperatureed
        in runExample(). But since we want to keep the example simple, we skip
        this check and assume the existing data is OK.
        (FIXME?)
        """
        self.bShouldRecalculateCollisions = False

        """
        newParams = model.modelParameters
        if not oldParams or newParams["g3"] != oldParams["g3"]:
            self.bNeedsNewCollisions = True
        """

    def getBenchmarkPoints(self) -> list[ExampleInputPoint]:
        """
        Input parameters, phase info, and settings for the effective potential and
        wall solver for the xSM benchmark point.
        """

        output: list[ExampleInputPoint] = []
        output.append(
            ExampleInputPoint(
                {
                    "RGScale": 125.0,
                    "v0": 246.0,
                    "MW": 80.379,
                    "MZ": 91.1876,
                    "Mt": 173.0,
                    "g3": 1.2279920495357861,
                    "mh1": 125.0,
                    "mh2": 120.0,
                    "a2": 0.9,
                    "b4": 1.0,
                },
                WallGo.PhaseInfo(
                    temperature=100.0,  # nucleation temperature
                    phaseLocation1=WallGo.Fields([0.0, 200.0]),
                    phaseLocation2=WallGo.Fields([246.0, 0.0]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=10.0, fieldValueVariationScale=[10.0, 10.0]
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

    example = SingletStandardModelExample()
    example.runExample()
