"""
This Python script, manySinglets.py,
implements a Standard Model extension via
a N scalar singlets and incorporating a Z2 symmetry.
Only the top quark is out of equilibrium, and only
QCD-interactions are considered in the collisions.

Features:
- Definition of the extended model parameters including the N singlet scalar fields.
- Definition of the out-of-equilibrium particles.
- Implementation of the one-loop thermal potential, with high-T expansion.
- Functions for computing the critical temperature and position of the minimum.

Usage:
- This script is intended to compute the wall speed of the model.

Dependencies:
- NumPy for numerical calculations
- the WallGo package
- CollisionIntegrals in read-only mode using the default path for the collision
integrals as the "CollisonOutput" directory

"""

import sys
import pathlib
import argparse
import numpy as np
from typing import TYPE_CHECKING

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import Fields, GenericModel, Particle, EffectivePotential

# Add the Models folder to the path; need to import the base example
# template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase
from wallGoExampleBase import ExampleInputPoint

if TYPE_CHECKING:
    import WallGoCollision


class NSinglets(GenericModel):
    r"""
    Generalization of the Z2 symmetric SM + singlet model now including N singlets.

    The potential is given by:
    V = msq |phi|^2 + lam |phi|^4 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    def __init__(self, nbrSinglets: int):
        """
        Initialize the NSinglets model.

        Parameters
        ----------
            FIXME
        Returns
        ----------
        cls: NSinglets
            An object of the NSinglets class.
        """
        self.nbrSinglets = nbrSinglets

        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialNSinglets(self, self.fieldCount)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return self.nbrSinglets + 1  # N singlets and 1 Higgs

    def getEffectivePotential(self) -> "EffectivePotentialNSinglets":
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
                [
                    (1 if i == 0 else 0) * fields.getField(i)
                    for i in range(self.fieldCount)
                ]
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

        # Higgs VEV
        v0 = inputParameters["v0"]
        # Higgs mass
        mh = inputParameters["mh"]  # 125 GeV

        # Couplings between Higgs and singlets (should be an array of length N)
        modelParameters["lHS"] = np.array(inputParameters["lHS"])
        # Singlets self-couplings (array of length N)
        modelParameters["lSS"] = np.array(inputParameters["lSS"])

        # Higgs self-coupling
        modelParameters["lHH"] = 0.5 * mh**2 / v0**2
        # mu^2 parameters
        modelParameters["muHsq"] = -(mh**2) / 2
        modelParameters["muSsq"] = np.array(inputParameters["muSsq"])

        # Then the gauge and Yukawa sector
        massT = inputParameters["Mt"]
        massW = inputParameters["MW"]
        massZ = inputParameters["MZ"]

        # helper
        g0 = 2.0 * massW / v0
        modelParameters["g1"] = g0 * np.sqrt((massZ / massW) ** 2 - 1)
        modelParameters["g2"] = g0
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(1.0 / 2.0) * g0 * massT / massW

        # High-T expansion coefficients
        modelParameters["cH"] = (
            6 * modelParameters["lHH"]
            + sum(modelParameters["lHS"])
            + 6 * modelParameters["yt"] ** 2
            + (9 / 4) * modelParameters["g2"] ** 2
            + (3 / 4) * modelParameters["g1"] ** 2
        ) / 12
        modelParameters["cS"] = (
            3 * modelParameters["lSS"] + 4 * modelParameters["lHS"]
        ) / 12

        return modelParameters

    def updateModel(self, newInputParams: dict[str, float]) -> None:
        """Computes new Lagrangian parameters from given input and caches them
        internally. These changes automatically propagate to the associated
        EffectivePotential, particle masses etc.
        """
        newParams = self.calculateLagrangianParameters(newInputParams)
        # Copy to the model dict, do NOT replace the reference.
        # This way the changes propagate to Veff and particles
        self.modelParameters.update(newParams)


# end model


class EffectivePotentialNSinglets(EffectivePotential):
    r"""
    Effective potential for the NSinglets model.

    For this benchmark model we use the UNRESUMMED 4D potential.
    Furthermore we use customized interpolation tables for Jb/Jf

    Implementation of the Z2-symmetric N-singlet scalars + SM model with the high-T
    1-loop thermal corrections. This model has the potential
    :math:`V = \frac{1}{2}\sum_{i=0}^N\mu_i^2(T)\phi_i^2
    + \frac{1}{4}\sum_{i,j=0}^N\lambda_{ij}\phi_i^2\phi_j^2`
    where :math:`\phi_0` is assumed to be the Higgs and :math:`\phi_{i>0}` the
    singlet scalars.
    For simplicity, we only consider models with no couplings between the different
    singlets; only couplings between the Higgs and the singlets are allowed.
    This means :math:`\lambda_{ij}=0` when :math:`i,j>0` and :math:`i\neq j`.
    """

    # ~ EffectivePotential interface
    fieldCount = 3
    """How many classical background fields"""

    effectivePotentialError = 1e-8
    """
    Relative accuracy at which the potential can be computed. Here it is set by the
    error tolerance of the thermal integrals Jf/Jb.
    """

    def __init__(self, owningModel: NSinglets, fieldCount: int) -> None:
        """
        Initialize the EffectivePotentialNSinglets.
        """

        super().__init__()

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 27 + fieldCount
        self.numFermionDof = 90

    def canTunnel(self, tunnelingTemperature: float = None) -> bool:
        """
        Function used to determine if tunneling can happen with this potential.
        Verifies that the Higgs phase exists at T=0 and that it is stable and the
        true vacuum. Also verifies that both phases exist and are stable at T=Tc
        (or T=tunnelingTemperature).

        Parameters
        ----------
        tunnelingTemperature : float, optional
            Temperature at which the tunneling takes place. If None, uses Tc.
            The default is None.

        Returns
        -------
        tunnel : bool
            Returns True if all the conditions mentioned above are satisfied.
            Returns False otherwise.

        """
        tunnel = True

        # Higgs phase is the true vacuum at T=0
        if self.modelParameters["muHsq"] ** 2 / self.modelParameters["lHH"] <= sum(
            self.modelParameters["muSsq"] ** 2 / self.modelParameters["lSS"]
        ):
            print("Higgs phase is not the true vacuum at T=0")
            print(
                f"""{self.modelParameters["muHsq"]**2/self.modelParameters["lHH"] -
                     sum(self.modelParameters["muSsq"]**2/
                         self.modelParameters["lSS"])=}"""
            )
            tunnel = False

        # Higgs phase exists at T=0
        if self.modelParameters["muHsq"] >= 0 or self.modelParameters["lHH"] <= 0:
            print("Higgs phase doesn't exist at T=0")
            print(
                f"""{self.modelParameters["muHsq"]=} {self.modelParameters["lHH"]=}"""
            )
            tunnel = False
        # Higgs phase is stable at T=0
        if np.any(
            self.modelParameters["muSsq"]
            - self.modelParameters["lHS"]
            * self.modelParameters["muHsq"]
            / self.modelParameters["lHH"]
            <= 0
        ):
            print("Higgs phase is not stable at T=0")
            print(
                f"""{self.modelParameters["muSsq"]-self.modelParameters["lHS"]*
                     self.modelParameters["muHsq"]/self.modelParameters["lHH"]=}"""
            )
            tunnel = False

        if tunnelingTemperature is None:
            # If no temperature was provided, computes and uses Tc
            T = self.findTc()
            print(f"Tc={T}")
            if T is None:
                tunnel = False
        else:
            T = tunnelingTemperature

        if T is not None:
            muSsqT = self.modelParameters["muSsq"] + self.modelParameters["cS"] * T**2
            muHsqT = self.modelParameters["muHsq"] + self.modelParameters["cH"] * T**2

            # Higgs phase exists at T=Tc
            if muHsqT >= 0:
                print("Higgs phase doesn't exist at T=Tc")
                print(f"{muHsqT=}")
                tunnel = False
            # Higgs phase is stable at T=Tc
            if np.any(
                muSsqT
                - self.modelParameters["lHS"] * muHsqT / self.modelParameters["lHH"]
                <= 0
            ):
                print("Higgs phase is not stable at T=Tc")
                print(
                    f"""{muSsqT-self.modelParameters["lHS"]*
                         muHsqT/self.modelParameters["lHH"]}"""
                )
                tunnel = False

            # Singlets phase exists at T=Tc
            if np.any(muSsqT >= 0) or np.any(self.modelParameters["lSS"] <= 0):
                print("Singlets phase doesn't exist at T=Tc")
                print(f"{muSsqT=} {self.modelParameters['lSS']=}")
                tunnel = False
            # Singlets phase is stable at T=Tc
            if (
                muHsqT
                - sum(
                    self.modelParameters["lHS"] * muSsqT / self.modelParameters["lSS"]
                )
                <= 0
            ):
                print("Singlets phase is not stable at T=Tc")
                print(
                    f"""{muHsqT - sum(self.modelParameters["lHS"]*
                                      muSsqT/self.modelParameters["lSS"])=}"""
                )
                tunnel = False

        return tunnel

    def findTc(self) -> float:
        """
        Computes the critical temperature

        Returns
        -------
        float
            Value of the critical temperature. If there is no solution, returns None.

        """
        a = self.modelParameters["cH"] ** 2 / self.modelParameters["lHH"] - sum(
            self.modelParameters["cS"] ** 2 / self.modelParameters["lSS"]
        )
        b = 2 * (
            self.modelParameters["cH"]
            * self.modelParameters["muHsq"]
            / self.modelParameters["lHH"]
            - sum(
                self.modelParameters["cS"]
                * self.modelParameters["muSsq"]
                / self.modelParameters["lSS"]
            )
        )
        c = self.modelParameters["muHsq"] ** 2 / self.modelParameters["lHH"] - sum(
            self.modelParameters["muSsq"] ** 2 / self.modelParameters["lSS"]
        )

        discr = b**2 - 4 * a * c
        if discr < 0:
            # The discriminant is negative, which would lead to imaginary Tc^2.
            print("No critical temperature : negative discriminant")
            return None

        # Finds the two solutions for Tc^2, and keep the smallest positive one.
        Tc1 = (-b + np.sqrt(discr)) / (2 * a)
        Tc2 = (-b - np.sqrt(discr)) / (2 * a)

        if Tc1 <= 0 and Tc2 <= 0:
            print("Negative critical temperature squared")
            return None
        if Tc1 > 0 and Tc2 > 0:
            return min(np.sqrt(Tc1), np.sqrt(Tc2))
        if Tc1 > 0:
            return np.sqrt(Tc1)
        if Tc2 > 0:
            return np.sqrt(Tc2)

        print("No critical temperature : both solutions are negative")
        return None

    def findPhases(self, temperature: float) -> tuple:
        """
        Computes the position of the two phases at T=temperature.

        Parameters
        ----------
        temperature : float
            Temperature at which to evaluate the position of the phases.

        Returns
        -------
        phase1 : array-like
            Array containing the position of the singlet phase.
        phase2 : array-like
            Array containing the position of the Higgs phase.

        """
        muHsqT = (
            self.modelParameters["muHsq"] + self.modelParameters["cH"] * temperature**2
        )
        muSsqT = (
            self.modelParameters["muSsq"] + self.modelParameters["cS"] * temperature**2
        )

        phase1 = np.sqrt(np.append([0], -muSsqT / self.modelParameters["lSS"]))
        phase2 = np.sqrt(
            np.append(
                [-muHsqT / self.modelParameters["lHH"]], (self.fieldCount - 1) * [0]
            )
        )

        return phase1, phase2

    def evaluate(  # pylint: disable = R0914
        self, fields: Fields, temperature: float
    ) -> np.ndarray:
        """
        Evaluates the tree-level potential with the 1-loop high-T thermal corrections.

        Parameters
        ----------
        fields : Fields
            Fields object containing the VEVs of the fields.
        temperature : float or array-like
            Temperature at which the potential is evaluated.

        Returns
        -------
        array-like
            Values of the potential.

        """

        h, s = fields[..., 0], fields[..., 1:]
        temperature = np.array(temperature)

        muHsq = self.modelParameters["muHsq"]
        muSsq = self.modelParameters["muSsq"]
        lHH = self.modelParameters["lHH"]
        lHS = self.modelParameters["lHS"]
        lSS = self.modelParameters["lSS"]
        cH = self.modelParameters["cH"]
        cS = self.modelParameters["cS"]

        muHsqT = muHsq + cH * temperature**2
        if len(temperature.shape) > 0:  # If temperature is an array
            muSsqT = muSsq + cS * temperature[:, None] ** 2
        else:  # If temperature is a float
            muSsqT = muSsq + cS * temperature**2

        # Tree level potential with high-T 1-loop thermal corrections.
        potentialTree = (
            0.5 * muHsqT * h**2
            + 0.5 * np.sum(muSsqT * s**2, axis=-1)
            + 0.25 * lHH * h**4
            + 0.25 * np.sum(lSS * s**4, axis=-1)
            + 0.5 * h**2 * np.sum(lHS * s**2, axis=-1)
        )

        # Adding the terms proportional to T^4
        potentialTotal = potentialTree + self.constantTerms(temperature)

        return potentialTotal

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

        # Fermions contribute with a magic 7/8 prefactor as usual.
        # Overall minus sign since Veff(min) = -pressure
        return (
            -(self.numBosonDof + (7.0 / 8.0) * self.numFermionDof)
            * np.pi**2
            * temperature**4
            / 90.0
        )


class NSingletsModelExample(WallGoExampleBase):
    """
    Sets up the standard model coupled to two singlets,
    computes or loads the collison integrals, and computes the wall velocity.
    """

    def __init__(self) -> None:
        """"""
        self.bShouldRecalculateMatrixElements = False
        self.bShouldRecalculateCollisions = False

        self.matrixElementFile = pathlib.Path(
            self.exampleBaseDirectory / "MatrixElements/MatrixElements_QCD.json"
        )

    # ~ Begin WallGoExampleBase interface
    def initCommandLineArgs(self) -> argparse.ArgumentParser:
        """Non-abstract override to add a SM + singlet specific cmd option"""

        argParser: argparse.ArgumentParser = super().initCommandLineArgs()
        return argParser

    def getDefaultCollisionDirectory(self, momentumGridSize: int) -> pathlib.Path:
        """Returns the path to the directory with collisions."""
        return pathlib.Path(super().getDefaultCollisionDirectory(momentumGridSize))

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        # Number of singlets
        nbrSinglets = 2
        return NSinglets(nbrSinglets)

    def initCollisionModel(
        self, wallGoModel: "NSinglets"
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

        collisionModel = setupCollisionModel_QCD(wallGoModel.modelParameters)

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
            pathlib.Path(self.exampleBaseDirectory / "manySingletsConfig.ini")
        )
        super().configureManager(inOutManager)

    def updateModelParameters(
        self, model: "NSinglets", inputParameters: dict[str, float]
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
        The bool flag here is inherited from WallGoExampleBase and checked in
        runExample(). But since we want to keep the example simple, we skip this
        check and assume the existing data is OK.
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
        wall solver for the manySinglets benchmark points.
        """
        inputParameters = {
            "RGScale": 125.0,
            "v0": 246.0,
            "MW": 80.379,
            "MZ": 91.1876,
            "Mt": 173.0,
            "g3": 1.2279920495357861,
            # scalar specific
            "mh": 125.0,
            "muSsq": [-8000, -10000],
            "lHS": [0.75, 0.9],
            "lSS": [0.5, 0.7],
        }

        model = self.initWallGoModel()
        model.updateModel(inputParameters)

        Tc = model.effectivePotential.findTc()
        if Tc is None:
            return 0
        Tn = 0.8 * Tc
        if model.effectivePotential.canTunnel(Tn) is False:
            print("Tunneling impossible. Try with different parameters.")
            return 0

        phase1, phase2 = model.effectivePotential.findPhases(Tn)

        output: list[ExampleInputPoint] = []
        output.append(
            ExampleInputPoint(
                inputParameters,
                WallGo.PhaseInfo(
                    temperature=Tn,  # nucleation temperature
                    phaseLocation1=WallGo.Fields(phase1[None, :]),
                    phaseLocation2=WallGo.Fields(phase2[None, :]),
                ),
                WallGo.VeffDerivativeSettings(
                    temperatureVariationScale=10.0,
                    fieldValueVariationScale=[10.0, 10.0, 10.0],
                ),
                WallGo.WallSolverSettings(
                    # we actually do both cases in the common example
                    bIncludeOffEquilibrium=True,
                    meanFreePathScale=100.0,  # In units of 1/Tnucl
                    wallThicknessGuess=5.0,  # In units of 1/Tnucl
                ),
            )
        )

        return output

    # ~ End WallGoExampleBase interface


if __name__ == "__main__":

    example = NSingletsModelExample()
    example.runExample()
