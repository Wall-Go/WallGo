"""
This Python script, inertDoubletModel.py,
implements an extension of the Standard Model by
an inert SU(2) doublet. This is a special case of the
Two Higgs Doublet Model.
The top quark and W-bosons are out of equilibrium, and
QCD and weak interactions are considered in the collisions.

Features:
- Definition of the extended model parameters including the inert doublet.
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
S. Jiang, F. Peng Huang, and X. Wang, Bubble wall velocity during electroweak
phase transition in the inert doublet model, Phys.Rev.D 107 (2023) 9, 095005
doi:10.1103/PhysRevD.107.095005
"""

import sys
import pathlib
from typing import TYPE_CHECKING
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo._initializeInternal()
from WallGo import Fields, GenericModel, Particle

from WallGo.PotentialTools import EffectivePotentialNoResum, EImaginaryOption

# Add the Models folder to the path; need to import the base example
# template
modelsBaseDir = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(modelsBaseDir))

from wallGoExampleBase import WallGoExampleBase  # pylint: disable=C0411, C0413, E0401
from wallGoExampleBase import ExampleInputPoint

if TYPE_CHECKING:
    import WallGoCollision


# Inert doublet model, as implemented in 2211.13142
class InertDoubletModel(GenericModel):
    r"""
    Inert doublet model.

    The tree-level potential is given by
    V = msq |phi|^2 + msq2 |eta|^2 + lambda |phi|^4 + lambda2 |eta|^4
        + lambda3 |phi|^2 |eta|^2 + lambda4 |phi^dagger eta|^2
        + (lambda5 (phi^dagger eta)^2 +h.c.)
    Note that there are some differences in normalization compared to
    Jiang, Peng Huang, and Wang.

    Only the Higgs field undergoes the phase transition, the new scalars only
    modify the effective potential.

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    # ~

    def __init__(self) -> None:
        """
        Initialize the InertDoubletModel.
        """

        self.modelParameters: dict[str, float] = {}

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialIDM(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def getEffectivePotential(self) -> "EffectivePotentialIDM":
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

    ## Go from whatever input params --> action params
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

        # Zero-tempreature Higgs vev
        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0

        # Higgs parameters
        massh = inputParameters["mh"]
        modelParameters["lambda"] = 0.5 * massh**2 / v0**2
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2

        ## Then the Yukawa sector
        massT = inputParameters["Mt"]
        modelParameters["yt"] = np.sqrt(2.0) * massT / v0

        ## Then the inert doublet parameters
        massH = inputParameters["mH"]
        massA = inputParameters["mA"]
        massHp = inputParameters["mHp"]

        lambda5 = (massH**2 - massA**2) / v0**2
        lambda4 = -2 * (massHp**2 - massA**2) / v0**2 + lambda5
        lambda3 = 2 * inputParameters["lambdaL"] - lambda4 - lambda5
        msq2 = massHp**2 - lambda3 * v0**2 / 2

        modelParameters["msq2"] = msq2

        modelParameters["lambda3"] = lambda3
        modelParameters["lambda4"] = lambda4
        modelParameters["lambda5"] = lambda5

        ## Some couplings are input parameters
        modelParameters["g1"] = inputParameters["g1"]
        modelParameters["g2"] = inputParameters["g2"]
        modelParameters["g3"] = inputParameters["g3"]
        modelParameters["lambda2"] = inputParameters["lambda2"]
        modelParameters["lambdaL"] = inputParameters["lambdaL"]

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


class EffectivePotentialIDM(EffectivePotentialNoResum):
    """
    Effective potential for the InertDoubletModel.

    This class inherits from the EffectivePotentialNoResum class and provides the
    necessary methods for calculating the effective potential.

    For this benchmark model we use the 4D potential without high-temperature expansion.
    """

    # ~ EffectivePotential interface
    fieldCount = 1
    """How many classical background fields"""

    effectivePotentialError = 1e-8
    """
    Relative accuracy at which the potential can be computed. Here it is set by the
    error tolerance of the thermal integrals Jf/Jb.
    """

    def __init__(self, owningModel: InertDoubletModel):
        """
        Initialize the EffectivePotentialIDM.
        """
        super().__init__(
            integrals=None,
            useDefaultInterpolation=True,
            imaginaryOption=EImaginaryOption.PRINCIPAL_PART,
        )

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        # Count particle degrees-of-freedom to facilitate inclusion of light particle
        # contributions to ideal gas pressure
        self.numBosonDof = 32
        self.numFermionDof = 90

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

        # For this benchmark we don't use the high-T approximation in the evaluation of
        # the one-loop thermal potential. We do use daisy-resummed masses.
        # The RG-scale in the CW-potential is given by the zero-temperature mass
        # of the relevant particle.

        # phi ~ 1/sqrt(2) (0, v)
        fields = Fields(fields)
        v = fields.getField(0)

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        # tree level potential
        potentialTree = 0.5 * msq * v**2 + 0.25 * lam * v**4

        # Particle masses and coefficients for the CW potential
        bosonInformation = self.bosonInformation(fields)
        fermionInformation = self.fermionInformation(fields)

        # Particle masses and coefficients for the one-loop thermal potential
        bosonInformationResummed = self.bosonInformationResummed(fields, temperature)

        potentialTotal = (
            potentialTree
            + self.constantTerms(temperature)
            + self.potentialOneLoop(bosonInformation, fermionInformation)
            + self.potentialOneLoopThermal(
                bosonInformationResummed, fermionInformation, temperature
            )
        )

        return np.array(potentialTotal)

    def jCW(
        self,
        massSq: np.ndarray,
        degreesOfFreedom: int | np.ndarray,
        c: float | np.ndarray,
        rgScale: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        One-loop Coleman-Weinberg contribution to the effective potential,
        as implemented in Jiang, Peng Huang, and Wang.

        Parameters
        ----------
        msq : array_like
            A list of the boson particle masses at each input point `X`.
        degreesOfFreedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c: float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        rgScale : float or array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Typically, one
            takes the same rgScale for all particles, but different scales
            for each particle are possible.

        Returns
        -------
        jCW : float or array_like
            One-loop Coleman-Weinberg potential for given particle spectrum.
        """

        # Note that we are taking the absolute value of the mass in the log here,
        # instead of using EImaginaryOption = ABS_ARGUMENT, because we do not 
        # want the absolute value in the product of massSq ans rgScale
        return degreesOfFreedom*np.array( 
            massSq * massSq * (np.log(np.abs(massSq / rgScale**2)) - c)
            + 2 * massSq * rgScale**2
        ) / (64 * np.pi * np.pi)

    def fermionInformation(self, fields: Fields) -> tuple[
        np.ndarray,
        float | np.ndarray,
        float | np.ndarray,
        float | np.ndarray,
    ]:
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
        mtsq = yt**2 * v**2 / 2 + 1e-100
        mtsq0T = yt**2 * self.modelParameters["v0"] ** 2 / 2

        massSq = np.stack((mtsq,), axis=-1)
        massSq0T = np.stack((mtsq0T,), axis=-1)
        degreesOfFreedom = np.array([12])

        return massSq, degreesOfFreedom, 3 / 2, np.sqrt(massSq0T)

    def bosonInformation(  # pylint: disable=too-many-locals
        self, fields: Fields
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes parameters for the one-loop potential (Coleman-Weinberg).

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

        v = fields.getField(0)
        v0 = self.modelParameters["v0"]

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        msq2 = self.modelParameters["msq2"]
        lam3 = self.modelParameters["lambda3"]
        lam4 = self.modelParameters["lambda4"]
        lam5 = self.modelParameters["lambda5"]

        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        # Scalar masses
        mhsq = msq + 3 * lam * v**2
        mHsq = msq2 + (lam3 + lam4 + lam5) / 2 * v**2
        mAsq = msq2 + (lam3 + lam4 - lam5) / 2 * v**2
        mHpmsq = msq2 + lam3 / 2 * v**2

        # Scalar masses at the zero-temperature vev (for RG-scale)
        mhsq0T = msq + 3 * lam * v0**2
        mHsq0T = msq2 + (lam3 + lam4 + lam5) / 2 * v0**2
        mAsq0T = msq2 + (lam3 + lam4 - lam5) / 2 * v0**2
        mHpmsq0T = msq2 + lam3 / 2 * v0**2

        # Gauge boson masses
        mWsq = g2**2 * v**2 / 4.0 + 1e-100
        mZsq = (g1**2 + g2**2) * v**2 / 4.0 + 1e-100

        # Gauge boson masses at the zero temperature vev (for RG-scale)
        mWsq0T = g2**2 * v0**2 / 4.0
        mZsq0T = (g1**2 + g2**2) * v0**2 / 4.0

        # W, Z, h, H, A, Hpm
        massSq = np.column_stack((mWsq, mZsq, mhsq, mHsq, mAsq, mHpmsq))
        massSq0 = np.column_stack((mWsq0T, mZsq0T, mhsq0T, mHsq0T, mAsq0T, mHpmsq0T))
        degreesOfFreedom = np.array([6, 3, 1, 1, 1, 2])
        c = 3 / 2 * np.ones(6)

        return massSq, degreesOfFreedom, c, np.sqrt(massSq0)

    def bosonInformationResummed(  # pylint: disable=too-many-locals
        self, fields: Fields, temperature: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        """
        Computes parameters for the thermal one-loop potential.

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

        v = fields.getField(0)

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        msq2 = self.modelParameters["msq2"]
        lam2 = self.modelParameters["lambda2"]
        lam3 = self.modelParameters["lambda3"]
        lam4 = self.modelParameters["lambda4"]
        lam5 = self.modelParameters["lambda5"]

        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]

        # Thermal masses of the scalars
        piPhi = (
            temperature**2
            / 12.0
            * (6 * lam + 2 * lam3 + lam4 + 3 / 4 * (3 * g2**2 + g1**2) + 3 * yt**2)
        )  # Eq. (15) of  2211.13142 (note the different normalization of lam)
        piEta = (
            temperature**2
            / 12.0
            * (6 * lam2 + 2 * lam3 + lam4 + 3 / 4 * (3 * g2**2 + g1**2))
        )  # Eq. (16) of 2211.13142 (note the different normalization of lam2)

        # Scalar masses including thermal contribution
        # Need to take the absolute value because we can not
        # use EImaginaryOption = ABS_ARGUMENT for the full potential
        mhsq = np.abs(msq + 3 * lam * v**2 + piPhi)
        mGsq = np.abs(msq + lam * v**2 + piPhi)  # Goldstone bosons
        mHsq = msq2 + (lam3 + lam4 + lam5) / 2 * v**2 + piEta
        mAsq = msq2 + (lam3 + lam4 - lam5) / 2 * v**2 + piEta
        mHpmsq = msq2 + lam3 / 2 * v**2 + piEta

        # Gauge boson masses, with thermal contribution to longitudinal W mass
        mWsq = g2**2 * v**2 / 4.0
        mWsqL = g2**2 * v**2 / 4.0 + 2 * g2**2 * temperature**2
        mZsq = (g1**2 + g2**2) * v**2 / 4.0

        # Eigenvalues of the Z&B-boson mass matrix
        piB = 2 * g1**2 * temperature**2
        piW = 2 * g2**2 * temperature**2
        m1sq = g1**2 * v**2 / 4
        m2sq = g2**2 * v**2 / 4
        m12sq = -g1 * g2 * v**2 / 4

        msqEig1 = (
            m1sq
            + m2sq
            + piB
            + piW
            - np.sqrt(4 * m12sq**2 + (m2sq - m1sq - piB + piW) ** 2)
        ) / 2
        msqEig2 = (
            m1sq
            + m2sq
            + piB
            + piW
            + np.sqrt(4 * m12sq**2 + (m2sq - m1sq - piB + piW) ** 2)
        ) / 2

        # HACK make sure the masses have the right shape
        if mWsq.shape != mWsqL.shape:
            mWsq = mWsq * np.ones(mWsqL.shape[0])
            mZsq = mZsq * np.ones(mWsqL.shape[0])

        # W, Wlong, Z,Zlong,photonLong, h, Goldstone H, A, Hpm
        massSq = np.column_stack(
            (mWsq, mWsqL, mZsq, msqEig1, msqEig2, mhsq, mGsq, mHsq, mAsq, mHpmsq)
        )
        degreesOfFreedom = np.array([4, 2, 2, 1, 1, 1, 3, 1, 1, 2])

        # As c and the RG-scale don't enter in the one-loop effective potential,
        # we just set them to 0
        return massSq, degreesOfFreedom, 0.0, 0.0

    def constantTerms(self, temperature: float | np.ndarray) -> float | np.ndarray:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in
        high-T expansion these are just (minus) the ideal gas pressure of light
        particles that were not integrated over in the one-loop part.

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
        dofsBoson = self.numBosonDof - 17
        dofsFermion = self.numFermionDof - 12  ## we only did top quark loops

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0


class InertDoubletModelExample(WallGoExampleBase):
    """
    Sets up the Inert doublet model, computes or loads the collison
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

    def getDefaultCollisionDirectory(self, momentumGridSize: int) -> pathlib.Path:
        """Returns the path to the directory with collisions"""
        return super().getDefaultCollisionDirectory(momentumGridSize)

    def initWallGoModel(self) -> "WallGo.GenericModel":
        """
        Initialize the model. This should run after cmdline argument parsing
        so safe to use them here.
        """
        return InertDoubletModel()

    def initCollisionModel(
        self, wallGoModel: "InertDoubletModel"
    ) -> "WallGoCollision.PhysicsModel":
        """Initialize the Collision model and set the seed."""

        import WallGoCollision  # pylint: disable = C0415

        # Collision integrations utilize Monte Carlo methods, so RNG is involved. We can
        # set the global seed for collision integrals as follows.
        # This is optional; by default the seed is 0.
        WallGoCollision.setSeed(0)

        # This example comes with a very explicit example function on how to setup and
        # configure the collision module. It is located in a separate module (same
        # directory) to avoid bloating this file. Import and use it here.
        from exampleCollisionDefs import (
            setupCollisionModel_QCDEW,
        )  # pylint: disable = C0415

        collisionModel = setupCollisionModel_QCDEW(
            wallGoModel.modelParameters,
        )

        return collisionModel

    def updateCollisionModel(
        self,
        inWallGoModel: "InertDoubletModel",
        inOutCollisionModel: "WallGoCollision.PhysicsModel",
    ) -> None:
        """Propagete changes in WallGo model to the collision model."""
        import WallGoCollision  # pylint: disable = C0415

        changedParams = WallGoCollision.ModelParameters()

        gs = inWallGoModel.modelParameters["g3"]  # names differ for historical reasons
        gw = inWallGoModel.modelParameters["g2"]  # names differ for historical reasons
        changedParams.addOrModifyParameter("gs", gs)
        changedParams.addOrModifyParameter("gw", gw)
        changedParams.addOrModifyParameter(
            "mq2", gs**2 / 6.0
        )  # quark thermal mass^2 in units of T
        changedParams.addOrModifyParameter(
            "mg2", 2.0 * gs**2
        )  # gluon thermal mass^2 in units of T
        changedParams.addOrModifyParameter(
            "mw2", 11.0 * gw**2 / 6.0
        )  # W boson thermal mass^2 in units of T

        inOutCollisionModel.updateParameters(changedParams)

    def configureCollisionIntegration(
        self, inOutCollisionTensor: "WallGoCollision.CollisionTensor"
    ) -> None:
        """Non-abstract override"""

        import WallGoCollision  # pylint: disable = C0415

        """Configure the integrator. Default settings should be reasonably OK
        so you can modify only what you need, or skip this step entirely. 
        Here we set everything manually to show how it's done.
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

        """We can also configure various verbosity settings that are useful when
        you want to see what is going on in long-running integrations. These include
        progress reporting and time estimates, as well as a full result dump of each
        individual integral to stdout. By default these are all disabled. Here we
        enable some for demonstration purposes.
        """
        verbosity = WallGoCollision.CollisionTensorVerbosity()
        verbosity.bPrintElapsedTime = (
            True  # report total time when finished with all integrals
        )

        """Progress report when this percentage of total integrals (approximately
        have been computed. Note that this percentage is per-particle-pair, ie. 
        each (particle1, particle2) pair reports when this percentage of their own 
        integrals is done. Note also that in multithreaded runs the progress tracking
        is less precise.
        """
        verbosity.progressReportPercentage = 0.25

        # Print every integral result to stdout? This is very slow and verbose,
        # intended only for debugging purposes
        verbosity.bPrintEveryElement = False

        inOutCollisionTensor.setIntegrationVerbosity(verbosity)

    def configureManager(self, inOutManager: "WallGo.WallGoManager") -> None:
        inOutManager.config.loadConfigFromFile(
            pathlib.Path(self.exampleBaseDirectory / "inertDoubletModelConfig.ini")
        )
        super().configureManager(inOutManager)

    def updateModelParameters(
        self, model: "InertDoubletModel", inputParameters: dict[str, float]
    ) -> None:
        """Convert Inert Doublet Model inputs to Lagrangian params and update
        internal model parameters. This example is constructed so that the
        effective potential and particle mass functions refer to model.modelParameters,
        so be careful not to replace that reference here.
        """
        oldParams = model.modelParameters  # pylint: disable = W0612
        model.updateModel(inputParameters)
        newParams = model.modelParameters  # pylint: disable = W0612

        """Collisions integrals for this example depend on the QCD and Electroweak
        coupling, if it changes we must recompute collisions before running the
        wall solver. The bool flag here is inherited from WallGoExampleBase and 
        checked in runExample(). But since we want to keep the example simple, we 
        skip this check and assume the existing data is OK.
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
        wall solver for the inert doublet model benchmark point.
        """
        output: list[ExampleInputPoint] = []

        output.append(
            ExampleInputPoint(
                {
                    "v0": 246.22,
                    "Mt": 172.76,
                    "g1": 0.35,
                    "g2": 0.65,
                    "g3": 1.2279920495357861,
                    "lambda2": 0.1,
                    "lambdaL": 0.0015,
                    "mh": 125.0,
                    "mH": 62.66,
                    "mA": 300.0,
                    "mHp": 300.0,
                    # We don't use mHm as input parameter, as it is equal to mHp
                },
                WallGo.PhaseInfo(
                    temperature=117.1,
                    phaseLocation1=WallGo.Fields([0.0]),
                    phaseLocation2=WallGo.Fields([246.0]),
                ),
                WallGo.VeffDerivativeSettings(temperatureVariationScale=0.5, fieldValueVariationScale=[10.0]),
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

    example = InertDoubletModelExample()
    example.runExample()
