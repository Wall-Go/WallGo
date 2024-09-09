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

import os
import pathlib
import sys
import numpy as np

# WallGo imports
import WallGo  # Whole package, in particular we get WallGo.initialize()
from WallGo import Fields, GenericModel, Particle, WallGoManager

# Adding the Models folder to the path and import effectivePotentialNoResum
modelsPath = pathlib.Path(__file__).parents[1]
sys.path.insert(0, str(modelsPath))
from effectivePotentialNoResum import (  # pylint: disable=C0411, C0413, E0401
    EffectivePotentialNoResum,
)

# Inert doublet model, as implemented in 2211.13142
class InertDoubletModel(GenericModel):
    r"""
    Inert doublet model.

    The tree-level potential is given by
    V = msq |phi|^2 + msq2 |eta|^2 + lambda |phi|^4 + lambda2 |eta|^4
        + lambda3 |phi|^2 |eta|^2 + lambda4 |phi^dagger eta|^2
        + (lambda5 (phi^dagger eta)^2 +h.c.)
    Note that there are some differences in normalization compared to Jiang, Peng Huang, and Wang

    Only the Higgs field undergoes the phase transition, the new scalars only 
    modify the effective potential.

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    particles: list[Particle] = []
    outOfEquilibriumParticles: list[Particle] = []
    modelParameters: dict[str, float] = {}
    collisionParameters: dict[str, float] = {}

    ## Specifying this is REQUIRED
    fieldCount = 1

    def __init__(self, initialInputParameters: dict[str, float]):
        """
        Initialize the InertDoubletModel.

        Parameters
        ----------
        initialInputParameters: dict[str, float]
            A dictionary of initial input parameters for the model.

        Returns
        ----------
        cls: InertDoubletModel
            An object of the InertDoubletModel class.
        
        """

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict.
        self.effectivePotential = EffectivePotentialIDM(
            self.modelParameters, self.fieldCount
        )

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

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
            return 0.5 * self.modelParameters["yt"]**2 * fields.GetField(0)**2
    
        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"] ** 2 * fields.GetField(0)
        
        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuarkL = Particle(
            "topL",
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=6,
        )
        self.addParticle(topQuarkL)

        topQuarkR = Particle(
            "topR",
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=6,
        )
        self.addParticle(topQuarkR)

        ## === SU(2) gauge boson ===
        def WMsqVacuum(fields: Fields) -> Fields:
            return self.modelParameters["g2"]**2*fields.GetField(0)**2/4
        def WMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["g2"]**2*fields.GetField(0)/2
        def WMsqThermal(T: float) -> float:
            return self.modelParameters["g2"] ** 2 * T**2 * 11.0 / 6.0

        W = Particle(
            "W",
            msqVacuum=WMsqVacuum,
            msqDerivative=WMsqDerivative,
            msqThermal=WMsqThermal,
            statistics="Boson",
            inEquilibrium=False,
            ultrarelativistic=True,
            totalDOFs=9,
        )
        self.addParticle(W)

        ## === SU(3) gluon ===
        def gluonMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 * 2.0

        gluon = Particle(
            "gluon",
            msqVacuum=0.0,
            msqDerivative=0.0,
            msqThermal=gluonMsqThermal,
            statistics="Boson",
            inEquilibrium=True,
            ultrarelativistic=True,
            totalDOFs=16,
        )
        self.addParticle(gluon)

        ## === Light quarks, 5 of them ===
        def lightQuarkMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        lightQuark = Particle(
            "lightQuark",
            msqVacuum= 0.0,
            msqDerivative= 0.0,
            msqThermal=lightQuarkMsqThermal,
            statistics="Fermion",
            inEquilibrium=True,
            ultrarelativistic=True,
            totalDOFs=60,
        )
        self.addParticle(lightQuark)

    ## Go from whatever input params --> action params
    def calculateModelParameters(
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
        super().calculateModelParameters(inputParameters)

        modelParameters = {}

        v0 = inputParameters["v0"]
        modelParameters["v0"] = v0

        # Higgs parameters
        mh = inputParameters["mh"]
        modelParameters["lambda"] = 0.5 * mh**2 / v0**2
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2

        ## Then the Yukawa sector
        Mt = inputParameters["Mt"]
        modelParameters["yt"] = np.sqrt(2.0) * Mt / v0

        ## Then the inert doublet parameters
        mH = inputParameters["mH"]
        mA = inputParameters["mA"]
        mHp = inputParameters["mHp"]

        lambda5 = (mH**2 - mA**2) / v0**2
        lambda4 = -2 * (mHp**2 - mA**2) / v0**2 + lambda5
        lambda3 = 2 * inputParameters["lambdaL"] - lambda4 - lambda5
        msq2 = mHp**2 - lambda3 * v0**2 / 2

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

# end model

class EffectivePotentialIDM(EffectivePotentialNoResum):
    """
    Effective potential for the InertDoubletModel.

    This class inherits from the EffectivePotentialNoResum class and provides the
    necessary methods for calculating the effective potential.

    For this benchmark model we use the 4D potential without high-temperature expansion.
    """

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        """
        Initialize the EffectivePotentialIDM.

        Parameters
        ----------
        modelParameters: dict[str, float]
            A dictionary of model parameters.
        fieldCount: int
            The number of fields undergoing the phase transition

        Returns
        ----------
        cls: EffectivePotentialIDM
            an object of the EffectivePotentialIDM class
        """
        super().__init__(
            modelParameters, fieldCount, integrals=None, useDefaultInterpolation=True
        )
        # The super call already gave us the model params

        # Count particle degrees-of-freedom to facilitate inclusion of light particle contributions
        # to ideal gas pressure
        self.num_boson_dof = 32
        self.num_fermion_dof = 90


    def evaluate(
        self, fields: Fields, temperature: float, checkForImaginary: bool = False
    ) -> complex | np.ndarray:
        """
        Evaluate the effective potential.

        Parameters
        ----------
        fields: Fields
            The field configuration
        temperature: float
            The temperature
        checkForImaginary: bool
            Setting to check for imaginary parts of the potential

        Returns
        ----------
        potentialTotal: complex | np.ndarray
            The value of the effective potential
        """

        # For this benchmark we don't use the high-T approximation in the evaluation of
        # the one-loop thermal potential. We do use daisy-resummed masses. The RG-scale
        # in the CW-potential is given by the zero-temperature mass of the relevant particle.

        # phi ~ 1/sqrt(2) (0, v)
        fields = Fields(fields)
        v = fields.GetField(0)

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]

        # tree level potential
        potentialTree = 0.5 * msq * v**2 + 0.25 * lam * v**4

        # Particle masses and coefficients for the CW potential
        bosonStuff = self.bosonStuff(fields)
        fermionStuff = self.fermionStuff(fields)

        # Particle masses and coefficients for the one-loop thermal potential
        bosonStuffResummed = self.bosonStuffResummed(fields, temperature)

        potentialTotal = (
            potentialTree
            + self.constantTerms(temperature)
            + self.potentialOneLoop(bosonStuff, fermionStuff, checkForImaginary)
            + self.potentialOneLoopThermal(
                bosonStuffResummed, fermionStuff, temperature, checkForImaginary
            )
        )

        return potentialTotal

    def jCW(self, massSq: np.ndarray, degreesOfFreedom: int | np.ndarray, c: float| np.ndarray, rgScale: float| np.ndarray
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
        
        return degreesOfFreedom * (
            massSq * massSq * (np.log(np.abs(massSq / rgScale**2) + 1e-100) - c)
            + 2 * massSq * rgScale**2
        )

    def fermionStuff(
        self, fields: Fields
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # TODO: fix return type inheritance error
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

        v = fields.GetField(0)

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2 + 1e-100
        mtsq0T = yt**2 * self.modelParameters["v0"] ** 2 / 2

        massSq = np.stack((mtsq,), axis=-1)
        massSq0T = np.stack((mtsq0T,), axis=-1)
        degreesOfFreedom = np.array([12])

        return massSq, degreesOfFreedom, 3 / 2, np.sqrt(massSq0T)

    def bosonStuff(  # pylint: disable=too-many-locals
        self, fields: Fields
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # TODO: fix return type inheritance error
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
        
        v = fields.GetField(0)
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

    def bosonStuffResummed(  # pylint: disable=too-many-locals
        self, fields: Fields, temperature: float | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # TODO: fix return type inheritance error
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

        v = fields.GetField(0)

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

        # As c and the RG-scale don't enter in the one-loop effective potential, we just set them to 0
        return massSq, degreesOfFreedom, 0, 0

    def constantTerms(self, temperature: np.ndarray) -> np.ndarray:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        
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
        dofsBoson = self.num_boson_dof - 17
        dofsFermion = self.num_fermion_dof - 12  ## we only did top quark loops

        # Fermions contribute with a magic 7/8 prefactor as usual. Overall minus
        # sign since Veff(min) = -pressure
        return -(dofsBoson + 7.0 / 8.0 * dofsFermion) * np.pi**2 * temperature**4 / 90.0


def main() -> None:

    WallGo.initialize()

    ## Modify the config, we use N=5 for this example
    WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "5")

    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    # Guess of the wall thickness: (approximately) 5/Tn
    wallThicknessIni = 0.05

    # Estimate of the mean free path of the particles in the plasma: (approximately) 100/Tn
    meanFreePath = 1.0

    # Create WallGo control object
    # The following 2 parameters are used to estimate the optimal value of dT used
    # for the finite difference derivatives of the potential.
    # Temperature scale (in GeV) over which the potential changes by O(1).
    # A good value would be of order Tc-Tn.
    temperatureScale = 1.0
    # Field scale (in GeV) over which the potential changes by O(1). A good value
    # would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the
    # same scale, or an array.
    fieldScale = 10.0
    manager = WallGoManager(
        wallThicknessIni, meanFreePath, temperatureScale, fieldScale
    )

    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input,
    but this is likely to change in the future
    """

    ## QFT model input.
    inputParameters = {
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
        "mHp": 300.0,  # We don't use mHm as input parameter, as it is equal to mHp
    }

    model = InertDoubletModel(inputParameters)

    # ---- collision integration and path specifications

    # Directory name for collisions integrals defaults to "CollisionOutput/"
    # these can be loaded or generated given the flag "generateCollisionIntegrals"
    WallGo.config.config.set("Collisions", "pathName", "CollisionOutput/")

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    # Generates or reads collision integrals
    manager.generateCollisionFiles()

    print("\n=== WallGo parameter scan ===")
    # ---- This is where you'd start an input parameter
    # loop if doing parameter-space scans ----

    """ Example mass loop that just does one value of the Higgs mass.
    Note that the WallGoManager class is NOT thread safe internally, so it 
    is NOT safe to parallelize this loop eg. with OpenMP. We recommend 
    ``embarrassingly parallel`` runs for large-scale parameter scans. 
    """
    valuesmH = [62.66]

    for mH in valuesmH:

        inputParameters["mH"] = mH

        """In addition to model parameters, WallGo needs info about the phases at the
        nucleation temperature. Use the WallGo.PhaseInfo dataclass for this purpose.
        The transition goes from phase1 to phase2.
        """

        Tn = 117.1  # nucleation temperature

        phaseInfo = WallGo.PhaseInfo(
            temperature=Tn,
            phaseLocation1=WallGo.Fields([0.0]),
            phaseLocation2=WallGo.Fields([246.0]),
        )

        """Give the input to WallGo. It is NOT enough to change parameters directly
           in the GenericModel instance because
            1) WallGo needs the PhaseInfo 
            2) WallGoManager.setParameters() does parameter-specific initializations
               of internal classes
        """

        manager.setParameters(phaseInfo)

        """WallGo can now be used to compute wall stuff!"""

        # ---- Solve wall speed in Local Thermal Equilibrium approximation

        vwLTE = manager.wallSpeedLTE()

        print(f"LTE wall speed:    {vwLTE:.6f}")

        # ---- Solve field EOM. For illustration, first solve it without any
        # out-of-equilibrium contributions. The resulting wall speed should
        # be close to the LTE result

        bIncludeOffEq = False
        print(f"=== Begin EOM with {bIncludeOffEq=} ===")

        results = manager.solveWall(bIncludeOffEq)

        print("\n=== Local equilibrium results ===")
        print(f"wallVelocity:      {results.wallVelocity:.6f}")
        print(f"wallVelocityError: {results.wallVelocityError:.6f}")
        print(f"wallWidths:        {results.wallWidths}")

        # Repeat with out-of-equilibrium parts included. This requires
        # solving Boltzmann equations, invoked automatically by solveWall()
        bIncludeOffEq = True
        print(f"\n=== Begin EOM with {bIncludeOffEq = } ===")

        results = manager.solveWall(bIncludeOffEq)

        print("\n=== Out-of-equilibrium results ===")
        print(f"wallVelocity:      {results.wallVelocity:.6f}")
        print(f"wallVelocityError: {results.wallVelocityError:.6f}")
        print(f"wallWidths:        {results.wallWidths}")


    # end parameter-space loop


# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
