import pytest
from dataclasses import dataclass
import numpy as np
from scipy.integrate import odeint
import WallGo

class QuarticZ2(WallGo.GenericModel):
    r"""
    Z2 symmetric quartic potential.

    The potential is given by:
    V = 1/2 muSq |phi|^2 + 1/4 lam |phi|^4

    This class inherits from the GenericModel class and implements the necessary
    methods for the WallGo package.
    """

    def __init__(self, modelParameters: dict[str, float]):
        """
        Initialize the QuarticZ2 model.
        """

        self.modelParameters = modelParameters

        # Initialize internal effective potential
        self.effectivePotential = EffectivePotentialQuarticZ2(self)

        # Create a list of particles relevant for the Boltzmann equations
        self.defineParticles()

    # ~ GenericModel interface
    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def getEffectivePotential(self) -> "EffectivePotentialQuarticZ2":
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
        def topMsqVacuum(fields: WallGo.Fields) -> WallGo.Fields:
            return 0.5 * fields.getField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: WallGo.Fields) -> WallGo.Fields:
            return np.transpose(
                [fields.getField(0)]
            )

        topQuark = WallGo.Particle(
            "top",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            statistics="Fermion",
            totalDOFs=12,
        )
        self.addParticle(topQuark)


# end model


class EffectivePotentialQuarticZ2(WallGo.EffectivePotential):
    """
    Effective potential for the QuarticZ2 model.

    This class inherits from the EffectivePotential class and provides the
    necessary methods for calculating the effective potential.
    """

    # ~ EffectivePotential interface
    fieldCount = 1
    """How many classical background fields"""

    effectivePotentialError = 1e-15
    """
    Relative accuracy at which the potential can be computed. 
    """

    def __init__(self, owningModel: QuarticZ2) -> None:
        """
        Initialize the EffectivePotentialQuarticZ2.
        """

        assert owningModel is not None, "Invalid model passed to Veff"

        self.owner = owningModel
        self.modelParameters = self.owner.modelParameters

        # Count particle degrees-of-freedom to facilitate inclusion of
        # light particle contributions to ideal gas pressure
        self.numBosonDof = 29
        self.numFermionDof = 90

    def evaluate(
        self, fields: WallGo.Fields, temperature: float
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

        # phi ~ 1/sqrt(2) (0, v), S ~ x
        fields = WallGo.Fields(fields)
        v = fields.getField(0)

        muSq = self.modelParameters["muSq"]
        lam = self.modelParameters["lam"]
        # We include a small cubic term to make the second minimum slightly deeper.
        # Otherwise WallGo would not converge.
        smallCubic = -1e-4

        # tree level potential
        potentialTree = (
            0.5 * muSq * v**2
            + 0.25 * lam * v**4
            + smallCubic * v**3 / 3
        )

        # Include only the T^4 terms in the potential
        potentialTotal = (
            potentialTree
            + self.constantTerms(temperature)
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

    
@pytest.mark.parametrize(
    "muSq, lam",
    [
         (-50**2, 0.05),
         (-50**2, 0.5),
         (-100**2, 0.05),
         (-100**2, 0.5),
     ]
)
def test_EOMSolver(muSq, lam):
    """
    Compare the width of the wall computed by WallGo without out-of-equilibrium effects
    to the exact result that solves the EOM in the Z2-symmetric quartic potential.

    Parameters
    ----------
    muSq : float
        Mass term. Must be negative.
    lam : float
        Quartic coupling.

    Returns
    -------
    None.

    """
    
    # Exact VEV
    vev = np.sqrt(abs(muSq)/lam)
    # Exact wall width
    trueL = np.sqrt(2/abs(muSq))
    
    # Initialise WallGo. Many parameters are now unimportant because we don't solve the
    # Boltzmann equation
    
    Tn = 100
    
    temperatureScale = Tn / 100
    fieldScale = vev
    derivSettings = WallGo.VeffDerivativeSettings(temperatureVariationScale=float(temperatureScale), fieldValueVariationScale=fieldScale)
    
    modelParameters = {'muSq': muSq, 'lam': lam}
    model = QuarticZ2(modelParameters)
    model.getEffectivePotential().configureDerivatives(derivSettings)
    
    phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                            phaseLocation1 = WallGo.Fields( [-vev]),
                                            phaseLocation2 = WallGo.Fields( [vev] ))
    
    thermodynamics = WallGo.Thermodynamics(
        model.getEffectivePotential(),
        Tn,
        phaseInfo.phaseLocation2,
        phaseInfo.phaseLocation1,
    )
    thermodynamics.freeEnergyHigh.newInterpolationTable(Tn/10, 10*Tn, 100)
    thermodynamics.freeEnergyHigh.setExtrapolationType(WallGo.EExtrapolationType.CONSTANT, WallGo.EExtrapolationType.CONSTANT)
    thermodynamics.freeEnergyLow.newInterpolationTable(Tn/10, 10*Tn, 100)
    thermodynamics.freeEnergyLow.setExtrapolationType(WallGo.EExtrapolationType.CONSTANT, WallGo.EExtrapolationType.CONSTANT)
    
    # Hack to avoid complains from Hydrodynamics
    thermodynamics.TMaxHighT = Tn-1e-6
    thermodynamics.TMaxLowT = Tn-1e-6
    thermodynamics.TMinHighT = Tn-1e-6
    thermodynamics.TMinLowT = Tn-1e-6
    thermodynamics.muMinHighT = 4
    thermodynamics.aMinHighT = 1
    thermodynamics.epsilonMinHighT = 1.5e-3*Tn**4
    thermodynamics.muMaxHighT = 4
    thermodynamics.aMaxHighT = 1
    thermodynamics.epsilonMaxHighT = 1.5e-3*Tn**4
    thermodynamics.muMinLowT = 4
    thermodynamics.aMinLowT = 1-3e-3
    thermodynamics.epsilonMinLowT = 0.0
    thermodynamics.muMaxLowT = 4
    thermodynamics.aMaxLowT = 1-3e-3
    thermodynamics.epsilonMaxLowT = 0.0
    
    hydrodynamics = WallGo.Hydrodynamics(thermodynamics, 10, 0.1, rtol=1e-6, atol=1e-8)
    grid = WallGo.Grid3Scales(
        50,
        11,
        3*trueL,
        3*trueL,
        trueL,
        Tn,
        0.75,
        0.1,
    )
    boltzmannSolver = WallGo.BoltzmannSolver(
        grid,
        basisM="Cardinal",
        basisN="Chebyshev",
        collisionMultiplier=1,
        )
    
    eom = WallGo.EOM(
        boltzmannSolver,
        thermodynamics,
        hydrodynamics,
        grid,
        1,
        3*trueL,
        [0.1, 100.0],
        [-10,10],
        includeOffEq=False,
        forceEnergyConservation=False,
        forceImproveConvergence=False,
        errTol=1e-5,
        maxIterations=20,
        pressRelErrTol=0.01,
    )
    
    # Start with a wrong width to see if WallGo can find the correct one.
    wallParams = WallGo.WallParams(widths=np.array([1.5*trueL]), offsets=np.array([0]))
    # Computes the true wallParams. The wall velocity is irrelevant here.
    (
        pressure,
        wallParams,
        boltzmannResults,
        boltzmannBackground,
        hydroResults,
        *_
    ) = eom.wallPressure(0.3, wallParams)
    
    tanhError = eom.estimateTanhError(wallParams, boltzmannResults, boltzmannBackground, hydroResults)[0]
    
    # Compare the exact and WallGo wall widths
    np.testing.assert_allclose(trueL, wallParams.widths[0], atol=0, rtol=1e-4)
    # Make sure that the tanh error is small
    np.testing.assert_allclose(tanhError, 0, atol=1e-8)