import numpy as np
import numpy.typing as npt

# WallGo imports
import WallGo
from .collisionWrapper import Collision
from .boltzmann import BoltzmannSolver
from .containers import PhaseInfo
from .EffectivePotential import EffectivePotential
from .equationOfMotion import EOM
from .exceptions import WallGoError, WallGoPhaseValidationError
from .genericModel import GenericModel
from .grid3Scales import Grid3Scales
from .hydrodynamics import Hydrodynamics
from .hydrodynamicsTemplateModel import HydrodynamicsTemplateModel
from .Integrals import Integrals
from .Thermodynamics import Thermodynamics
from .results import WallGoResults
from .WallGoUtils import getSafePathToResource

class WallGoManager:
    """Defines a 'control' class for managing the program flow.
    This should be better than writing the same stuff in every example main
    function, and is good for hiding some of our internal implementation
    details from the user.
    """

    def __init__(
        self,
        wallThicknessIni: float,
        meanFreePath: float,
        temperatureScaleInput: float,
        fieldScaleInput: npt.ArrayLike,
    ):
        """do common model-independent setup here"""

        # TODO cleanup, should not read the config here if we have a global WallGo config object
        # self.config = Config()
        # self.config.readINI( getSafePathToResource("Config/WallGoDefaults.ini") )

        self.config = WallGo.config

        # self.integrals = Integrals()

        # self._initalizeIntegralInterpolations(self.integrals)

        # -- Order of initialization matters here

        # Grid
        self._initGrid(
            wallThicknessIni,
            meanFreePath,
        )

        self._initBoltzmann()
        self.temperatureScaleInput = temperatureScaleInput
        self.fieldScaleInput = fieldScaleInput
        self.collision: Collision = None
        self.model: GenericModel = None

    def _initModel(self, model: GenericModel) -> None:
        """Initializes the model."""
        self.model = model

    def _initCollision(self, model: GenericModel) -> None:
        """Initializes the collision module.
            Creates Collision singleton which automatically loads the collision module
            Use help(Collision.manager) for info about what functionality is available

        Args:
            model (GenericModel): The model to be used for collision detection.

        Returns:
            None
        """
        self.collision = Collision(model)

    def registerModel(self, model: GenericModel) -> None:
        """Register a physics model with WallGo."""
        assert isinstance(model, GenericModel)
        self._initModel(model)
        self._initCollision(model)

        potentialError = self.config.getfloat("EffectivePotential", "potentialError")

        self.model.effectivePotential.setPotentialError(potentialError)
        self.model.effectivePotential.setScales(
            float(self.temperatureScaleInput), self.fieldScaleInput
        )

        # Update Boltzmann off-eq particle list to match that defined in model
        self.boltzmannSolver.updateParticleList(model.outOfEquilibriumParticles)

    
    # Name of this function does not really describe what it does (it also calls the function that finds the temperature range)
    def setParameters(self, phaseInput: PhaseInfo) -> None:
        """Parameters
        ----------
        modelParameters: dict[str, float]
                        Dict containing all QFT model parameters:
                        Those that enter the action and the renormalization scale.
        phaseInput: WallGo.PhaseInfo
                    Should contain approximate field values at the two phases that WallGo will analyze,
                    and the nucleation temperature. Transition is assumed to go phaseLocation1 --> phaseLocation2.
        """

        # self.model.modelParameters = modelParameters

        # Checks that phase input makes sense with the user-specified Veff
        self.validatePhaseInput(phaseInput)

        # Change the falloff scale in grid now that we have a good guess for
        # the plasma temperature
        self.grid.changeMomentumFalloffScale(phaseInput.temperature)

        self.initTemperatureRange()

        print("Temperature ranges:")
        print(
            f"High-T phase: TMin = {self.thermodynamics.freeEnergyHigh.minPossibleTemperature}, "
            f"TMax = {self.thermodynamics.freeEnergyHigh.maxPossibleTemperature}"
        )
        print(
            f"Low-T phase: TMin = {self.thermodynamics.freeEnergyLow.minPossibleTemperature}, "
            f"TMax = {self.thermodynamics.freeEnergyLow.maxPossibleTemperature}"
        )

        # LN: Giving sensible temperature ranges to Hydro seems to be very important.
        # I propose hydro routines be changed so that we have easy control over what temperatures are used
        self._initHydrodynamics(self.thermodynamics)
        self._initEOM()

        if not np.isfinite(self.hydrodynamics.vJ) or self.hydrodynamics.vJ > 1 or self.hydrodynamics.vJ < 0:
            raise WallGoError(
                "Failed to solve Jouguet velocity at input temperature!",
                data={
                    "vJ": self.hydrodynamics.vJ,
                    "temperature": phaseInput.temperature,
                    "TMin": self.TMin,
                    "TMax": self.TMax,
                },
            )

        print(f"Jouguet: {self.hydrodynamics.vJ}")

    #  print(f"Matching at the Jouguet velocity {self.hydro.findMatching(0.99*self.hydro.vJ)}")

    def changeInputParameters(
        self, inputParameters: dict[str, float], effectivePotential: EffectivePotential
    ) -> None:
        """Recomputes the model parameters when the user provides new inputparameters.
        Also updates the effectivePotential correspondingly.
        """
        self.model.modelParameters = self.model.calculateModelParameters(
            inputParameters
        )
        self.model.effectivePotential = effectivePotential(
            self.model.modelParameters, self.model.fieldCount
        )

        potentialError = self.config.getfloat("EffectivePotential", "potentialError")

        self.model.effectivePotential.setPotentialError(potentialError)
        self.model.effectivePotential.setScales(
            float(self.temperatureScaleInput), self.fieldScaleInput
        )

    def validatePhaseInput(self, phaseInput: PhaseInfo) -> None:
        """This checks that the user-specified phases are OK.
        Specifically, the effective potential should have two minima at the given T,
        otherwise phase transition analysis is not possible.
        """

        T = phaseInput.temperature

        # Find the actual minima at input T, should be close to the user-specified locations
        phaseLocation1, VeffValue1 = self.model.effectivePotential.findLocalMinimum(
            phaseInput.phaseLocation1, T
        )
        phaseLocation2, VeffValue2 = self.model.effectivePotential.findLocalMinimum(
            phaseInput.phaseLocation2, T
        )

        print(f"Found phase 1: phi = {phaseLocation1}, Veff(phi) = {VeffValue1}")
        print(f"Found phase 2: phi = {phaseLocation2}, Veff(phi) = {VeffValue2}")

        if np.allclose(phaseLocation1, phaseLocation2, rtol=1e-05, atol=1e-05):
            raise WallGoPhaseValidationError(
                "It looks like both phases are the same, this will not work",
                phaseInput,
                {
                    "phaseLocation1": phaseLocation1,
                    "Veff(phi1)": VeffValue1,
                    "phaseLocation2": phaseLocation2,
                    "Veff(phi2)": VeffValue2,
                },
            )

        ## Currently we assume transition phase1 -> phase2. This assumption shows up at least when initializing FreeEnergy objects
        if np.real(VeffValue1) < np.real(VeffValue2):
            raise WallGoPhaseValidationError(
                "Phase 1 has lower free energy than Phase 2, this will not work",
                phaseInput,
                {
                    "phaseLocation1": phaseLocation1,
                    "Veff(phi1)": VeffValue1,
                    "phaseLocation2": phaseLocation2,
                    "Veff(phi2)": VeffValue2,
                },
            )

        foundPhaseInfo = PhaseInfo(
            temperature=T, phaseLocation1=phaseLocation1, phaseLocation2=phaseLocation2
        )

        self.phasesAtTn = foundPhaseInfo

    def initTemperatureRange(self) -> None:
        """Get initial guess for the relevant temperature range and store in internal TMin, TMax"""

        # LN: this routine is probably too heavy. We could at least drop the
        # Tc part, or find it after FreeEnergy interpolations are done

        assert self.phasesAtTn != None

        Tn = self.phasesAtTn.temperature

        self.thermodynamics = Thermodynamics(
            self.model.effectivePotential,
            Tn,
            self.phasesAtTn.phaseLocation2,
            self.phasesAtTn.phaseLocation1,
        )

        # Let's turn these off so that things are more transparent
        self.thermodynamics.freeEnergyHigh.disableAdaptiveInterpolation()
        self.thermodynamics.freeEnergyLow.disableAdaptiveInterpolation()

        # Use the template model to find an estimate of the minimum and
        # maximum required temperatures

        try:
            ## ---- Use the template model to find an estimate of the minimum and maximum required temperature
            hydrodynamicsTemplate = HydrodynamicsTemplateModel(self.thermodynamics)
            print(f"vwLTE in the template model: {hydrodynamicsTemplate.findvwLTE()}")

        except WallGoError as error:
            # Throw new error with more info
            raise WallGoPhaseValidationError(error.message, self.phasesAtTn, error.data)
            
        # Raise an error if this is an inverse PT (if epsilon is negative)
        if hydrodynamicsTemplate.epsilon < 0:
            raise WallGoError(
                "WallGo cannot treat inverse PTs. epsilon must be positive.")

        _, _, THighTMaxTemplate, TLowTMaxTemplate = hydrodynamicsTemplate.findMatching(
            0.99 * hydrodynamicsTemplate.vJ
        )
        
        if THighTMaxTemplate is None:
            THighTMaxTemplate = self.config.getfloat("Hydrodynamics", "tmax")*Tn
        if TLowTMaxTemplate is None:
            TLowTMaxTemplate = self.config.getfloat("Hydrodynamics", "tmax")*Tn

        phaseTracerTol = self.config.getfloat("EffectivePotential", "phaseTracerTol")

        # Estimate of the dT needed to reach the desired tolerance considering
        # the error of a cubic spline scales like dT**4.
        dT = self.model.effectivePotential.temperatureScale * phaseTracerTol**0.25

        """If TMax, TMin are too close to real temperature boundaries
        the program can slow down significantly, but TMax must be large
        enough, and the template model only provides an estimate.
        HACK! fudgeFactor, see issue #145 """
        fudgeFactor = 1.2  # should be bigger than 1, but not know a priori
        TMinHighT, TMaxHighT = 0, max(2*Tn, fudgeFactor * THighTMaxTemplate)
        TMinLowT, TMaxLowT = 0, max(2*Tn, fudgeFactor * TLowTMaxTemplate)

        # Interpolate phases and check that they remain stable in this range
        fHighT = self.thermodynamics.freeEnergyHigh
        fLowT = self.thermodynamics.freeEnergyLow

        fHighT.tracePhase(TMinHighT, TMaxHighT, dT, phaseTracerTol)
        fLowT.tracePhase(TMinLowT, TMaxLowT, dT, phaseTracerTol)

    def _initHydrodynamics(self, thermodynamics: Thermodynamics) -> None:
        """"""
        tmax = self.config.getfloat("Hydrodynamics", "tmax")
        tmin = self.config.getfloat("Hydrodynamics", "tmin")
        rtol = self.config.getfloat("Hydrodynamics", "relativeTol")
        atol = self.config.getfloat("Hydrodynamics", "absoluteTol")
        self.hydrodynamics = Hydrodynamics(thermodynamics, tmax, tmin, rtol, atol)

    def _initGrid(self, wallThicknessIni: float, meanFreePath: float) -> None:
        r"""
        Parameters
        ----------
        M : int
            Number of basis functions in the :math:`\xi` (and :math:`\chi`)
            direction.
        N : int
            Number of basis functions in the :math:`p_z` and :math:`p_\Vert`
            (and :math:`\rho_z` and :math:`\rho_\Vert`) directions.
            This number has to be odd
        L_xi : float
            Length scale determining transform in the xi direction.
        """

        # To initialize Grid we need to specify a "temperature" scale that has
        # analogous role as L_xi, but for the momenta. In practice this scale
        # needs to be close to temperature near the wall, but we don't know
        # that yet, so just initialize with some value here and update once the
        # nucleation temperature is obtained.
        initialMomentumFalloffScale = 50.0

        N = self.config.getint("PolynomialGrid", "momentumGridSize")
        M = self.config.getint("PolynomialGrid", "spatialGridSize")
        ratioPointsWall = self.config.getfloat("PolynomialGrid", "ratioPointsWall")
        smoothing = self.config.getfloat("PolynomialGrid", "smoothing")
        self.meanFreePath = meanFreePath

        tailLength = max(
            meanFreePath, wallThicknessIni * (1 + 3 * smoothing) / ratioPointsWall
        )

        if N % 2 == 0:
            raise ValueError(
                "You have chosen an even number N of momentum-grid points. "
                "WallGo only works with odd N, please change it to an odd number."
            )
        self.grid = Grid3Scales(
            M,
            N,
            tailLength,
            tailLength,
            wallThicknessIni,
            initialMomentumFalloffScale,
            ratioPointsWall,
            smoothing,
        )

    def _initBoltzmann(self) -> None:
        # Hardcode basis types here: Cardinal for z, Chebyshev for pz, pp
        self.boltzmannSolver = BoltzmannSolver(
            self.grid, basisM="Cardinal", basisN="Chebyshev"
        )

    def _initEOM(self) -> None:
        numberOfFields = self.model.fieldCount

        errTol = self.config.getfloat("EOM", "errTol")
        maxIterations = self.config.getint("EOM", "maxIterations")
        pressRelErrTol = self.config.getfloat("EOM", "pressRelErrTol")
        
        wallThicknessBounds = (self.config.getfloat("EOM", "wallThicknessLowerBound"),
                               self.config.getfloat("EOM", "wallThicknessUpperBound"))
        wallOffsetBounds = (self.config.getfloat("EOM", "wallOffsetLowerBound"),
                               self.config.getfloat("EOM", "wallOffsetUpperBound"))

        self.eom = EOM(
            self.boltzmannSolver,
            self.thermodynamics,
            self.hydrodynamics,
            self.grid,
            numberOfFields,
            self.meanFreePath,
            wallThicknessBounds,
            wallOffsetBounds,
            includeOffEq=True,
            forceImproveConvergence=False,
            errTol=errTol,
            maxIterations=maxIterations,
            pressRelErrTol=pressRelErrTol,
        )

    def loadCollisionFiles(self, collision: Collision) -> None:
        """
        Loads collision files and reads them using the Boltzmann solver.

        This method takes a collision object as input and uses the Boltzmann solver to read the collision files.
        The collision object should contain the path of the collision file to be loaded.

        Args:
            collision (Collision): The collision object from collision_wrapper.py.

        Returns:
            None
        """
        print("=== WallGo collision generation ===")
        self.boltzmannSolver.readCollisions(collision)
    
    def generateCollisionFiles(self) -> None:
        """
        Loads collision files and reads them using the Boltzmann solver.

        This method takes a collision object as input and uses the Boltzmann solver to read the collision files.
        The collision object should contain the path of the collision file to be loaded.

        Args:
            collision (Collision): The collision object from collision_wrapper.py.

        Returns:
            None
        """
        self.loadCollisionFiles(self.collision)
    

    def wallSpeedLTE(self) -> float:
        """Solves wall speed in the Local Thermal Equilibrium approximation."""

        return self.hydrodynamics.findvwLTE()

    # Call after initGrid. I guess this would be the main workload function

    def solveWall(
        self, bIncludeOffEq: bool, wallThicknessIni: float = None
    ) -> WallGoResults:
        """Returns wall speed and wall parameters (widths and offsets)."""
        self.eom.includeOffEq = bIncludeOffEq
        # returning results
        return self.eom.findWallVelocityDeflagrationHybrid(wallThicknessIni)

    def solveWallDetonation(
        self,
        bIncludeOffEq: bool = True,
        wallThicknessIni: float = None,
        onlySmallest: bool = True,
    ) -> WallGoResults:
        """
        Finds all the detonation solutions by computing the pressure on a grid
        and interpolating to find the roots.

        Parameters
        ----------
        bIncludeOffEq : bool, optional
            If True, includes the out-of-equilibrium effects. The default is True.
        wallThicknessIni : float, optional
            Initial wall thickness. The default is None.
        overshootProb : float, optional
            Desired overshoot probability. A smaller value will lead to smaller step
            sizes which will take longer to evaluate, but with less chances of missing a
            root. The default is 0.05.

        Returns
        -------
        WallGoInterpolationResults
            Object containing the solutions and the pressures computed on the
            velocity grid.

        """
        self.eom.includeOffEq = bIncludeOffEq
        rtol = self.config.getfloat("EOM", "errTol")
        vmax = self.config.getfloat("EOM", "vwMaxDeton")
        nbrPointsMin = self.config.getfloat("EOM", "nbrPointsMinDeton")
        nbrPointsMax = self.config.getfloat("EOM", "nbrPointsMaxDeton")
        overshootProb = self.config.getfloat("EOM", "overshootProbDeton")
        vmin = max(self.hydrodynamics.vJ + 1e-10, self.hydrodynamics.slowestDeton())
        
        return self.eom.findWallVelocityDetonation(
            vmin,
            vmax,
            wallThicknessIni,
            nbrPointsMin,
            nbrPointsMax,
            overshootProb,
            rtol,
            onlySmallest
        )

    def _initalizeIntegralInterpolations(self, integrals: Integrals) -> None:

        assert self.config != None

        integrals.Jb.readInterpolationTable(
            getSafePathToResource(
                self.config.get("DataFiles", "InterpolationTable_Jb")
            ),
            bVerbose=False,
        )
        integrals.Jf.readInterpolationTable(
            getSafePathToResource(
                self.config.get("DataFiles", "InterpolationTable_Jf")
            ),
            bVerbose=False,
        )
