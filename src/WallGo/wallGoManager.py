"""
Defines the WallGoManager class which initializes the different object needed for the
wall velocity calculation.
"""

from dataclasses import dataclass
import pathlib
import logging
import numpy as np

# WallGo imports
import WallGo
from .boltzmann import BoltzmannSolver
from .containers import PhaseInfo
from .equationOfMotion import EOM
from .exceptions import WallGoError, WallGoPhaseValidationError
from .genericModel import GenericModel
from .grid3Scales import Grid3Scales
from .hydrodynamics import Hydrodynamics
from .hydrodynamicsTemplateModel import HydrodynamicsTemplateModel
from .thermodynamics import Thermodynamics
from .results import WallGoResults
from .config import Config


@dataclass
class WallSolverSettings:
    """
    Settings for the WallSolver.
    """

    bIncludeOffEquilibrium: bool = True
    """If False, will ignore all out-of-equilibrium effects (no Boltzmann solving).
    """

    meanFreePathScale: float = 100.0
    """Estimate of the mean free path of the plasma in units of 1/Tn. This will be used
    to set the tail lengths in the Grid object. Default is 100.
    """

    wallThicknessGuess: float = 5.0
    """
    Initial guess of the wall thickness that will be used to solve the EOM, in units
    1/Tn. Default is 5.
    """


@dataclass
class WallSolver:
    """
    Data class containing classes and settings for the wall velocity computation.
    """

    eom: EOM
    """EOM object"""

    grid: Grid3Scales
    """Grid3Scales object used to describe the mapping between the physical
    coordinates to the compact ones."""

    boltzmannSolver: BoltzmannSolver
    """ BoltzmannSolver object used to solve the Boltzmann equation."""

    initialWallThickness: float
    """Initial wall thickness used by the solver. Should be expressed in physical
    units (the units used in EffectivePotential)."""


class WallGoManager:
    """Defines a 'control' class for managing the program flow.
    This should be better than writing the same stuff in every example main
    function, and is good for hiding some of our internal implementation
    details from the user.
    """

    def __init__(self) -> None:
        """"""

        # Initialise the configs with the default values
        self.config = Config()

        # Set the default verbosity level to logging.INFO
        self.setVerbosity(logging.INFO)

        # default to working directory
        self.collisionDirectory = pathlib.Path.cwd()

        # These we currently have to keep cached, otherwise we can't construct
        # a sensible WallSolver:
        ## TODO init these to None or have other easy way of checking if they
        ## have been properly initialized
        self.model: GenericModel
        self.hydrodynamics: Hydrodynamics
        self.phasesAtTn: PhaseInfo
        self.thermodynamics: Thermodynamics

    def getMomentumGridSize(self) -> int:
        """
        Returns the momentum grid size.
        """
        return self.config.configGrid.momentumGridSize

    def setVerbosity(self, verbosityLevel: int) -> None:
        """
        Set the verbosity level.

        Parameters
        ----------
        verbosityLevel : int
            Verbosity level. Follows the standard convention of the logging module where
            DEBUG=10, INFO=20, WARNING=30 and ERROR=40. In WallGo, most of the
            information is shown at the INFO level. At the DEBUG level, more
            information about the calculation of the pressure at each iteration is
            shown.

        """
        logging.basicConfig(format='%(message)s', level=verbosityLevel, force=True)

    def setupThermodynamicsHydrodynamics(
        self,
        phaseInfo: WallGo.PhaseInfo,
        veffDerivativeScales: WallGo.VeffDerivativeSettings,
    ) -> None:
        """Must run before solveWall() and companions.
        Initialization of internal objects related to equilibrium thermodynamics and
        hydrodynamics. Specifically, we verify that the input PhaseInfo is valid
        (distinct phases can be found in the effective potential),
        estimate the relevant temperature range for wall solving and create efficient
        approximations for phase free energies over this range using interpolation.
        Finally, it initializes the Hydrodynamics and confirms that it can find a
        reasonable value for the Jouguet velocity (the transition from hybdrid to
        detonation solutions).
        You are required to run this function whenever details of your
        physics model change to keep the manager's internal state up to date.

        Parameters
        ----------
        phaseInfo : PhaseInfo
            WallGo object containing the approximate positions of the minima and
            nucleation temperature.
        veffDerativeScales : VeffDerivativeSettings
            WallGo dataclass containing the typical temprature and field scale over
            which the potential varies.

        Returns
        -------

        """

        assert (
            phaseInfo.phaseLocation1.numFields() == self.model.fieldCount
            and phaseInfo.phaseLocation2.numFields() == self.model.fieldCount
        ), "Invalid PhaseInfo input, field counts don't match those defined in model"

        self.model.getEffectivePotential().configureDerivatives(veffDerivativeScales)

        # Checks that phase input makes sense with the user-specified Veff
        self.validatePhaseInput(phaseInfo)

        self.initTemperatureRange()

        ## Should we write these to a result struct?
        logging.info("Temperature ranges:")
        logging.info(
            "High-T phase: TMin = "
            f"{self.thermodynamics.freeEnergyHigh.minPossibleTemperature[0]}, "
            f"TMax = {self.thermodynamics.freeEnergyHigh.maxPossibleTemperature[0]}"
        )
        logging.info(
            "Low-T phase: TMin = "
            f"{self.thermodynamics.freeEnergyLow.minPossibleTemperature[0]}, "
            f"TMax = {self.thermodynamics.freeEnergyLow.maxPossibleTemperature[0]}"
        )

        self.thermodynamics.setExtrapolate()
        self._initHydrodynamics(self.thermodynamics)

        if (
            not np.isfinite(self.hydrodynamics.vJ)
            or self.hydrodynamics.vJ > 1
            or self.hydrodynamics.vJ < 0
        ):
            raise WallGoError(
                "Failed to solve Jouguet velocity at input temperature!",
                data={
                    "vJ": self.hydrodynamics.vJ,
                    "temperature": phaseInfo.temperature,
                },
            )

        logging.info(f"Jouguet: {self.hydrodynamics.vJ}")
        # TODO return some results struct

    def isModelValid(self) -> bool:
        """True if a valid model is currently registered."""
        return self.model is not None

    def registerModel(self, model: GenericModel) -> None:
        """
        Register a physics model with WallGo.

        Parameters
        ----------
        model : GenericModel
            GenericModel object that describes the model studied.
        """
        assert isinstance(model, GenericModel)
        assert (
            model.fieldCount > 0
        ), "WallGo model must contain at least one classical field"

        self.model = model

    def validatePhaseInput(self, phaseInput: PhaseInfo) -> None:
        """
        This checks that the user-specified phases are OK.
        Specifically, the effective potential should have two minima at the given T,
        otherwise phase transition analysis is not possible.

        Parameters
        ----------
        phaseInput : PhaseInfo
            Should contain approximate field values at the two phases that WallGo will
            analyze, and the nucleation temperature. Transition is assumed to go
            phaseLocation1 --> phaseLocation2.
        """

        T = phaseInput.temperature

        # Find the actual minima at T, should be close to the user-specified locations
        (
            phaseLocation1,
            effPotValue1,
        ) = self.model.getEffectivePotential().findLocalMinimum(
            phaseInput.phaseLocation1, T
        )
        (
            phaseLocation2,
            effPotValue2,
        ) = self.model.getEffectivePotential().findLocalMinimum(
            phaseInput.phaseLocation2, T
        )

        logging.info(f"Found phase 1: phi = {phaseLocation1}, Veff(phi) = {effPotValue1}")
        logging.info(f"Found phase 2: phi = {phaseLocation2}, Veff(phi) = {effPotValue2}")

        if np.allclose(phaseLocation1, phaseLocation2, rtol=1e-05, atol=1e-05):
            raise WallGoPhaseValidationError(
                "It looks like both phases are the same, this will not work",
                phaseInput,
                {
                    "phaseLocation1": phaseLocation1,
                    "Veff(phi1)": effPotValue1,
                    "phaseLocation2": phaseLocation2,
                    "Veff(phi2)": effPotValue2,
                },
            )

        ## Currently we assume transition phase1 -> phase2. This assumption shows up at
        ## least when initializing FreeEnergy objects
        if np.real(effPotValue1) < np.real(effPotValue2):
            raise WallGoPhaseValidationError(
                "Phase 1 has lower free energy than Phase 2, this will not work",
                phaseInput,
                {
                    "phaseLocation1": phaseLocation1,
                    "Veff(phi1)": effPotValue1,
                    "phaseLocation2": phaseLocation2,
                    "Veff(phi2)": effPotValue2,
                },
            )

        foundPhaseInfo = PhaseInfo(
            temperature=T, phaseLocation1=phaseLocation1, phaseLocation2=phaseLocation2
        )

        self.phasesAtTn = foundPhaseInfo

    def initTemperatureRange(self) -> None:
        """
        Determine the relevant temperature range and trace the phases
        over this range. Interpolate the free energy in both phases and
        store in internal thermodynamics object.
        """

        assert self.phasesAtTn is not None
        assert self.isModelValid()
        assert (
            self.model.getEffectivePotential().areDerivativesConfigured()
        ), "Must have called effectivePotential.configureDerivatives()"

        Tn = self.phasesAtTn.temperature

        self.thermodynamics = Thermodynamics(
            self.model.getEffectivePotential(),
            Tn,
            self.phasesAtTn.phaseLocation2,
            self.phasesAtTn.phaseLocation1,
        )

        # Let's turn these off so that things are more transparent
        self.thermodynamics.freeEnergyHigh.disableAdaptiveInterpolation()
        self.thermodynamics.freeEnergyLow.disableAdaptiveInterpolation()

        try:
            # Use the template model to find an estimate of the minimum and maximum
            # required temperature. We do not solve hydrodynamics inside the bubble, so
            # we are only interested in T- (the temperature right at the wall).
            hydrodynamicsTemplate = HydrodynamicsTemplateModel(self.thermodynamics)
            logging.info(f"vwLTE in the template model: {hydrodynamicsTemplate.findvwLTE()}")

        except WallGoError as error:
            # Throw new error with more info
            raise WallGoPhaseValidationError(
                error.message, self.phasesAtTn, error.data
            ) from error

        # Raise an error if this is an inverse PT (if epsilon is negative)
        if hydrodynamicsTemplate.epsilon < 0:
            raise WallGoError(
                f"WallGo requires epsilon={hydrodynamicsTemplate.epsilon} to be "
                "positive."
            )

        # Maximum values for T+ and T- are reached at the Jouguet velocity
        _, _, THighTMaxTemplate, TLowTMaxTemplate = hydrodynamicsTemplate.findMatching(
            0.99 * hydrodynamicsTemplate.vJ
        )

        # Minimum value for T- is reached at small wall velocity. The minimum value
        # for T+ is the nucleation temperature.
        _, _, TLowTMinTemplate, _ = hydrodynamicsTemplate.findMatching(1e-3)

        if THighTMaxTemplate is None:
            THighTMaxTemplate = self.config.configHydrodynamics.tmax * Tn
        if TLowTMaxTemplate is None:
            TLowTMaxTemplate = self.config.configHydrodynamics.tmax * Tn

        if TLowTMinTemplate is None:
            TLowTMinTemplate = self.config.configHydrodynamics.tmin * Tn

        phaseTracerTol = self.config.configThermodynamics.phaseTracerTol

        # Estimate of the dT needed to reach the desired tolerance considering
        # the error of a cubic spline scales like dT**4.
        dT = (
            self.model.getEffectivePotential().derivativeSettings.temperatureVariationScale
            * phaseTracerTol**0.25
        )

        """Since the template model is an approximation of the full model, 
        and since the temperature profile in the wall could be non-monotonous,
        we should not take exactly the TMin and TMax from the template model.
        We use a configuration parameter to determine the TMin and TMax that we
        use in the phase tracing.
        """
        TMinHighT = Tn * self.config.configThermodynamics.tmin
        TMaxHighT = THighTMaxTemplate * self.config.configThermodynamics.tmax
        TMinLowT = TLowTMinTemplate * self.config.configThermodynamics.tmin
        TMaxLowT = TLowTMaxTemplate * self.config.configThermodynamics.tmax

        # Interpolate phases and check that they remain stable in this range
        fHighT = self.thermodynamics.freeEnergyHigh
        fLowT = self.thermodynamics.freeEnergyLow

        fHighT.tracePhase(TMinHighT, TMaxHighT, dT, phaseTracerTol)
        fLowT.tracePhase(TMinLowT, TMaxLowT, dT, phaseTracerTol)

    def setPathToCollisionData(self, directoryPath: pathlib.Path) -> None:
        """
        Specify path to collision files for use with the Boltzmann solver.
        This does not necessarily load the files immediately.

        Args:
            directoryPath (pathlib.Path): Directory containing the .hdf5 collision data.

        Returns:
            None
        """
        # TODO validate? or not if we want to allow the data to be generated on the fly
        # should at least validate that it is not an existing file
        self.collisionDirectory = directoryPath

    def getCurrentCollisionDirectory(self) -> pathlib.Path:
        """
        Returns the path to the directory with the collision files.
        """
        return self.collisionDirectory

    def wallSpeedLTE(self) -> float:
        """
        Solves wall speed in the Local Thermal Equilibrium (LTE) approximation.

        Returns
        -------
        float
            Wall velocity in LTE.
        """

        return self.hydrodynamics.findvwLTE()

    def solveWall(
        self,
        wallSolverSettings: WallSolverSettings,
    ) -> WallGoResults:
        """
        Solves the EOM and computes the wall velocity.
        Must be ran after analyzeHydrodynamics() because
        the solver depends on thermodynamical and hydrodynamical
        info stored internally in the WallGoManager.

        Parameters
        ----------
        wallSolverSettings : WallSolverSettings
            Configuration settings for the solver.

        Returns
        -------
        WallGoResults
            Object containing the wall velocity and EOM solution, as well as different
            quantities used to assess the accuracy of the solution.
        """
        solver: WallSolver = self.setupWallSolver(wallSolverSettings)

        return solver.eom.findWallVelocityDeflagrationHybrid(
            solver.initialWallThickness
        )

    def solveWallDetonation(
        self,
        wallSolverSettings: WallSolverSettings,
        onlySmallest: bool = True,
    ) -> list[WallGoResults]:
        """
        Finds all the detonation solutions by computing the pressure on a grid
        and interpolating to find the roots.

        Parameters
        ----------
        wallSolverSettings : WallSolverSettings
            Configuration settings for the solver.

        onlySmallest : bool, optional
            Whether or not to only look for one solution. If True, the solver will
            stop the calculation after finding the first root. If False, it will
            continue looking for solutions until it reaches the maximal velocity.

        Returns
        -------
        list[WallGoResults]
            List containing the detonation solutions. If no solutions were found,
            returns a wall velocity of 0  if the pressure is always positive, or 1 if
            it is negative (runaway wall). If it is positive at vmin and negative at
            vmax, the outcome is uncertain and would require a time-dependent analysis,
            so it returns an empty list.

        """

        solver: WallSolver = self.setupWallSolver(wallSolverSettings)
        assert solver.initialWallThickness

        rtol = self.config.configEOM.errTol
        nbrPointsMin = self.config.configEOM.nbrPointsMinDeton
        nbrPointsMax = self.config.configEOM.nbrPointsMaxDeton
        overshootProb = self.config.configEOM.overshootProbDeton
        vmin = max(self.hydrodynamics.vJ + 1e-3, self.hydrodynamics.slowestDeton())
        vmax = self.config.configEOM.vwMaxDeton

        if vmin >= vmax:
            raise WallGoError(
                "In WallGoManager.solveWallDetonation(): vmax must be larger than vmin",
                {"vmin": vmin, "vmax": vmax},
            )

        return solver.eom.findWallVelocityDetonation(
            vmin,
            vmax,
            solver.initialWallThickness,
            nbrPointsMin,
            nbrPointsMax,
            overshootProb,
            rtol,
            onlySmallest,
        )

    def setupWallSolver(self, wallSolverSettings: WallSolverSettings) -> WallSolver:
        """Helper for constructing an EOM object whose state is tied to equilibrium
        and hydrodynamical information stored in the WallGoManager.
        Specifically, uses results of setupThermodynamicsHydrodynamics() to find
        optimal grid settings for wall solving, and creates Grid and BoltzmannSolver
        objects to use from within the EOM. Be aware that the created EOM object can
        change state if its creator WallGoManager instance is modified
        (eg. if setupThermodynamicsHydrodynamics() is called)!
        """

        assert (
            self.phasesAtTn.temperature is not None
            and self.isModelValid()
            and self.hydrodynamics is not None
        ), "Run WallGoManager.setupThermodynamicsHydrodynamics() before wall solving"

        Tnucl: float = self.phasesAtTn.temperature

        gridMomentumFalloffScale = Tnucl

        wallThickness = wallSolverSettings.wallThicknessGuess
        meanFreePathScale = wallSolverSettings.meanFreePathScale

        grid: Grid3Scales = self.buildGrid(
            wallThickness,
            meanFreePathScale,
            gridMomentumFalloffScale,
        )

        # Factor that multiplies the collision term in the Boltzmann equation.
        collisionMultiplier = self.config.configBoltzmannSolver.collisionMultiplier
        # Hardcode basis types here: Cardinal for z, Chebyshev for pz, pp
        boltzmannSolver = BoltzmannSolver(
            grid,
            basisM="Cardinal",
            basisN="Chebyshev",
            collisionMultiplier=collisionMultiplier,
            )

        boltzmannSolver.updateParticleList(self.model.outOfEquilibriumParticles)

        bShouldLoadCollisions = wallSolverSettings.bIncludeOffEquilibrium
        if bShouldLoadCollisions:
            # This throws if collision load fails, let caller handle the exception.
            # TODO may be cleaner to handle it here and return an invalid solver
            boltzmannSolver.loadCollisions(self.collisionDirectory)

        eom: EOM = self.buildEOM(grid, boltzmannSolver, meanFreePathScale)

        eom.includeOffEq = wallSolverSettings.bIncludeOffEquilibrium
        return WallSolver(eom, grid, boltzmannSolver, wallThickness / Tnucl)

    def _initHydrodynamics(self, thermodynamics: Thermodynamics) -> None:
        """
        Initialize the Hydrodynamics object.

        Parameters
        ----------
        thermodynamics : Thermodynamics
            Thermodynamics object.
        """
        tmax = self.config.configHydrodynamics.tmax
        tmin = self.config.configHydrodynamics.tmin
        rtol = self.config.configHydrodynamics.relativeTol
        atol = self.config.configHydrodynamics.absoluteTol

        self.hydrodynamics = Hydrodynamics(thermodynamics, tmax, tmin, rtol, atol)

    def buildGrid(
        self,
        wallThicknessIni: float,
        meanFreePathScale: float,
        initialMomentumFalloffScale: float,
    ) -> Grid3Scales:
        r"""
        Initialize a Grid3Scales object

        Parameters
        ----------
        wallThicknessIni : float
            Initial guess of the wall thickness that will be used to solve the EOM.
            Should be expressed in units of 1/Tnucl.
        meanFreePathScale : float
            Estimate of the mean free path of the plasma. This will be used to set the
            tail lengths in the Grid object. Should be expressed in units of 1/Tnucl.
        initialMomentumFalloffScale : float
            TODO documentation. Should be close to temperature at the wall
        """

        gridN = self.config.configGrid.momentumGridSize
        gridM = self.config.configGrid.spatialGridSize
        ratioPointsWall = self.config.configGrid.ratioPointsWall
        smoothing = self.config.configGrid.smoothing
        
        Tnucl = self.phasesAtTn.temperature

        # We divide by Tnucl to get it in physical units of length
        tailLength = max(
            meanFreePathScale, wallThicknessIni * (1.0 + 3.0 * smoothing) / ratioPointsWall
        ) / Tnucl

        if gridN % 2 == 0:
            raise ValueError(
                "You have chosen an even number N of momentum-grid points. "
                "WallGo only works with odd N, please change it to an odd number."
            )

        return Grid3Scales(
            gridM,
            gridN,
            tailLength,
            tailLength,
            wallThicknessIni / Tnucl,
            initialMomentumFalloffScale,
            ratioPointsWall,
            smoothing,
        )

    def buildEOM(
        self, grid: Grid3Scales, boltzmannSolver: BoltzmannSolver, meanFreePathScale: float
    ) -> EOM:
        """
        Constructs an EOM object using internal state from the WallGoManager,
        and the input Grid/BoltzmannSolver.

        Parameters
        ----------
        grid : Grid3Scales
            Grid3Scales object used to describe the mapping between the physical
            coordinates to the compact ones.
        boltzmannSolver : BoltzmannSolver
            BoltzmannSolver object used to solve the Boltzmann equation.
        meanFreePathScale : float
            Estimate of the mean free path of the plasma. This will be used to set the
            tail lengths in the Grid object. Should be expressed in units of 1/Tnucl.

        Returns
        -------
        EOM
            EOM object.

        """
        numberOfFields = self.model.fieldCount

        errTol = self.config.configEOM.errTol
        maxIterations = self.config.configEOM.maxIterations
        pressRelErrTol = self.config.configEOM.pressRelErrTol
        conserveEnergy = self.config.configEOM.conserveEnergyMomentum

        wallThicknessBounds = self.config.configEOM.wallThicknessBounds
        wallOffsetBounds = self.config.configEOM.wallOffsetBounds
        
        Tnucl = self.phasesAtTn.temperature

        return EOM(
            boltzmannSolver,
            self.thermodynamics,
            self.hydrodynamics,
            grid,
            numberOfFields,
            meanFreePathScale / Tnucl, # We divide by Tnucl to get physical units
            wallThicknessBounds,
            wallOffsetBounds,
            includeOffEq=True,
            forceEnergyConservation=conserveEnergy,
            forceImproveConvergence=False,
            errTol=errTol,
            maxIterations=maxIterations,
            pressRelErrTol=pressRelErrTol,
        )

