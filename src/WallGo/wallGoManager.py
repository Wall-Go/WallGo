"""
Defines the WallGoManager class which initializes the different object needed for the
wall velocity calculation.
"""

from typing import Type, TYPE_CHECKING
import numpy as np
from deprecated import deprecated
from dataclasses import dataclass
import pathlib

# WallGo imports
import WallGo
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
from .thermodynamics import Thermodynamics
from .results import WallGoResults, HydroResults
from .WallGoUtils import getSafePathToResource
from .EffectivePotential import EffectivePotential


@dataclass
class WallSolverSettings:
    """"""

    bIncludeOffEquilibrium: bool = True
    """If False, will ignore all out-of-equilibrium effects (no Boltzmann solving).
    """

    meanFreePath: float = 1.0
    """Estimate of the mean free path of the plasma. This will be used to set the
    tail lengths in the Grid object.
    [FIXME which units? what is a good default value?]
    """

    wallThicknessGuess: float | None = None
    """Initial guess of the wall thickness that will be used to solve the EOM.
    If None, we use value of 5/Tnucl. Default is None.
    [FIXME which units? Are we happy with None as the default?]
    """


@dataclass
class WallSolver:
    eom: EOM
    initialWallThickness: float


class WallGoManager:
    """Defines a 'control' class for managing the program flow.
    This should be better than writing the same stuff in every example main
    function, and is good for hiding some of our internal implementation
    details from the user.
    """

    def __init__(self) -> None:
        """"""

        # TODO cleanup, should not read the config here if we have a global WallGo
        # config object

        self.config = WallGo.config

        # default to working directory
        self.collisionDirectory = pathlib.Path.cwd()

        # These we currently have to keep cached, otherwise can't construct sensible WallSolver:
        ## TODO init these to None or have other easy way of checking if they have been properly initialized
        self.model: GenericModel
        self.hydrodynamics: Hydrodynamics
        self.phasesAtTn: PhaseInfo
        self.thermodynamics: Thermodynamics

    def setupThermodynamicsHydrodynamics(
        self,
        phaseInfo: WallGo.PhaseInfo,
        veffDerivativeScales: WallGo.VeffDerivativeScales,
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
        veffDerativeScales : VeffDerivativeScales
            WallGo dataclass containing the typical temprature and field scale over
            which the potential varies.

        Returns
        -------

        """

        assert (
            phaseInfo.phaseLocation1.numFields() == self.model.fieldCount
            and phaseInfo.phaseLocation2.numFields() == self.model.fieldCount
        ), "Invalid PhaseInfo input, field counts do not match those defined in the model"

        self.model.getEffectivePotential().setScales(veffDerivativeScales)

        # Checks that phase input makes sense with the user-specified Veff
        self.validatePhaseInput(phaseInfo)

        self.initTemperatureRange()

        ## Should we write these to a result struct?
        print("Temperature ranges:")
        print(
            "High-T phase: TMin = "
            f"{self.thermodynamics.freeEnergyHigh.minPossibleTemperature}, "
            f"TMax = {self.thermodynamics.freeEnergyHigh.maxPossibleTemperature}"
        )
        print(
            "Low-T phase: TMin = "
            f"{self.thermodynamics.freeEnergyLow.minPossibleTemperature}, "
            f"TMax = {self.thermodynamics.freeEnergyLow.maxPossibleTemperature}"
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

        print(f"Jouguet: {self.hydrodynamics.vJ}")
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

        potentialError = self.config.getfloat("EffectivePotential", "potentialError")

        ## FIXME default scales for derivatives, or just the simple ones built-in to the Veff initializer
        self.model.getEffectivePotential().setPotentialError(potentialError)

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

        print(f"Found phase 1: phi = {phaseLocation1}, Veff(phi) = {effPotValue1}")
        print(f"Found phase 2: phi = {phaseLocation2}, Veff(phi) = {effPotValue2}")

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
        Get initial guess for the relevant temperature range and store in internal thermodynamics object.
        """

        assert self.phasesAtTn is not None
        assert self.isModelValid()

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

        # Use the template model to find an estimate of the minimum and
        # maximum required temperatures

        try:
            ## Use the template model to find an estimate of the minimum and maximum
            ## required temperature
            hydrodynamicsTemplate = HydrodynamicsTemplateModel(self.thermodynamics)
            print(f"vwLTE in the template model: {hydrodynamicsTemplate.findvwLTE()}")

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

        _, _, THighTMaxTemplate, TLowTMaxTemplate = hydrodynamicsTemplate.findMatching(
            0.99 * hydrodynamicsTemplate.vJ
        )

        if THighTMaxTemplate is None:
            THighTMaxTemplate = self.config.getfloat("Hydrodynamics", "tmax") * Tn
        if TLowTMaxTemplate is None:
            TLowTMaxTemplate = self.config.getfloat("Hydrodynamics", "tmax") * Tn

        phaseTracerTol = self.config.getfloat("EffectivePotential", "phaseTracerTol")

        # Estimate of the dT needed to reach the desired tolerance considering
        # the error of a cubic spline scales like dT**4.
        dT = self.model.getEffectivePotential().temperatureScale * phaseTracerTol**0.25

        """If TMax, TMin are too close to real temperature boundaries
        the program can slow down significantly, but TMax must be large
        enough, and the template model only provides an estimate.
        HACK! fudgeFactor, see issue #145 """
        fudgeFactor = 1.2  # should be bigger than 1, but not known a priori
        ## TODO make a better choice for TMinHighT to speed up this function
        TMinHighT, TMaxHighT = 0, max(2 * Tn, fudgeFactor * THighTMaxTemplate)
        TMinLowT, TMaxLowT = 0, max(2 * Tn, fudgeFactor * TLowTMaxTemplate)

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
        """"""
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
        assert solver.initialWallThickness

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

        rtol = self.config.getfloat("EquationOfMotion", "errTol")
        nbrPointsMin = self.config.getint("EquationOfMotion", "nbrPointsMinDeton")
        nbrPointsMax = self.config.getint("EquationOfMotion", "nbrPointsMaxDeton")
        overshootProb = self.config.getfloat("EquationOfMotion", "overshootProbDeton")
        vmin = max(self.hydrodynamics.vJ + 1e-3, self.hydrodynamics.slowestDeton())
        vmax = self.config.getfloat("EquationOfMotion", "vwMaxDeton")

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
        Specifically, uses results of setupThermodynamicsHydrodynamics() to find optimal grid settings for wall solving,
        and creates Grid and BoltzmannSolver objects to use from within the EOM.
        Be aware that the created EOM object can change state if its creator WallGoManager instance is modified
        (eg. if setupThermodynamicsHydrodynamics() is called)!
        """

        assert (
            self.phasesAtTn.temperature is not None
            and self.isModelValid()
            and self.hydrodynamics is not None
        ), "Must run WallGoManager.setupThermodynamicsHydrodynamics() before wall solving"

        Tnucl: float = self.phasesAtTn.temperature

        gridMomentumFalloffScale = Tnucl

        wallThickness = wallSolverSettings.wallThicknessGuess

        if wallThickness is None:
            # Default guess: 5 / Tn
            wallThickness = 5.0 / Tnucl

        grid: Grid3Scales = self.buildGrid(
            wallThickness,
            wallSolverSettings.meanFreePath,
            gridMomentumFalloffScale,
        )

        # Hardcode basis types here: Cardinal for z, Chebyshev for pz, pp
        boltzmannSolver = BoltzmannSolver(grid, basisM="Cardinal", basisN="Chebyshev")

        boltzmannSolver.updateParticleList(self.model.outOfEquilibriumParticles)

        bShouldLoadCollisions = wallSolverSettings.bIncludeOffEquilibrium
        if bShouldLoadCollisions:
            boltzmannSolver.loadCollisions(self.collisionDirectory)

        eom: EOM = self.buildEOM(grid, boltzmannSolver, wallSolverSettings.meanFreePath)

        eom.includeOffEq = wallSolverSettings.bIncludeOffEquilibrium
        return WallSolver(eom, wallThickness)

    def _initHydrodynamics(self, thermodynamics: Thermodynamics) -> None:
        """
        Initialize the Hydrodynamics object.

        Parameters
        ----------
        thermodynamics : Thermodynamics
            Thermodynamics object.
        """
        tmax = self.config.getfloat("Hydrodynamics", "tmax")
        tmin = self.config.getfloat("Hydrodynamics", "tmin")
        rtol = self.config.getfloat("Hydrodynamics", "relativeTol")
        atol = self.config.getfloat("Hydrodynamics", "absoluteTol")

        self.hydrodynamics = Hydrodynamics(thermodynamics, tmax, tmin, rtol, atol)

    def buildGrid(
        self,
        wallThicknessIni: float,
        meanFreePath: float,
        initialMomentumFalloffScale: float,
    ) -> Grid3Scales:
        r"""
        Initialize a Grid3Scales object

        Parameters
        ----------
        wallThicknessIni : float
            Initial guess of the wall thickness that will be used to solve the EOM.
        meanFreePath : float
            Estimate of the mean free path of the plasma. This will be used to set the
            tail lengths in the Grid object.
        initialMomentumFalloffScale : float
            TODO documentation. Should be close to temperature at the wall
        """

        gridN = self.config.getint("PolynomialGrid", "momentumGridSize")
        gridM = self.config.getint("PolynomialGrid", "spatialGridSize")
        ratioPointsWall = self.config.getfloat("PolynomialGrid", "ratioPointsWall")
        smoothing = self.config.getfloat("PolynomialGrid", "smoothing")

        tailLength = max(
            meanFreePath, wallThicknessIni * (1.0 + 3.0 * smoothing) / ratioPointsWall
        )

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
            wallThicknessIni,
            initialMomentumFalloffScale,
            ratioPointsWall,
            smoothing,
        )

    def buildEOM(
        self, grid: Grid3Scales, boltzmannSolver: BoltzmannSolver, meanFreePath: float
    ) -> EOM:
        """
        Constructs an EOM object using internal state from the WallGoManager,
        and the input Grid/BoltzmannSolver.
        """
        numberOfFields = self.model.fieldCount

        errTol = self.config.getfloat("EquationOfMotion", "errTol")
        maxIterations = self.config.getint("EquationOfMotion", "maxIterations")
        pressRelErrTol = self.config.getfloat("EquationOfMotion", "pressRelErrTol")

        wallThicknessBounds = (
            self.config.getfloat("EquationOfMotion", "wallThicknessLowerBound"),
            self.config.getfloat("EquationOfMotion", "wallThicknessUpperBound"),
        )
        wallOffsetBounds = (
            self.config.getfloat("EquationOfMotion", "wallOffsetLowerBound"),
            self.config.getfloat("EquationOfMotion", "wallOffsetUpperBound"),
        )

        return EOM(
            boltzmannSolver,
            self.thermodynamics,
            self.hydrodynamics,
            grid,
            numberOfFields,
            meanFreePath,
            wallThicknessBounds,
            wallOffsetBounds,
            includeOffEq=True,
            forceImproveConvergence=False,
            errTol=errTol,
            maxIterations=maxIterations,
            pressRelErrTol=pressRelErrTol,
        )

    def _initalizeIntegralInterpolations(self, integrals: Integrals) -> None:
        """
        Initialize the interpolation of the thermal integrals.

        Parameters
        ----------
        integrals : Integrals
            Integrals object.
        """

        assert self.config is not None

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
