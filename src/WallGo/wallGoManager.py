"""
Defines the WallGoManager class which initializes the different object needed for the
wall velocity calculation.
"""

from typing import Type, TYPE_CHECKING
import numpy as np
from deprecated import deprecated
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
from .integrals import Integrals
from .thermodynamics import Thermodynamics
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
        fieldScaleInput: list | np.ndarray | float,
    ):
        """
        Do common model-independent setup here

        Parameters
        ----------
        wallThicknessIni : float
            Initial guess of the wall thickness that will be used to solve the EOM.
        meanFreePath : float
            Estimate of the mean free path of the plasma. This will be used to set the
            tail lengths in the Grid object.
        temperatureScaleInput : float
            Temperature scale over which the potential changes by O(1). A good value
            would be of order Tc-Tn.
        fieldScaleInput : list, np.ndarray or float
            Field scale over which the potential changes by O(1). A good value would be
            similar to the field VEV. If a float is given, all the fields are assumed to
            have the same scale.
        """

        # TODO cleanup, should not read the config here if we have a global WallGo
        # config object

        self.config = WallGo.config

        # -- Order of initialization matters here

        # Grid
        self.grid: Grid3Scales
        self._initGrid(
            wallThicknessIni,
            meanFreePath,
        )

        self._initBoltzmann()
        self.temperatureScaleInput = temperatureScaleInput
        self.fieldScaleInput = fieldScaleInput
        self.model: GenericModel
        self.boltzmannSolver: BoltzmannSolver
        self.hydrodynamics: Hydrodynamics
        self.phasesAtTn: PhaseInfo
        self.thermodynamics: Thermodynamics
        self.eom: EOM

    def _initModel(self, model: GenericModel) -> None:
        """
        Initializes the model.
        """
        self.model = model

    def registerModel(self, model: GenericModel) -> None:
        """
        Register a physics model with WallGo.

        Parameters
        ----------
        model : GenericModel
            GenericModel object that describes the model studied.
        """
        assert isinstance(model, GenericModel)
        self._initModel(model)

        potentialError = self.config.getfloat("EffectivePotential", "potentialError")

        self.model.effectivePotential.setPotentialError(potentialError)
        self.model.effectivePotential.setScales(
            float(self.temperatureScaleInput), self.fieldScaleInput
        )

        # Update Boltzmann off-eq particle list to match that defined in model
        self.boltzmannSolver.updateParticleList(model.outOfEquilibriumParticles)

    def setParameters(self, phaseInput: PhaseInfo) -> None:
        """
        Validate the phase input and initialize the temperature range and several
        objects that will be used for the calculation.

        Parameters
        ----------
        phaseInput: WallGo.PhaseInfo
            Should contain approximate field values at the two phases that WallGo will
            analyze, and the nucleation temperature. Transition is assumed to go
            phaseLocation1 --> phaseLocation2.
        """

        # Checks that phase input makes sense with the user-specified Veff
        self.validatePhaseInput(phaseInput)

        # Change the falloff scale in grid now that we have a good guess for
        # the plasma temperature
        self.grid.changeMomentumFalloffScale(phaseInput.temperature)

        self.initTemperatureRange()

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
        self._initEOM()

        if (
            not np.isfinite(self.hydrodynamics.vJ)
            or self.hydrodynamics.vJ > 1
            or self.hydrodynamics.vJ < 0
        ):
            raise WallGoError(
                "Failed to solve Jouguet velocity at input temperature!",
                data={
                    "vJ": self.hydrodynamics.vJ,
                    "temperature": phaseInput.temperature,
                },
            )

        print(f"Jouguet: {self.hydrodynamics.vJ}")

    def changeInputParameters(
        self,
        inputParameters: dict[str, float],
        effectivePotential: Type[EffectivePotential],
    ) -> None:
        """
        Recomputes the model parameters when the user provides new inputparameters.
        Also updates the effectivePotential correspondingly.

        Parameters
        ----------
        inputParameters : dict
            Parameters used to compute the model parameters.
        effectivePotential : EffectivePotential
            Class that inherits from EffectivePotential
            (not an object of EffectivePotential). A new object of this class will be
            created and stored in self.model.effectivePotential.
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
        phaseLocation1, effPotValue1 = self.model.effectivePotential.findLocalMinimum(
            phaseInput.phaseLocation1, T
        )
        phaseLocation2, effPotValue2 = self.model.effectivePotential.findLocalMinimum(
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
        Get initial guess for the relevant temperature range and store in internal TMin,
        TMax.
        """

        # LN: this routine is probably too heavy. We could at least drop the
        # Tc part, or find it after FreeEnergy interpolations are done

        assert self.phasesAtTn is not None

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
        dT = self.model.effectivePotential.temperatureScale * phaseTracerTol**0.25

        """If TMax, TMin are too close to real temperature boundaries
        the program can slow down significantly, but TMax must be large
        enough, and the template model only provides an estimate.
        HACK! fudgeFactor, see issue #145 """
        fudgeFactor = 1.2  # should be bigger than 1, but not known a priori
        TMinHighT, TMaxHighT = 0, max(2 * Tn, fudgeFactor * THighTMaxTemplate)
        TMinLowT, TMaxLowT = 0, max(2 * Tn, fudgeFactor * TLowTMaxTemplate)

        # Interpolate phases and check that they remain stable in this range
        fHighT = self.thermodynamics.freeEnergyHigh
        fLowT = self.thermodynamics.freeEnergyLow

        fHighT.tracePhase(TMinHighT, TMaxHighT, dT, phaseTracerTol)
        fLowT.tracePhase(TMinLowT, TMaxLowT, dT, phaseTracerTol)

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

    def _initGrid(self, wallThicknessIni: float, meanFreePath: float) -> None:
        r"""
        Initialize the Grid object

        Parameters
        ----------
        wallThicknessIni : float
            Initial guess of the wall thickness that will be used to solve the EOM.
        meanFreePath : float
            Estimate of the mean free path of the plasma. This will be used to set the
            tail lengths in the Grid object.
        """

        # To initialize Grid we need to specify a "temperature" scale that has
        # analogous role as L_xi, but for the momenta. In practice this scale
        # needs to be close to temperature near the wall, but we don't know
        # that yet, so just initialize with some value here and update once the
        # nucleation temperature is obtained.
        initialMomentumFalloffScale = 50.0

        gridN = self.config.getint("PolynomialGrid", "momentumGridSize")
        gridM = self.config.getint("PolynomialGrid", "spatialGridSize")
        ratioPointsWall = self.config.getfloat("PolynomialGrid", "ratioPointsWall")
        smoothing = self.config.getfloat("PolynomialGrid", "smoothing")
        self.meanFreePath = meanFreePath

        tailLength = max(
            meanFreePath, wallThicknessIni * (1 + 3 * smoothing) / ratioPointsWall
        )

        if gridN % 2 == 0:
            raise ValueError(
                "You have chosen an even number N of momentum-grid points. "
                "WallGo only works with odd N, please change it to an odd number."
            )
        self.grid = Grid3Scales(
            gridM,
            gridN,
            tailLength,
            tailLength,
            wallThicknessIni,
            initialMomentumFalloffScale,
            ratioPointsWall,
            smoothing,
        )

    def _initBoltzmann(self) -> None:
        """
        Initialize the BoltzmannSolver object.
        """
        # Hardcode basis types here: Cardinal for z, Chebyshev for pz, pp
        self.boltzmannSolver = BoltzmannSolver(
            self.grid, basisM="Cardinal", basisN="Chebyshev"
        )

    def _initEOM(self) -> None:
        """
        Initialize the EOM object.
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

    def loadCollisionFiles(self, directoryPath: pathlib.Path) -> None:
        """
        Loads collision files for use with the Boltzmann solver.

        Args:
            directoryPath (pathlib.Path): Directory containing the .hdf5 collision data.

        Returns:
            None
        """
        self.boltzmannSolver.loadCollisions(directoryPath)

    # this actually crashes now, so its more than deprecated
    @deprecated("Use WallGoManager.loadCollisionFiles")
    def generateCollisionFiles(self) -> None:
        """
        Loads collision files and reads them using the Boltzmann solver.

        This method takes a collision object as input and uses the Boltzmann solver to
        read the collision files. The collision object should contain the path of the
        collision file to be loaded.
        """
        self.loadCollisionFiles(self.collision)

    def wallSpeedLTE(self) -> float:
        """
        Solves wall speed in the Local Thermal Equilibrium approximation.

        Returns
        -------
        float
            Wall velocity in LTE.
        """

        return self.hydrodynamics.findvwLTE()

    # Call after initGrid. I guess this would be the main workload function

    def solveWall(
        self, bIncludeOffEq: bool, wallThicknessIni: float | None = None
    ) -> WallGoResults:
        """
        Solves the EOM and computes the wall velocity.

        Parameters
        ----------
        bIncludeOffEq : bool
            Weither or not to include out-of-equilibrium effects.
        wallThicknessIni : float or None, optional
            Initial guess of the wall thickness that will be used to solve the EOM. If
            None, set it to 5/Tnucl. Default is None.

        Returns
        -------
        WallGoResults
            Object containing the wall velocity and EOM solution, as well as different
            quantities used to assess the accuracy of the solution.
        """
        self.eom.includeOffEq = bIncludeOffEq
        # returning results
        return self.eom.findWallVelocityDeflagrationHybrid(wallThicknessIni)

    def solveWallDetonation(
        self,
        bIncludeOffEq: bool = True,
        wallThicknessIni: float | None = None,
        onlySmallest: bool = True,
    ) -> list[WallGoResults]:
        """
        Finds all the detonation solutions by computing the pressure on a grid
        and interpolating to find the roots.

        Parameters
        ----------
        bIncludeOffEq : bool, optional
            If True, includes the out-of-equilibrium effects. The default is True.
        wallThicknessIni : float or None, optional
            Initial wall thickness used by the solver. If None, uses the value 5/Tn.
            The default is None.
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
        self.eom.includeOffEq = bIncludeOffEq
        rtol = self.config.getfloat("EquationOfMotion", "errTol")
        nbrPointsMin = self.config.getint("EquationOfMotion", "nbrPointsMinDeton")
        nbrPointsMax = self.config.getint("EquationOfMotion", "nbrPointsMaxDeton")
        overshootProb = self.config.getfloat("EquationOfMotion", "overshootProbDeton")
        vmin = max(self.hydrodynamics.vJ + 1e-3, self.hydrodynamics.slowestDeton())
        vmax = self.config.getfloat("EquationOfMotion", "vwMaxDeton")

        if vmin >= vmax:
            raise WallGoError(
                "vmax must be larger than vmin", {"vmin": vmin, "vmax": vmax}
            )

        return self.eom.findWallVelocityDetonation(
            vmin,
            vmax,
            wallThicknessIni,
            nbrPointsMin,
            nbrPointsMax,
            overshootProb,
            rtol,
            onlySmallest,
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
