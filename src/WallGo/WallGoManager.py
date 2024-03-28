import numpy as np

# WallGo imports
from .Boltzmann import BoltzmannSolver
from .EOM import EOM
from .GenericModel import GenericModel
from .Grid import Grid
from .Hydro import Hydro  # TODO why is this not Hydrodynamics? compare with Thermodynamics
from .HydroTemplateModel import HydroTemplateModel
from .Integrals import Integrals
from .Thermodynamics import Thermodynamics
from .WallGoExceptions import WallGoError, WallGoPhaseValidationError
from .WallGoTypes import PhaseInfo, WallGoResults
from .WallGoUtils import getSafePathToResource

import WallGo


class WallGoManager:
    """ Defines a 'control' class for managing the program flow.
    This should be better than writing the same stuff in every example main
    function, and is good for hiding some of our internal implementation
    details from the user.
    """

    def __init__(self):
        """do common model-independent setup here"""

        # TODO cleanup, should not read the config here if we have a global WallGo config object
        #self.config = Config()
        #self.config.readINI( getSafePathToResource("Config/WallGoDefaults.ini") )

        self.config = WallGo.config

        #self.integrals = Integrals()

        #self._initalizeIntegralInterpolations(self.integrals)

        # -- Order of initialization matters here

        # Grid
        self._initGrid(
            self.config.getint("PolynomialGrid", "spatialGridSize"),
            self.config.getint("PolynomialGrid", "momentumGridSize"),
            self.config.getfloat("PolynomialGrid", "L_xi"),
        )

        self._initBoltzmann()

    def registerModel(self, model: GenericModel) -> None:
        """Register a physics model with WallGo."""
        assert isinstance(model, GenericModel)
        self.model = model
        
        temperatureScale = self.config.getfloat("EffectivePotential", "temperatureScale")
        potentialError = self.config.getfloat("EffectivePotential", "potentialError")
        fieldScaleStr = self.config.get("EffectivePotential", "fieldScale")
        try:
            fieldScale = float(fieldScaleStr)
        except:
            fieldScale = [float(x) for x in fieldScaleStr.split(',')]
        
        self.model.effectivePotential.setPotentialError(potentialError)
        self.model.effectivePotential.setScales(temperatureScale, fieldScale)

        # Update Boltzmann off-eq particle list to match that defined in model
        self.boltzmannSolver.updateParticleList(model.outOfEquilibriumParticles)

    def setParameters(
        self, modelParameters: dict[str, float], phaseInput: PhaseInfo
    ) -> None:
        """Parameters
        ----------
        modelParameters: dict[str, float]
                        Dict containing all QFT model parameters:
                        Those that enter the action and the renormalization scale.
        phaseInput: WallGo.PhaseInfo
                    Should contain approximate field values at the two phases that WallGo will analyze,
                    and the nucleation temperature. Transition is assumed to go phaseLocation1 --> phaseLocation2.
        """

        self.model.modelParameters = modelParameters

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
        self._initHydro(self.thermodynamics)


        if not np.isfinite(self.hydro.vJ) or self.hydro.vJ > 1 or self.hydro.vJ < 0:
            raise WallGoError("Failed to solve Jouguet velocity at input temperature!", 
                              data = {"vJ" : self.hydro.vJ, "temperature" : phaseInput.temperature, "TMin" : self.TMin, "TMax" : self.TMax})

        print(f"Jouguet: {self.hydro.vJ}")

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
            raise WallGoPhaseValidationError("It looks like both phases are the same, this will not work",
                                             phaseInput, {"phaseLocation1" : phaseLocation1, "Veff(phi1)" : VeffValue1, 
                                                          "phaseLocation2" : phaseLocation2, "Veff(phi2)" : VeffValue2}) 

        ## Currently we assume transition phase1 -> phase2. This assumption shows up at least when initializing FreeEnergy objects
        if (np.real(VeffValue1) < np.real(VeffValue2)):
            raise WallGoPhaseValidationError("Phase 1 has lower free energy than Phase 2, this will not work",
                                             phaseInput, {"phaseLocation1" : phaseLocation1, "Veff(phi1)" : VeffValue1, 
                                                          "phaseLocation2" : phaseLocation2, "Veff(phi2)" : VeffValue2}) 
        
        foundPhaseInfo = PhaseInfo(temperature=T, phaseLocation1=phaseLocation1, phaseLocation2=phaseLocation2)

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
            hydrotemplate = HydroTemplateModel(self.thermodynamics)

        except WallGoError as error:
            # Throw new error with more info
            raise WallGoPhaseValidationError(error.message, self.phasesAtTn, error.data)

        _, _, THighTMaxTemplate, TLowTTMaxTemplate = hydrotemplate.findMatching(
            0.99 * hydrotemplate.vJ
        )
        
        phaseTracerTol = self.config.getfloat("EffectivePotential", "phaseTracerTol")
        
        # Estimate of the dT needed to reach the desired tolerance considering 
        # the error of a cubic spline scales like dT**4.
        dT = self.model.effectivePotential.temperatureScale * phaseTracerTol**0.25

        """If TMax, TMin are too close to real temperature boundaries
        the program can slow down significantly, but TMax must be large
        enough, and the template model only provides an estimate.
        HACK! fudgeFactor and -2 * dT, see issue #145 """
        fudgeFactor = 1.2  # should be bigger than 1, but not know a priori
        TMinHighT, TMaxHighT = 0, fudgeFactor * THighTMaxTemplate   ## Jorinde: it does not work if TMinHightT is set to Tn - 2*dT. It does if it is set to 0
        TMinLowT, TMaxLowT = 0, fudgeFactor * TLowTTMaxTemplate

        # Interpolate phases and check that they remain stable in this range
        fHighT = self.thermodynamics.freeEnergyHigh
        fLowT = self.thermodynamics.freeEnergyLow
        fHighT.tracePhase(TMinHighT, TMaxHighT, dT, phaseTracerTol)
        fLowT.tracePhase(TMinLowT, TMaxLowT, dT, phaseTracerTol)

        # Find critical temperature for dT
        self.Tc = self.thermodynamics.findCriticalTemperature(
            dT=dT, rTol=phaseTracerTol,
        )

        if (self.Tc < Tn):
            raise WallGoPhaseValidationError(
                f"Got Tc < Tn, should not happen!",
                Tn,
                {"Tc" : self.Tc},
            )
        print(f"Found Tc = {self.Tc} GeV.")

    def _initHydro(
        self, thermodynamics: Thermodynamics
    ) -> None:
        """"""
        
        self.hydro = Hydro(thermodynamics)

    def _initGrid(self, M: int, N: int, L_xi: float) -> Grid:
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

        N, M = int(N), int(M)
        if N % 2 == 0:
            raise ValueError(
                "You have chosen an even number N of momentum-grid points. "
                "WallGo only works with odd N, please change it to an odd number."
            )

        self.grid = Grid(M, N, L_xi, initialMomentumFalloffScale)

    def _initBoltzmann(self):
        # Hardcode basis types here: Cardinal for z, Chebyshev for pz, pp
        self.boltzmannSolver = BoltzmannSolver(
            self.grid, basisM="Cardinal", basisN="Chebyshev"
        )

    def loadCollisionFiles(self, fileName: str) -> None:
        self.boltzmannSolver.readCollisions(fileName)

    def wallSpeedLTE(self) -> float:
      
        """Solves wall speed in the Local Thermal Equilibrium approximation."""

        return self.hydro.findvwLTE()

    # Call after initGrid. I guess this would be the main workload function

    def solveWall(self, bIncludeOffEq: bool) -> WallGoResults:
        """Returns wall speed and wall parameters (widths and offsets).
        """

        numberOfFields = self.model.fieldCount

        errTol = self.config.getfloat("EOM", "errTol")
        maxIterations = self.config.getint("EOM", "maxIterations")
        pressRelErrTol = self.config.getfloat("EOM", "pressRelErrTol")

        eom = EOM(
            self.boltzmannSolver,
            self.thermodynamics,
            self.hydro,
            self.grid,
            numberOfFields,
            includeOffEq=bIncludeOffEq,
            errTol=errTol,
            maxIterations=maxIterations,
            pressRelErrTol=pressRelErrTol,
        )

        # returning results
        return eom.findWallVelocityMinimizeAction()

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
