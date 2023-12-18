import numpy as np
import math
from dataclasses import dataclass

## WallGo imports
from .Particle import Particle
from .EffectivePotential import EffectivePotential
from .GenericModel import GenericModel
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro # why is this not Hydrodynamics? compare with Thermodynamics
from .HydroTemplateModel import HydroTemplateModel
from .EOM import EOM
from .Grid import Grid
from .Config import Config
from .Integrals import Integrals

from .WallGoUtils import getSafePathToResource


@dataclass
class PhaseInfo:
    # Field values at the two phases at T (we go from 1 to 2)
    phaseLocation1: np.ndarray[float]
    phaseLocation2: np.ndarray[float]
    temperature: float

    

""" Defines a 'control' class for managing the program flow.
This should be better than writing the same stuff in every example main function, 
and is good for hiding some of our internal implementation details from the user """
class WallGoManager:

    # Critical temperature
    Tc: float

    ### WallGo objects
    config: Config
    integrals: Integrals ## use a dedicated Integrals object to make management of interpolations easier
    model: GenericModel
    thermodynamics: Thermodynamics
    hydro: Hydro
    grid: Grid
    eom: EOM
    # ...

    def __init__(self):
        """do common model-independent setup here
        """

        self.config = Config()
        self.config.readINI( getSafePathToResource("Config/WallGoDefaults.ini") )

        self.integrals = Integrals()

        self._initalizeIntegralInterpolations(self.integrals)

        ## Grid
        self._initGrid( self.config.getint("PolynomialGrid", "spatialGridSize"), 
                        self.config.getint("PolynomialGrid", "momentumGridSize"),
                        self.config.getfloat("PolynomialGrid", "L_xi")
        )


    def registerModel(self, model: GenericModel) -> None:
        """Register a physics model with WallGo.
        """
        self.model = model


    def setParameters(self, modelParameters: dict[str, float], phaseInput: PhaseInfo) -> None:
        """Parameters
        ----------
        modelParameters: dict[str, float]
                        Dict containing all QFT model parameters: 
                        Those that enter the action and the renormalization scale.
        phaseInput: WallGo.PhaseInfo
                    Should contain approximate field values at the two phases that WallGo will analyze,
                    and the nucleation temperature. Transition is assumed to go phaseLocation1 --> phaseLocation2.
        """

        ## LN: this routine is probably too heavy

        self.model.modelParameters = modelParameters

        ## Checks that phase input makes sense with the user-specified Veff
        self.validatePhaseInput(phaseInput)

        """ Find critical temperature. Do we even need to do this though?? """

        ## TODO!! upper temperature here
        self.Tc = self.model.effectivePotential.findCriticalTemperature(self.phasesAtTn.phaseLocation1, self.phasesAtTn.phaseLocation2, 
                                                                        TMin = self.phasesAtTn.temperature, TMax = 10 * self.phasesAtTn.temperature)

        print(f"Found Tc = {self.Tc} GeV.")
        # @todo should check that this Tc is really for the transition between the correct phases. 
        # At the very least print the field values for the user

        if (self.Tc < self.phasesAtTn.temperature):
            raise RuntimeError(f"Got Tc < Tn, should not happen! Tn = {self.phasesAtTn.temperature}, Tc = {self.Tc}")


        self.thermodynamics = Thermodynamics(self.model.effectivePotential, self.Tc, phaseInput.temperature, 
                                            phaseInput.phaseLocation2, phaseInput.phaseLocation1)

        ## Let's turn these off so that things are more transparent
        self.thermodynamics.freeEnergyHigh.disableAdaptiveInterpolation()
        self.thermodynamics.freeEnergyLow.disableAdaptiveInterpolation()

        ## Use the template model to find an estimate of the minimum and maximum required temperature
        self.hydrotemplate = HydroTemplateModel(self.thermodynamics)
        _,_,_,TminTemplate = self.hydrotemplate.findMatching(0.01) # Minimum temperature is obtained by Tm of a really slow wall
        _,_,TmaxTemplate,_ = self.hydrotemplate.findMatching(self.hydrotemplate.vJ) # Maximum temperature is obtained by Tp of the fastes possible wall (Jouguet velocity)


        """ TEMPORARY. Interpolate FreeEnergy between T = [0, 1.2*Tc]. This is here because the old model example does this.
        But this will need to be done properly in the near future, using the temperatures from HydroTemplateModel.
        """

        TMin, TMax, dT = 0.0, 1.2*self.thermodynamics.Tc, 1.0
        interpolationPointCount = math.ceil((TMax - TMin) / dT)

        self.thermodynamics.freeEnergyHigh.newInterpolationTable(TMin, TMax, interpolationPointCount)
        self.thermodynamics.freeEnergyLow.newInterpolationTable(TMin, TMax, interpolationPointCount)

        """LN: Giving sensible temperature ranges to Hydro seems to be very important. 
        I propose hydro routines be changed so that we have easy control over what temperatures are used."""

        self.initHydro(self.thermodynamics, TMin, TMax)

        print(f"Jouguet: {self.hydro.vJ}")


    def validatePhaseInput(self, phaseInput: PhaseInfo) -> None:
        """This checks that the user-specified phases are OK.
        Specifically, the effective potential should have two minima at the given T,
        otherwise phase transition analysis is not possible.
        """

        T = phaseInput.temperature

        ## Find the actual minima at input T, should be close to the user-specified locations
        phaseLocation1, VeffValue1 = self.model.effectivePotential.findLocalMinimum(phaseInput.phaseLocation1, T)
        phaseLocation2, VeffValue2 = self.model.effectivePotential.findLocalMinimum(phaseInput.phaseLocation2, T)

        print(f"Found phase 1: phi = {phaseLocation1}, Veff(phi) = {VeffValue1}")
        print(f"Found phase 2: phi = {phaseLocation2}, Veff(phi) = {VeffValue2}")

        foundPhaseInfo = PhaseInfo(temperature=T, phaseLocation1=phaseLocation1, phaseLocation2=phaseLocation2)

        self.phasesAtTn = foundPhaseInfo
        

    def initHydro(self, thermodynamics: Thermodynamics, TMinGuess: float, TMaxGuess: float) -> None:
        self.hydro = Hydro(thermodynamics, TminGuess=TMinGuess, TmaxGuess=TMaxGuess)



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

        N, M = int(N), int(M)
        if (N % 2 == 0):
            raise ValueError("You have chosen an even number N of momentum-grid points. WallGo only works with odd N, please change it to an odd number.")

        ## TODO remove the temperature from here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.grid = Grid(M, N, L_xi, 100)
        


    # Call after initGrid. I guess this would be the main workload function 
    def solveWall(self):

        ## Begin with Local Thermal Equilibrium approximation to wall speed
        vwLTE = self.hydro.findvwLTE()

        print(f"LTE wall speed: {vwLTE}")

        print("======= Next would be EOM routines, but these are TODO. The program will now crash :^)")
        input()

        numberOfFields = self.model.fieldCount

        ### EOMs just for the first out-of-eq particle for now. TODO generalize
        outOfEqParticle = self.model.outOfEquilibriumParticles[0]
        

        # Without out-of-equilibrium contributions
        eom = EOM(outOfEqParticle, self.thermodynamics, self.hydro, self.grid, numberOfFields)
        #eomGeneral = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2)

        eom.findWallVelocityMinimizeAction()

        ## TODO should not need to create a new object just for this...

        #With out-of-equilibrium contributions
        eomOffEq = EOM(outOfEqParticle, self.thermodynamics, self.hydro, self.grid, numberOfFields, includeOffEq=True)
        #eomGeneralOffEq = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2, True)
        
        
    def _initalizeIntegralInterpolations(self, integrals: Integrals) -> None:
    
        assert self.config != None

        integrals.Jb.readInterpolationTable(
            getSafePathToResource(self.config.get("DataFiles", "InterpolationTable_Jb")), bVerbose=False 
            )
        integrals.Jf.readInterpolationTable(
            getSafePathToResource(self.config.get("DataFiles", "InterpolationTable_Jf")), bVerbose=False 
            )
