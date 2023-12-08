import numpy as np
import math

## WallGo imports
from .Particle import Particle
from .EffectivePotential import EffectivePotential
from .GenericModel import GenericModel
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro # why is this not Hydrodynamics? compare with Thermodynamics
from .HydroTemplateModel import HydroTemplateModel
from .EOM import EOM
from .Grid import Grid

""" Defines a 'control' class for managing the program flow.
This should be better than writing the same stuff in every example main function, 
and is good for hiding some of our internal implementation details from the user """
class WallGoManager:

    # Field values at the two phases at Tn (we go from 1 to 2)
    phaseLocation1: np.ndarray[float]   # high-T
    phaseLocation2: np.ndarray[float]   # low-T

    # These are the user specified values, keep stored just in case
    phaseLocation1Input: np.ndarray[float]
    phaseLocation2Input: np.ndarray[float]

    ## TODO we'd probably precalculate phase locations over some sensible temperature range and store them in arrays

    # Nucleation temperature
    Tn: float
    # Critical temperature
    Tc: float

    ### WallGo objects
    model: GenericModel
    thermodynamics: Thermodynamics
    hydro: Hydro
    grid: Grid
    # ...

    def __init__(self, inputModel: GenericModel, userInput: dict):
        self.model = inputModel

        self.readUserInput(userInput)

        ## Validates model stuff, including Veff and that Tn < Tc
        self.validateUserInput()

        self.thermodynamics = Thermodynamics(self.model.effectivePotential, self.Tc, self.Tn, self.phaseLocation2, self.phaseLocation1)

        ## Let's turn these off so that things are more transparent
        self.thermodynamics.freeEnergyHigh.disableAdaptiveInterpolation()
        self.thermodynamics.freeEnergyLow.disableAdaptiveInterpolation()

        ## Use the template model to find an estimate of the minimum and maximum required temperature
        self.hydrotemplate = HydroTemplateModel(self.thermodynamics)
        _,_,_,TminTemplate = self.hydrotemplate(findMatching(0.01)) #Minimum temperature is obtained by Tm of a really slow wall
        _,_,TmaxTemplate,_ = self.hydrotemplate(findmatching(hydrotemplate.vJ)) #Maximum temperature is obtained by Tp of the fastes possible wall (Jouguet velocity)

        """ TEMPORARY. Interpolate minima between T = [0, 1.2*Tc]. This is here because the old model example does this.
        But this will need to be done properly in the near future.
        """

        TMin, TMax, dT = 0.0, 1.2*self.thermodynamics.Tc, 1.0
        interpolationPointCount = math.ceil((TMax - TMin) / dT)

        self.thermodynamics.freeEnergyHigh.newInterpolationTable(TMin, TMax, interpolationPointCount)
        self.thermodynamics.freeEnergyLow.newInterpolationTable(TMin, TMax, interpolationPointCount)

        self.initHydro(self.thermodynamics)

        ## I think this is where we'd validate/init collision integrals and then end the __init__
        # Can have a separate function for doing the collision/EOM work
        # But until everything works I'm just putting test stuff here: 

        print(f"Jouguet: {self.hydro.vJ}")

    # end __init__


    ## WIP/draft function, read things like Tn, approx locations of the 2 minima etc
    def readUserInput(self, userInput: dict) -> None:
        self.phaseLocation1Input = userInput["phaseLocation1"]
        self.phaseLocation2Input = userInput["phaseLocation2"]
        self.Tn = userInput["Tn"]

        

    ## Check that the user input makes sense in context of the specified model. IE. can be found, Tn < Tc etc
    def validateUserInput(self) -> None:

        ## Find the actual minima at Tn, should be close to the user-specified locations
        self.phaseLocation1, VeffValue1 = self.model.effectivePotential.findLocalMinimum(self.phaseLocation1Input, self.Tn)
        self.phaseLocation2, VeffValue2 = self.model.effectivePotential.findLocalMinimum(self.phaseLocation2Input, self.Tn)

        print(f"Found phase 1: phi = {self.phaseLocation1}, Veff(phi) = {VeffValue1}")
        print(f"Found phase 2: phi = {self.phaseLocation2}, Veff(phi) = {VeffValue2}")


        self.Tc = self.model.effectivePotential.findCriticalTemperature(self.phaseLocation1, self.phaseLocation2, TMin = self.Tn, TMax = 500)

        print(f"Found Tc = {self.Tc} GeV.")
        # @todo should check that this Tc is really for the transition between the correct phases. 
        # At the very least print the field values for the user

        if (self.Tc < self.Tn):
            print("Got Tc < Tn, should not happen!")
            raise RuntimeError("Invalid nucleation temperature or critical temperature")



    def initHydro(self, thermodynamics: Thermodynamics) -> None:
        self.hydro = Hydro(thermodynamics)



    def initGrid(self, M: int, N: int) -> Grid:
        r"""
        Parameters
        ----------
        M : int
            Number of basis functions in the :math:`\xi` (and :math:`\chi`)
            direction.
        N : int
            Number of basis functions in the :math:`p_z` and :math:`p_\Vert`
            (and :math:`\rho_z` and :math:`\rho_\Vert`) directions.
        """

        ## LN: What are these magic numbers? Why does this need the temperature??
        self.grid = Grid(M, N, 0.05, 100)
        


    # Call after initGrid. I guess this would be the main workload function 
    def solveWall(self):

        # this should be somewhere else, if even needed. Maybe routines that need this could just read it from length of some input field array?
        numberOfFields = 2

        ### EOMs just for the first out-of-eq particle for now. TODO generalize
        outOfEqParticle = self.model.outOfEquilibriumParticles[0]
        
        ### LN: I feel this is bad use of objects, and in this case extremely unperformant. 
        # We should only need to initialize one EOM object, then call something like eom.wallSpeedLTE(), eom.wallSpeedBoltzmann() etc

        # Without out-of-equilibrium contributions
        eom = EOM(outOfEqParticle, self.thermodynamics, self.hydro, self.grid, numberOfFields)
        #eomGeneral = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2)
        print(f"LTE wall speed: {eom.wallVelocityLTE}")
        

        #With out-of-equilibrium contributions
        eomOffEq = EOM(outOfEqParticle, self.thermodynamics, self.hydro, self.grid, numberOfFields, includeOffEq=True)
        print(f"LTE wall speed but with includeOffEq=True in EOM: {eomOffEq.wallVelocityLTE}")
        #eomGeneralOffEq = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2, True)
        
