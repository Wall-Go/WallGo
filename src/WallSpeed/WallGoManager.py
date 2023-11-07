import numpy as np
import cmath # complex numbers

## WallGo imports
from .Particle import Particle
from .EffectivePotential import EffectivePotential
from .GenericModel import GenericModel
from .Thermodynamics import Thermodynamics
from .Hydro import Hydro # why is this not Hydrodynamics? compare with Thermodynamics
from .EOM import EOM
from .Grid import Grid

""" Defines a 'control' class for managing the program flow.
This should be better than writing the same stuff in every example main function, 
and is good for hiding some of our internal implementation details from the user """
class WallGoManager:

    ## Very anti-pythonic way of declaring members. But Python must be something like this...?

    model: GenericModel

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

    ### WallGo objects. do we really need all of these?
    #freeEnergy: FreeEnergy # I kinda feel this is redundant (effective potential)
    thermo: Thermodynamics
    hydro: Hydro
    grid: Grid
    # ...

    def __init__(self, inputModel, userInput: dict):
        self.model = inputModel

        self.readUserInput(userInput)

        ## Validates model stuff, including Veff and that Tn < Tc
        self.initValidate()

        thermodynamics = Thermodynamics(self.model.effectivePotential, self.Tc, self.Tn, self.phaseLocation2, self.phaseLocation1)

        hydro = Hydro(thermodynamics)

        ## I think this is where we'd validate/init collision integrals and then end the __init__
        # Can have a separate function for doing the collision/EOM work
        # But until everything works I'm just putting test stuff here: 

        # Wrong number, probably because pressure from Veff is wrong?
        print(f"Jouguet: {hydro.vJ}")




        """
        Grid (put this somewhere else)
        """
        M = 20
        N = 20
        grid = Grid(M, N, 0.05, 100)
        # ???
        #poly = Polynomial(grid)


        # this should be somewhere else, if even needed. Maybe routines that need this could just read it from length of some input field array?
        numberOfFields = 2

        ### EOMs just for the first out-of-eq particle for now. TODO generalize
        outOfEqParticle = self.model.outOfEquilibriumParticles[0]
        
        # Without out-of-equilibrium contributions
        eom = EOM(outOfEqParticle, thermodynamics, hydro, grid, numberOfFields)
        #eomGeneral = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2)

        """
        #With out-of-equilibrium contributions
        eomOffEq = EOM(offEqParticles[0], fxSM, grid, 2, True)
        eomGeneralOffEq = EOMGeneralShape(offEqParticles[0], fxSM, grid, 2, True)
        """



    ## WIP/draft function, read things like Tn, approx locations of the 2 minima etc
    def readUserInput(self, userInput: dict) -> None:
        self.phaseLocation1Input = userInput["phaseLocation1"]
        self.phaseLocation2Input = userInput["phaseLocation2"]
        self.Tn = userInput["Tn"]

        

    ## Do stuff like validations and initialization of all other classes here
    def initValidate(self) -> None:

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

        
