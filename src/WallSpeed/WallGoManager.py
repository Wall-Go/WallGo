import numpy as np
import cmath # complex numbers

## WallGo imports
from .Particle import Particle
from .EffectivePotential import EffectivePotential
from .GenericModel import GenericModel
from .Thermodynamics import Thermodynamics

""" Defines a 'control' class for managing the program flow """
class WallGoManager:

    ## Very anti-pythonic way of declaring members. But Python must be something like this...?

    model: GenericModel
    # Field values at the two phases at Tn (we go from 1 to 2)
    phaseLocation1: np.ndarray[float]
    phaseLocation2: np.ndarray[float]
    # These are the user specified values, keep stored just in case
    phaseLocation1Input: np.ndarray[float]
    phaseLocation2Input: np.ndarray[float]
    ## TODO we'd probably precalculate phase locations over some sensible temperature range and store them in arrays

    # Nucleation temperature
    Tn: float
    # Critical temperature
    Tc: float

    ### WallGo objects
    thermo: Thermodynamics


    def __init__(self, inputModel, userInput: dict):
        self.model = inputModel

        self.readUserInput(userInput)

        self.initValidate()


    ## WIP/draft function, read things like Tn, approx locations of the 2 minima etc
    def readUserInput(self, userInput: dict) -> None:
        self.phaseLocation1Input = userInput["phaseLocation1"]
        self.phaseLocation2Input = userInput["phaseLocation2"]
        self.Tn = userInput["Tn"]

        

    ## Do stuff like validations and initialization of all other classes here
    def initValidate(self) -> None:

        ## Find the actual minima at Tn, should be close to the user-specified locations
        self.phaseLocation1, VeffValue1 = self.model.Veff.findLocalMinimum(self.phaseLocation1Input, self.Tn)
        self.phaseLocation2, VeffValue2 = self.model.Veff.findLocalMinimum(self.phaseLocation2Input, self.Tn)

        print(f"phase1 Veff: {VeffValue1.real}")
        print(f"phase2 Veff: {VeffValue2.real}")
        
        self.Tc = self.model.Veff.findCriticalTemperature(self.phaseLocation1, self.phaseLocation2, TMin = self.Tn, TMax = 500)

        print(f"Found Tc = {self.Tc} GeV. Transition: ")

        if (self.Tc < self.Tn):
            print("Got Tc < Tn, should not happen!")
