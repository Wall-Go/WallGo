
import numpy as np
from abc import ABC, abstractmethod
import scipy.interpolate


class InterpolatableFunction(ABC):
    """ This is a totally-not-overengineered base class for defining optimized functions f(x) that, 
        in addition to normal evaluation, support the following: 
            1) Reading precalculated values from a file and interpolating. 
            Replaces the standard implementation in InterpolatedFunction.__evaluate().
            2) Producing said file for some range of inputs.
            3) Validating that what was read from a file makes sense, ie. matches the result given by __evaluate().

    Currently makes sense only for functions of one variable. 

    WallGo uses this for the thermal Jb, Jf integrals.
    """

    def __init__(self):

        ## Range for which we have precalculated data
        self._originalRange = None
        ## f(x) for x in self.originalRange
        self._originalValues = None
        # we keep these two stored because writeInterpolationTable() needs them
        
        self._interpolatedFunction = None


    @abstractmethod
    def _evaluate(self, x: float) -> float:
        # Override this with the function return value. However the __call__ method should be preferred
        # when using the function as it can make use of our interpolated values
        pass



    """ Non abstracts """
    
    def __call__(self, x: float, useInterpolatedValues=True) -> float:
        
        ## Prefer the interpolated version if we have that, or if otherwise specified
        canUseInterpolatedValues = (self._interpolatedFunction != None) and (x <= self._rangeMax) and (x >= self._rangeMin)

        if (not useInterpolatedValues or not canUseInterpolatedValues):
            return self._evaluate(x)
        else:
            return self._interpolatedFunction(x)
        
    
    def makeInterpolationTable(self, xMin: float, xMax: float, numberOfPoints: int) -> None:

        ## TODO Could make this much smarter
        xValues = np.linspace(xMin, xMax, numberOfPoints)
        fx = []

        for x in xValues:
            fx.append(self._evaluate(x))

        self.__interpolate(xValues, fx)


    ## Internal helper, sets our internal variables and does the actual interpolation. Input is array_like
    def __interpolate(self, xList, fxList) -> None:

        x = np.array(xList)
        fx = np.array(fxList)

        self._interpolatedFunction = scipy.interpolate.CubicSpline(x, fx, extrapolate=False)

        self._rangeMin = np.min(x)
        self._rangeMax = np.max(x)
        self._originalRange = x
        self._originalValues = fx

        

    ## Reads precalculated values and does cubic interpolation. Stores the interpolated funct to self.values
    def readInterpolationTable(self, fileToRead: str):

        # for logging
        selfName = self.__class__.__name__

        try:
            ## Each line should be of form x fx, we read those and interpolate to get fx at any x in between 
            data = np.genfromtxt(fileToRead, delimiter=' ', dtype=float, encoding=None)
            x, fx = zip(*data)

            self.__interpolate(x, fx)

            ## check that what we read matches our function definition (just evaluate and compare at a few values)
            self.__validateInterpolationTable(self._rangeMin)
            self.__validateInterpolationTable(self._rangeMax)
            self.__validateInterpolationTable((self._rangeMax - self._rangeMin) / 2.555)

            print(f"{selfName}: Succesfully read interpolation table from file. Range [{self._rangeMin}, {self._rangeMax}]")

        except IOError as ioError:
            print(f"IOError! {selfName} attempted to read interpolation table from file, but got error:")
            print(ioError)
            print(f"This is non-fatal. Interpolation table will not be updated.\n")



    def writeInterpolationTable(self, outputFileName: str):
        
        print(f"Storing interpolation table for function {self.__class__.__name__}, output file {outputFileName}.")

        try:
            ## Write to file, line i is of form: x[i] fx[i]
            with open(outputFileName, "w") as file:
                for val1, val2 in zip(self._originalRange, self._originalValues):
                    file.write(f"{val1} {val2}\n")

        except Exception as e:
            print(f"Error: {e}")



    ## Test self.values with some input. Result should agree with self.evaluate(x)
    def __validateInterpolationTable(self, x: float, absoluteTolerance: float = 1e-6):
        
        if (self._interpolatedFunction == None or not ( self._rangeMin <= x <= self._rangeMax)):
            print(f"{self.__class__.__name__}: __validateInterpolationTable called, but no valid interpolation table was found.")

        diff = self._interpolatedFunction(x) - self._evaluate(x)
        if (np.abs(diff) > absoluteTolerance):
            print(f"{self.__class__.__name__}: Could not validate interpolation table! Value discrepancy was {diff}")

               
