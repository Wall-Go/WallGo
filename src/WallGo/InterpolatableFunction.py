 
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import scipy.interpolate
from enum import Enum, auto
from typing import Callable

## Enums for extrapolation. Default is NONE, no extrapolation at all. 
class EExtrapolationType(Enum):
    NONE = auto()
    ## Use the boundary value
    CONSTANT = auto()
    ## Extrapolate the interpolated function directly
    FUNCTION = auto()



class InterpolatableFunction(ABC):
    """ This is a totally-not-overengineered base class for defining optimized functions f(x) that, 
        in addition to normal evaluation, support the following: 
            - Producing and using interpolation tables in favor of direct evaluation, where applicable.
            - Automatic adaptive updating of the interpolation table.
            - Reading interpolation tables from a file. 
            - Producing said file for some range of inputs.
            - Validating that what was read from a file makes sense, ie. matches the result given by __evaluate().

    WallGo uses this class for the thermal Jb, Jf integrals and for evaluating the free energy as function of the temperature.

    Works with numpy array input and applying the function element-wise, but it is the user's responsibility to ensure that the 
    implementation of _functionImplementation is compatible with this behavior. 

    This also works for functions returning many numbers, ie. vector functions V(x) = [V1, V2, ...]. 
    In this case each component gets its own interpolation table. Vector functions can also be called with numpy array input.

    Special care is needed if the function evaluation fails for some input x, eg. if the function is evaluated only on some interval.
    In this case it is the user's responsibility to return np.nan from _functionImplementation() for these input values; 
    this will mark these points as invalid and they will not be included in interpolations. Failure to return np.nan for bad input
    will likely break the interpolation.

    Limitations.
     - If the initial interpolation is bad, then it will remain bad: no functionality to improve existing interpolations, only increase of the range is possible. 
     - Currently makes sense only for functions of one variable. 
     - Does NOT support piecewise functions as interpolations would break for those.
    """

    ### Variables for adaptive interpolation
    # This can safely be changed at runtime and adjusted for different functions
    _evaluationsUntilAdaptiveUpdate: int = 500
    __directEvaluateCount: int = 0
    __bUseAdaptiveInterpolation: bool 
    __directlyEvaluatedAt: list ## keep list of values where the function had to be evaluated without interpolation, allows smart updating of ranges

    __interpolatedFunction: Callable

    ## These control out-of-bounds extrapolations. See toggleExtrapolation() function below.
    extrapolationTypeLower: EExtrapolationType; extrapolationTypeUpper: EExtrapolationType

    def __init__(self, bUseAdaptiveInterpolation: bool=True, initialInterpolationPointCount: int=1000, returnValueCount=1):
        """ Optional argument returnValueCount should be set by the user if using list-valued functions.
        """
        ## Vector-like functions can return many values from one input, user needs to specify this when constructing the object
        assert returnValueCount >= 1
        self.__RETURN_VALUE_COUNT = returnValueCount  # TODO deprecate this, seems unnecessary
        
        self.extrapolationTypeLower = EExtrapolationType.NONE
        self.extrapolationTypeUpper = EExtrapolationType.NONE

        if (bUseAdaptiveInterpolation): 
            self.enableAdaptiveInterpolation()
        else:
            self.disableAdaptiveInterpolation()

        ## Range for which we have precalculated data ("x")
        self.__interpolationPoints = []
        ## f(x) for x in self.__interpolationPoints
        self.__interpolationValues = []

        self.__interpolatedFunction = None


        """This specifies how many points are calculated the first time an interpolation table is constructed.
        If the interpolation range is changed later (adaptive interpolation), more points will be added outside the initial table.
        Point spacing is NOT guaranteed to be uniform in adaptive updating."""  
        self._initialInterpolationPointCount = initialInterpolationPointCount



    @abstractmethod
    def _functionImplementation(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Override this with the function return value.
        Do not call this directly, use the __call__ functionality instead. 
        If the function value is invalid for whatever reason, you should return np.nan. 
        This will guarantee that the invalid values are not included in interpolations
        
        The return value can be a scalar, or a list if the function is vector valued. 
        Can also be a numpy array, in which case the function should be applied element-wise. 
        The number of elements returned needs to match self.__RETURN_VALUE_COUNT;
        for numpy array input, list length self.__RETURN_VALUE_COUNT for each x value.
        A list containing np.nan anywhere in the list is interpreted as a failed evaluation, and
        this input x is not included in interpolation
        """
        pass



    """ Non abstracts """

    def interpolationRangeMin(self) -> float:
        """Get lower limit of our current interpolation table."""
        return self.__rangeMin
    
    def interpolationRangeMax(self) -> float:
        """Get upper limit of our current interpolation table."""
        return self.__rangeMax
    
    def numPoints(self):
        """How many input points in our interpolation table."""
        return len(self.__interpolationPoints)


    def hasInterpolation(self) -> bool:
        """Returns true if we have an interpolation table.
        """
        return (self.__interpolatedFunction != None)

    def setExtrapolationType(self, extrapolationTypeLower: EExtrapolationType, extrapolationTypeUpper: EExtrapolationType) -> None:
        """Changes extrapolation behavior, default is NONE. See the enum class EExtrapolationType. 
        NOTE: This will effectively prevent adaptive updates to the interpolation table.
        NOTE 2: Calling this function will force a rebuild of our interpolation table.
        """
        self.extrapolationTypeLower = extrapolationTypeLower
        self.extrapolationTypeUpper = extrapolationTypeUpper

        ## CubicSplines build the extrapolations when initialized, so reconstruct the interpolation here
        if self.__interpolatedFunction:
            self.newInterpolationTableFromValues(self.__interpolationPoints, self.__interpolationValues)




    def enableAdaptiveInterpolation(self) -> None:
        """ Enables adaptive interpolation functionality. 
        Will clear internal work arrays."""
        self.__bUseAdaptiveInterpolation = True
        self.__directEvaluateCount = 0
        self.__directlyEvaluatedAt: list[float] = []


    def disableAdaptiveInterpolation(self) -> None:
        """ Disables adaptive interpolation functionality.
        """
        self.__bUseAdaptiveInterpolation = False



    def newInterpolationTable(self, xMin: float, xMax: float, numberOfPoints: int) -> None:
        """Creates a new interpolation table over given range.
        This will purge any existing interpolation information.
        """

        xValues = np.linspace(xMin, xMax, numberOfPoints)

        fx = self._functionImplementation(xValues)

        self.__interpolate(xValues, fx)



    def newInterpolationTableFromValues(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:
        """Like initializeInterpolationTable but takes in precomputed function values 'fx'
        """
        self.__interpolate(x, fx)


    ## Add x, f(x) pairs to our pending interpolation table update 
    def scheduleForInterpolation(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:

        x = np.asanyarray(x)
        functionValues = np.asanyarray(fx)

        if (np.isscalar(x) or np.ndim(x) == 0):
            # Just got 1 input x
            bValidResult = np.all(np.isfinite(functionValues))

            # put x in array format for consistency with array input
            xValid = np.array([x]) if bValidResult else np.array([])

        else:
            ## Got many input x
            assert len(x) == len(functionValues)

            if (self.__RETURN_VALUE_COUNT > 1):
                validIndices = np.all(np.isfinite(functionValues), axis=1)
            else:
                validIndices = np.all(np.isfinite(functionValues))

            xValid = x[validIndices]

            # Avoid unnecessary nested lists. This flattens to a 1D array, which is fine here since we're just storing x values for later
            xValid = np.ravel(xValid)


        # add x to our internal work list 
        if (np.size(xValid) > 0):

            xValid = np.unique(xValid)

            self.__directEvaluateCount += len(xValid)
            self.__directlyEvaluatedAt = np.concatenate((self.__directlyEvaluatedAt, xValid)) # is this slow?
            
            if (self.__directEvaluateCount >= self._evaluationsUntilAdaptiveUpdate):
                self.__adaptiveInterpolationUpdate()



    def evaluateInterpolation(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Evaluates our interpolated function at input x
        """
        return self.__interpolatedFunction(x)


    def __evaluateOutOfBounds(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """This gets called when the function is called outside the range of its interpolation table.
        We either extrapolate (different extrapolations are possible) or evaluate the function directly based on _functionImplementation() 
        """

        x = np.asanyarray(x)

        """LN: This assert fails with a useless error if the function gets called with np.nan. Inputting np.nan is certainly not good but that's not our fault. 
        So let's not enforce the assert."""
        #assert np.all( (x > self.__rangeMax) | (x < self.__rangeMin))

        bNoExtrapolation = self.extrapolationTypeLower == EExtrapolationType.NONE and self.extrapolationTypeUpper == EExtrapolationType.NONE

        if (not self.__interpolatedFunction or bNoExtrapolation):
            res = self.__evaluateDirectly(x)
        
        else:
            ## Now we have something to extrapolate

            xLower = (x < self.__rangeMin)
            xUpper = (x > self.__rangeMax)
            res = np.empty_like(x)

            ## Lower range
            match self.extrapolationTypeLower:
                case EExtrapolationType.NONE:
                    res[xLower] = self.__evaluateDirectly(x[xLower])
                case EExtrapolationType.CONSTANT:
                    res[xLower] = self.evaluateInterpolation(self.__rangeMin)
                case EExtrapolationType.FUNCTION:
                    res[xLower] = self.evaluateInterpolation(x[xLower])

            ## Upper range
            match self.extrapolationTypeUpper:
                case EExtrapolationType.NONE:
                    res[xUpper] = self.__evaluateDirectly(x[xUpper])
                case EExtrapolationType.CONSTANT:
                    res[xUpper] = self.evaluateInterpolation(self.__rangeMax)
                case EExtrapolationType.FUNCTION:
                    res[xUpper] = self.evaluateInterpolation(x[xUpper])

        return res
    

    def __call__(self, x: npt.ArrayLike, useInterpolatedValues=True) -> npt.ArrayLike:
        
        x = np.asanyarray(x)

        if (not useInterpolatedValues):
            return self.__evaluateDirectly(x)
        elif (self.__interpolatedFunction == None):
            return self.__evaluateDirectly(x)
      
        
        ## np.isscalar does not catch the case when x is np.ndarray of dim 0
        if (np.isscalar(x) or np.ndim(x) == 0):
            canInterpolateCondition = (x <= self.__rangeMax) and (x >= self.__rangeMin)

            if (canInterpolateCondition):
                return self.evaluateInterpolation(x)
            else:
                return self.__evaluateOutOfBounds(x)

        else: 

            ## Now input array of many x values. Use interpolated values whenever possible, so split the x array into two parts.
            ## However, be careful to preserve the array shape
            
            canInterpolateCondition = (x <= self.__rangeMax) & (x >= self.__rangeMin)
            
            needsEvaluationCondition = ~canInterpolateCondition 

            xInterpolateRegion = x[ canInterpolateCondition ] 
            xEvaluateRegion = x[ needsEvaluationCondition ]
            yInterpolateRegion = self.evaluateInterpolation(xInterpolateRegion)
            shape = list(yInterpolateRegion.shape)
            # Make the shape of the results array match
            if len(shape) > len(x.shape):
                for i,s in enumerate(x.shape):
                    shape[i] = s
            else:
                shape = x.shape
            
            results = np.empty(shape)
            results[canInterpolateCondition] = yInterpolateRegion

            if (xEvaluateRegion.size > 0):
                results[needsEvaluationCondition] = self.__evaluateOutOfBounds(xEvaluateRegion)
                
            return results


    def __evaluateDirectly(self, x: npt.ArrayLike, bScheduleForInterpolation=True) -> npt.ArrayLike: 
        """Evaluate the function directly based on _functionImplementation, instead of using interpolations.
        This also accumulates data for the adaptive interpolation functionality which is best kept separate from 
        the abstract _functionImplementation method.
        """
        fx = self._functionImplementation(x)

        if (self.__bUseAdaptiveInterpolation and bScheduleForInterpolation):
            self.scheduleForInterpolation(x, fx)

        return fx 
    


    ## Helper, sets our internal variables and does the actual interpolation
    def __interpolate(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:

        ## Can't specify different extrapolation methods for x > xmax, x < xmin in CubicSpline! This logic is handled manually in __call__()
        bShouldExtrapolate = (self.extrapolationTypeLower == EExtrapolationType.FUNCTION) or (self.extrapolationTypeUpper == EExtrapolationType.FUNCTION)

        ## Explicitly drop non-numerics
        xFiltered, fxFiltered = self.__dropBadPoints(x, fx)
        
        ## This works even if f(x) is vector valued
        self.__interpolatedFunction = scipy.interpolate.CubicSpline(xFiltered, fxFiltered, extrapolate=bShouldExtrapolate, axis=0)

        self.__rangeMin = np.min(xFiltered)
        self.__rangeMax = np.max(xFiltered)
        self.__interpolationPoints = xFiltered
        self.__interpolationValues = fxFiltered

    
    @staticmethod
    def __dropBadPoints(x: npt.ArrayLike, fx: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Removes non-numerical (x, fx) pairs. For 2D fx the check is applied row-wise 
        """
        if fx.ndim > 1:
            validIndices = np.all(np.isfinite(fx), axis=1)
            fxValid = fx[validIndices]
        else:
            ## fx is 1D array
            validIndices = np.all(np.isfinite(fx))
            fxValid = np.ravel( fx[validIndices] )

        xValid = np.ravel( x[validIndices] )

        return xValid, fxValid

        


    def __adaptiveInterpolationUpdate(self) -> None:
        """ Handles interpolation table updates for adaptive interpolation.
        """

        ## Where did the new evaluations happen
        evaluatedPointMin = np.min(self.__directlyEvaluatedAt)
        evaluatedPointMax = np.max(self.__directlyEvaluatedAt)

        # Reset work variables (doing this here already to avoid spaghetti nesting)
        self.__directEvaluateCount = 0
        self.__directlyEvaluatedAt = []

        appendPointCount = 0.2 * self._initialInterpolationPointCount if self.hasInterpolation() else self._initialInterpolationPointCount / 2

        self.extendInterpolationTable(evaluatedPointMin, evaluatedPointMax, appendPointCount, appendPointCount)
            
    
    def extendInterpolationTable(self, newMin: float, newMax: float, pointsMin: int, pointsMax: int) -> None:
        """Extend our interpolation table. 
        NB: This will reset internal data of adaptive interpolation.
        """
        if not self.hasInterpolation():
            newPoints = int(pointsMin + pointsMax)
            print(f"Warning: {self.__class__.__name__}.extendInterpolationRange() called without existing interpolation. "
                  f"Creating new table in range [{newMin}, {newMax}] with {newPoints} points")
            self.newInterpolationTable(newMin, newMax, newPoints)
            return
        
        # what to append to lower end
        if (newMin < self.__rangeMin and pointsMin > 0):
            
            ## Point spacing to use at new lower end 
            spacing = np.abs(self.__rangeMin - newMin) / pointsMin
            # arange stops one spacing before the max value, which is what we want
            appendPointsMin = np.arange(newMin, self.__rangeMin, spacing)
        else:
            appendPointsMin = np.array([])

        # what to append to upper end
        if (newMax > self.__rangeMax and pointsMax > 0):

            ## Point spacing to use at new upper end 
            spacing = np.abs(newMax - self.__rangeMax) / pointsMax
            appendPointsMax = np.arange(self.__rangeMax + spacing, newMax + spacing, spacing)
        else:
            appendPointsMax = np.array([])

        
        appendValuesMin = self._functionImplementation(appendPointsMin)
        appendValuesMax = self._functionImplementation(appendPointsMax)

        # Ordering is important since interpolation needs the x values to be ordered. 
        # This works, but could be made safer by rearranging the resulting arrays accordingly:
        xRange = np.concatenate( (appendPointsMin, self.__interpolationPoints, appendPointsMax) )
        fxRange = np.concatenate( (appendValuesMin, self.__interpolationValues, appendValuesMax) )

        self.newInterpolationTableFromValues(xRange, fxRange)

        ## Hacky reset of adaptive routines
        if (self.__bUseAdaptiveInterpolation):
            self.disableAdaptiveInterpolation()
            self.enableAdaptiveInterpolation()

    # end extendInterpolationTable()
        

    ## Reads precalculated values and does cubic interpolation. Stores the interpolated funct to self.values
    def readInterpolationTable(self, fileToRead: str, bVerbose=True) -> None:

        # for logging
        selfName = self.__class__.__name__

        try:
            ## Each line should be of form x f(x). For vector valued functions, x f1(x) f2(x) ...  
            data = np.genfromtxt(fileToRead, delimiter=' ', dtype=float, encoding=None)

            rows, columns = data.shape

            # now slice this column-wise. First column is x:
            x = data[:, 0]
            # and for fx we remove the first column, using magic syntax 1: to leave all others
            fx = data[:, 1:]

            ## If f(x) is 1D, this actually gives it in messy format [ [fx1] [fx2] ...]. So let's fix that
            if (columns == 2): 
                fx = np.ravel(fx)

            self.__interpolate(x, fx)

            ## check that what we read matches our function definition (just evaluate and compare at a few values)
            self.__validateInterpolationTable(self.__rangeMin)
            self.__validateInterpolationTable(self.__rangeMax)
            self.__validateInterpolationTable((self.__rangeMax - self.__rangeMin) / 2.55)

            if (bVerbose):
                print(f"{selfName}: Succesfully read interpolation table from file. Range [{self.__rangeMin}, {self.__rangeMax}]")

        except IOError as ioError:
            print(f"IOError! {selfName} attempted to read interpolation table from file, but got error:")
            print(ioError)
            print(f"This is non-fatal. Interpolation table will not be updated.\n")



    def writeInterpolationTable(self, outputFileName: str, bVerbose=True) -> None:
        """ Write our interpolation table to file.
        """
        try:
            ## Write to file, line i is of form: x[i] fx[i]. If our function is vector valued then x[i] fx1[i] fx2[i] ...

            stackedArray = np.column_stack((self.__interpolationPoints, self.__interpolationValues))
            np.savetxt(outputFileName, stackedArray, fmt='%.15g', delimiter=' ')

            if (bVerbose):
                print(f"Stored interpolation table for function {self.__class__.__name__}, output file {outputFileName}.")

        except Exception as e:
            print(f"Error from {self.__class__.__name__}, function writeInterpolationTable(): {e}")
    


    ## Test the interpolation table with some input. Result should agree with self.__evaluateDirectly(x)
    def __validateInterpolationTable(self, x: float, absoluteTolerance: float = 1e-6) -> bool:
        
        if (self.__interpolatedFunction == None or not ( self.__rangeMin <= x <= self.__rangeMax)):
            print(f"{self.__class__.__name__}: __validateInterpolationTable called, but no valid interpolation table was found.")
            return False

        diff = self.evaluateInterpolation(x) - self._functionImplementation(x)
        if (np.any(np.abs(diff) > absoluteTolerance)):
            print(f"{self.__class__.__name__}: Could not validate interpolation table! Value discrepancy was {diff}")
            return False
