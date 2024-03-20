import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import scipy.interpolate
from enum import Enum, auto
from typing import Callable, Tuple

from . import helpers

## Enums for extrapolation. Default is NONE, no extrapolation at all. 
class EExtrapolationType(Enum):
    ## Throw and error
    ERROR = auto()
    ## Re-evaluate
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

    This also works for functions returning many numbers, ie. vector functions V(x) = [V1, V2, ...]. 
    In this case each component gets its own interpolation table.

    Works with numpy array input and applying the function element-wise, but it is the user's responsibility to ensure that the
    implementation of _functionImplementation is compatible with this behavior.
    The logic is such that if x is an array and idx is a index-tuple for an element in x, then fx[idx] is the value of f(x) at x[idx].
    Note that the shapes of fx and x will NOT match IF f(x) is vector valued. 

    Special care is needed if the function evaluation fails for some input x, eg. if the function is evaluated only on some interval.
    In this case it is the user's responsibility to return np.nan from _functionImplementation() for these input values; 
    this will mark these points as invalid and they will not be included in interpolations. Failure to return np.nan for bad input
    will likely break the interpolation.

    Limitations.
     - If the initial interpolation is bad, then it will remain bad: no functionality to improve existing interpolations, only increase of the range is possible. 
     - Currently makes sense only for functions of one variable. However, you CAN call this with numpy arrays of any shape (see above).
     - Does NOT support piecewise functions as interpolations would break for those.
    """


    def __init__(self, bUseAdaptiveInterpolation: bool=True, initialInterpolationPointCount: int=1000, returnValueCount=1):
        """ Optional argument returnValueCount should be set by the user if using list-valued functions.
        """
        ## Vector-like functions can return many values from one input, user needs to specify this when constructing the object
        assert returnValueCount >= 1
        self.__RETURN_VALUE_COUNT = returnValueCount  # TODO figure out how to do internal logic without requiring this as input
        
        self.__interpolatedFunction: Callable = None

        ## Will hold list of interpolated derivatives, 1st and 2nd derivatives only
        self.__interpolatedDerivatives: list[Callable] = None

        ## These control out-of-bounds extrapolations. See toggleExtrapolation() function below.
        self.extrapolationTypeLower = EExtrapolationType.NONE
        self.extrapolationTypeUpper = EExtrapolationType.NONE

        if (bUseAdaptiveInterpolation): 
            self.enableAdaptiveInterpolation()
        else:
            self.disableAdaptiveInterpolation()

        ### Variables for adaptive interpolation
        # This can safely be changed at runtime and adjusted for different functions
        self._evaluationsUntilAdaptiveUpdate = 500

        ## keep list of values where the function had to be evaluated without interpolation, allows smart updating of ranges
        self.__directEvaluateCount = 0
        self.__directlyEvaluatedAt = []

        ## Range for which we have precalculated data ("x")
        self.__interpolationPoints = []
        ## f(x) for x in self.__interpolationPoints
        self.__interpolationValues = []

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
        fx = np.asanyarray(fx)

        if (np.ndim(x) == 0):
            # Just got 1 input x
            bValidResult = np.all(np.isfinite(fx))

            # put x in array format for consistency with array input
            xValid = np.array([x]) if bValidResult else np.array([])

        else:
            ## Got many input x, keep only x values where f(x) is finite. For vector-valued f(x), keep x where ALL return values are finite

            if (self.__RETURN_VALUE_COUNT > 1):
                assert fx.shape == x.shape + (self.__RETURN_VALUE_COUNT, ), "" \
                    "Incompatable array shapes in scheduleForInterpolation(), should not happen!"
                validIndices = np.all(np.isfinite(fx), axis=-1)
            else:
                assert fx.shape == x.shape, "" \
                    "Incompatable array shapes in scheduleForInterpolation(), should not happen!"
                validIndices = np.all(np.isfinite(fx))

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

        bErrorExtrapolation = self.extrapolationTypeLower == EExtrapolationType.ERROR and self.extrapolationTypeUpper == EExtrapolationType.ERROR
        bNoExtrapolation = self.extrapolationTypeLower == EExtrapolationType.NONE and self.extrapolationTypeUpper == EExtrapolationType.NONE

        if bErrorExtrapolation:
            ## OG: I've added this for cases such as where the extrumum doesn't exist outside some range
            raise ValueError(f"Out of bounds: {x} outside [{self.__rangeMin}, {self.__rangeMax}]")
        elif not self.__interpolatedFunction or bNoExtrapolation:
            res = self.__evaluateDirectly(x)
        else:
            ## Now we have something to extrapolate

            xLower = (x <= self.__rangeMin)
            xUpper = (x >= self.__rangeMax)
            res = np.empty_like(x)

            ## Lower range
            match self.extrapolationTypeLower:
                case EExtrapolationType.ERROR:
                    raise ValueError(f"Out of bounds: {x} < {self.__rangeMin}")
                case EExtrapolationType.NONE:
                    res[xLower] = self.__evaluateDirectly(x[xLower])
                case EExtrapolationType.CONSTANT:
                    res[xLower] = self.evaluateInterpolation(self.__rangeMin)
                case EExtrapolationType.FUNCTION:
                    res[xLower] = self.evaluateInterpolation(x[xLower])

            ## Upper range
            match self.extrapolationTypeUpper:
                case EExtrapolationType.ERROR:
                    raise ValueError(f"Out of bounds: {x} > {self.__rangeMax}")
                case EExtrapolationType.NONE:
                    res[xUpper] = self.__evaluateDirectly(x[xUpper])
                case EExtrapolationType.CONSTANT:
                    res[xUpper] = self.evaluateInterpolation(self.__rangeMax)
                case EExtrapolationType.FUNCTION:
                    res[xUpper] = self.evaluateInterpolation(x[xUpper])

        return res
    

    def __call__(self, x: npt.ArrayLike, bUseInterpolatedValues=True) -> npt.ArrayLike:
        """Just calls evaluate()"""
        return self.evaluate(x, bUseInterpolatedValues)
    

    def evaluate(self, x: npt.ArrayLike, bUseInterpolatedValues=True) -> npt.ArrayLike:
        """"""

        x = np.asanyarray(x)

        if (not bUseInterpolatedValues or not self.hasInterpolation()):
            return self.__evaluateDirectly(x)

        # Use interpolated values whenever possible
        canInterpolateCondition, fxShape = self.__findInterpolatablePoints(x)

        needsEvaluationCondition = ~canInterpolateCondition 

        xInterpolateRegion = x[ canInterpolateCondition ] 
        xEvaluateRegion = x[ needsEvaluationCondition ]

        results = np.empty(fxShape)
        results[canInterpolateCondition] = self.evaluateInterpolation(xInterpolateRegion)

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
    

    def derivative(self, x: npt.ArrayLike, order: int = 1, bUseInterpolation=True, epsilon=1e-16, scale=1.0) -> npt.ArrayLike:
        """Takes derivative of the function at points x. If bUseInterpolation=True, will compute derivatives
        from the interpolated function (if it exists). nth order derivative can be taken with order=n,
        however we only support interpolated derivative of order=1,2 for now.
        epsilon and scale are parameters for the helpers.derivative() routine
        """
        x = np.asanyarray(x)
        if (not bUseInterpolation or not self.hasInterpolation() or order > 2):
            return helpers.derivative(self.__evaluateDirectly, x, n=order)


        # Use interpolated values whenever possible
        canInterpolateCondition, fxShape = self.__findInterpolatablePoints(x)
        needsEvaluationCondition = ~canInterpolateCondition 

        xEvaluateRegion = x[needsEvaluationCondition]

        results = np.empty(fxShape)
        results[canInterpolateCondition] = self.__interpolatedDerivatives[order-1]( x[canInterpolateCondition] )

        ## Outside the interpolation region use whatever extrapolation type the function uses
        if (xEvaluateRegion.size > 0):
            results[needsEvaluationCondition] = helpers.derivative(self.__evaluateOutOfBounds, x, n=order, epsilon=epsilon, scale=scale)
            
        return results

        


    def __findInterpolatablePoints(self, x: npt.ArrayLike) -> Tuple[npt.ArrayLike, Tuple]:
        """Finds x values where interpolation can be used.
        Return tuple is: canInterpolateCondition, fxShape
        where the condition is a numpy bool array and fxShape is the resulting shape of f(x). 
        """

        canInterpolateCondition = (x <= self.__rangeMax) & (x >= self.__rangeMin)
        
        """If x is N-dimensional array and idx is a tuple index for this array,
        we want to return fx so that fx[idx] is the result of function evaluation at x[idx].
        But if f(x) is vector-valued then necessarily fx shape will not match x shape.
        So figure out the shape here. 
        """
        if (self.__RETURN_VALUE_COUNT > 1):
            fxShape = x.shape + (self.__RETURN_VALUE_COUNT, )
        else:
            fxShape = x.shape

        return canInterpolateCondition, fxShape

    ## 
    def __interpolate(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:
        """Does the actual interpolation and sets some internal values.
        Input x needs to be 1D, and input fx needs to be at most 2D.
        """

        x = np.asanyarray(x)
        fx = np.asanyarray(fx)
        assert x.ndim == 1 and fx.ndim <= 2, "Shape error in __interpolate(), this should not happen!"

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

        """Store a cubic spline for the 1st and 2nd derivatives into a list.
        We do not attempt to spline the higher derivatives as they are not 
        guaranteed to be continuous."""
        self.__interpolatedDerivatives = [self.__interpolatedFunction.derivative(1), 
                                          self.__interpolatedFunction.derivative(2)]

    @staticmethod
    def __dropBadPoints(x: npt.ArrayLike, fx: npt.ArrayLike) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """Removes non-numerical (x, fx) pairs. For 2D fx the check is applied row-wise.
        Input x needs to be 1D, and input fx needs to be at most 2D.
        Output is same shape as input.
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
        NB: This will reset internally accumulated data of adaptive interpolation.
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

    def readInterpolationTable(self, fileToRead: str, bVerbose=True) -> None:
        """Reads precalculated values from a file and does cubic interpolation.
        Each line in the file must be of form x f(x).
        For vector valued functions: x f1(x) f2(x)
        """

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
        
        return True
