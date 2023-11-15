
import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod
import scipy.interpolate


class InterpolatableFunction(ABC):
    """ This is a totally-not-overengineered base class for defining optimized functions f(x) that, 
        in addition to normal evaluation, support the following: 
            1) Reading precalculated values from a file and interpolating. 
            Replaces the standard implementation in InterpolatedFunction._functionImplementation().
            2) Producing said file for some range of inputs.
            3) Validating that what was read from a file makes sense, ie. matches the result given by __evaluate().

    NB: Currently makes sense only for functions of one variable. Does NOT support piecewise functions as interpolations would break for those.
    Should work with numpy array input, but only if the implementation of _functionImplementation supports vectorization. 

    This also works for functions returning many numbers, ie. vector functions V(x) = [V1, V2, ...]. 
    In this case each component gets its own interpolation table. Vector functions can also be called with numpy array input,
    in which case the return value should be array of shape (len(V), len(x)), ie. in list notation 
    [ [V1(x1), V2(x1),...], [V1(x1), V2(x2),...]].

    WallGo uses this for the thermal Jb, Jf integrals and for evaluating the free energy as function of the temperature.

    Special care is needed if the function evaluation fails for some input x, eg. if the function is evaluated only on some interval.
    In this case it is the user's responsibility to return np.nan from _functionImplementation() for these input values; 
    this will mark these points as invalid and they will not be included in interpolations. Failure to return np.nan for bad input
    will likely break the interpolation.

    Weakness of the class: if the initial interpolation is bad, then it will remain bad.
    """

    ### Variables for adaptive interpolation
    # This can safely be changed at runtime and adjusted for different functions
    _evaluationsUntilAdaptiveUpdate: int = 500
    __directEvaluateCount: int = 0
    __bUseAdaptiveInterpolation: bool 
    __directlyEvaluatedAt: list ## keep list of values where the function had to be evaluated without interpolation, allows smart updating of ranges


    def __init__(self, bUseAdaptiveInterpolation: bool=True, initialInterpolationPointCount: int=1000, returnValueCount=1):

        ## Vector-like functions can return many values from one input, user needs to specify this when constructing the object
        assert returnValueCount >= 1
        self.__RETURN_VALUE_COUNT = returnValueCount
        

        if (bUseAdaptiveInterpolation): 
            self.enableAdaptiveInterpolation()
        else:
            self.disableAdaptiveInterpolation()

        ## Range for which we have precalculated data ("x")
        self.__interpolationPoints = []
        ## f(x) for x in self.__interpolationPoints
        self.__interpolationValues = []
        
        # this is the result of cubic interpolation
        self.__interpolatedFunction = None


        """This specifies how many points are calculated the first time an interpolation table is constructed.
        If the interpolation range is changed later (adaptive interpolation), more points will be added outside the initial table.
        Point spacing is NOT guaranteed to be uniform in adaptive updating."""  
        self._initialInterpolationPointCount = initialInterpolationPointCount



    @abstractmethod
    def _functionImplementation(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # Override this with the function return value.
        # Do not call this directly, use the __call__ functionality instead. 
        # If the function value is invalid for whatever reason, you should return np.nan. 
        # This will guarantee that the invalid values are not included in interpolations
        # 
        # The return value can be a scalar, or a list if the function is vector valued. 
        # The number of elements returned needs to match self.__RETURN_VALUE_COUNT.
        # A list containing np.nan anywhere in the list is interpreted as a failed evaluation, and
        # this input x is not included in interpolation
        pass



    """ Non abstracts """

    def enableAdaptiveInterpolation(self):
        self.__bUseAdaptiveInterpolation = True
        self.__directEvaluateCount = 0
        self.__directlyEvaluatedAt = []

    def disableAdaptiveInterpolation(self):
        self.__bUseAdaptiveInterpolation = False


    def newInterpolationTable(self, xMin: float, xMax: float, numberOfPoints: int) -> None:

        xValues = np.linspace(xMin, xMax, numberOfPoints)

        fx = self._functionImplementation(xValues)

        self.__interpolate(xValues, fx)



    ## Like initializeInterpolationTable but takes in precomputed function values 'fx'
    def newInterpolationTableFromValues(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:
        self.__interpolate(x, fx)


    ## Add x, f(x) pairs to our pending interpolation table update 
    def scheduleForInterpolation(self, x: npt.ArrayLike, fx: npt.ArrayLike):

        functionValues = np.asanyarray(fx)

        if (np.isscalar(x)):
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

            # Avoid unnecessary nested lists (is this safe?)
            xValid = np.ravel(xValid)


        # add x to our internal work list 
        if (np.size(xValid) > 0):

            xValid = np.unique(xValid)

            self.__directEvaluateCount += len(xValid)
            self.__directlyEvaluatedAt = np.concatenate((self.__directlyEvaluatedAt, xValid)) # is this slow?
            
            if (self.__directEvaluateCount >= self._evaluationsUntilAdaptiveUpdate):
                self.__updateInterpolationTable()



    def __call__(self, x: npt.ArrayLike, useInterpolatedValues=True) -> npt.ArrayLike:
        
        if (not useInterpolatedValues):
            return self.__evaluateDirectly(x)
        elif (self.__interpolatedFunction == None):
            return self.__evaluateDirectly(x)
        
        if (np.isscalar(x)):
            canInterpolateCondition = (x <= self.__rangeMax) and (x >= self.__rangeMin)

            if (not canInterpolateCondition):
                return self.__evaluateDirectly(x)
            else:
                return self.__interpolatedFunction(x)

        else: 

            ## Use interpolated values whenever possible, so split the x array into two parts

            # flatten in case the input x is of weird form [[ ]]
            xRavel = np.ravel(x)
        
            canInterpolateCondition = (xRavel <= self.__rangeMax) & (xRavel >= self.__rangeMin)
            needsEvaluationCondition = ~canInterpolateCondition 

            xInterpolateRegion = xRavel[ canInterpolateCondition ] 
            xEvaluateRegion = xRavel[ needsEvaluationCondition ]

            resultsInterpolated = self.__interpolatedFunction(xInterpolateRegion)

            if (xEvaluateRegion.size == 0):
                results = resultsInterpolated
            else:

                resultsEvaluated = self.__evaluateDirectly(xEvaluateRegion)

                ## combine and put in same row-order as the original x
                results = np.empty_like(xRavel, dtype=float)
                results[canInterpolateCondition] = resultsInterpolated
                results[needsEvaluationCondition] = resultsEvaluated
                
            return results
        

    """Evaluate the function directly based on _functionImplementation, instead of using interpolations.
    This also accumulates data for the adaptive interpolation functionality which is best kept separate from 
    the abstract _functionImplementation method.
    """
    def __evaluateDirectly(self, x: npt.ArrayLike, bScheduleForInterpolation=True) -> npt.ArrayLike:
        
        fx = self._functionImplementation(x)

        if (self.__bUseAdaptiveInterpolation and bScheduleForInterpolation):
            self.scheduleForInterpolation(x, fx)

<<<<<<< HEAD
        if (self.__bUseAdaptiveInterpolation):

            ## For interpolation we don't want to include points where the function is invalid. 
            ## The assumption here is that _functionImplementation() returns np.nan if its evaluation fails

            if (np.isscalar(x)):
                if (np.isfinite(RESULT)):
                    xValid = np.array([x])
                else:
                    xValid = np.array([])
            else:
                # input is array
                
                validIndices = np.where(np.isfinite(RESULT))
                xValid = x[validIndices]

                ## np.concatenate requires arrays of same shape. If the input x is nested then stuff can fail. So do this:
                xValid = np.ravel(xValid)
                ## Note that this will break things if our function is actually a function of many variables...


            if (np.size(xValid) > 0):

                self.__directEvaluateCount += 1
                self.__directlyEvaluatedAt = np.concatenate( [xValid, self.__directlyEvaluatedAt] )
                
                if (self.__directEvaluateCount >= self._evaluationsUntilAdaptiveUpdate):
                    self.__updateInterpolationTable()


        return RESULT 
=======
        return fx 
>>>>>>> GeneralizeModel
    



    ## Helper, sets our internal variables and does the actual interpolation
    def __interpolate(self, x: npt.ArrayLike, fx: npt.ArrayLike) -> None:

<<<<<<< HEAD
        x = np.asanyarray(xList)
        fx = np.asanyarray(fxList)

        """Drop non-number values. Ideally we wouldn't have any to begin with, but this is not guaranteed since
        1. This can be called from public initializeInterpolationTableFromValues with in principle any input
        2. In adaptive routines we've filtered out x values that evaluated to nan, but if the function involves RNG 
        then it's not ruled out that a re-evaluation in updateInterpolationTable() would not give nan
        """
        originalSize = np.size(x)

        validIndices = np.where(np.isfinite(fx))
        x = x[validIndices]
        fx = fx[validIndices]

        if (np.size(x) != originalSize):
            print(f"Warning: {self.__class__.__name__}__interpolate received non-number input. These points will be dropped from interpolation.")

        # Rearrange so that x is strictly increasing; this is required by CubicSpline
        sortedIndices = np.argsort(x)
        x = x[sortedIndices]
        fx = fx[sortedIndices]

        self.__interpolatedFunction = scipy.interpolate.CubicSpline(x, fx, extrapolate=False)
=======
        ## This works even if f(x) is vector valued
        self.__interpolatedFunction = scipy.interpolate.CubicSpline(x, fx, extrapolate=False, axis=0)
>>>>>>> GeneralizeModel

        self.__rangeMin = np.min(x)
        self.__rangeMax = np.max(x)
        self.__interpolationPoints = x
        self.__interpolationValues = fx

        
    def __updateInterpolationTable(self):

        ## Where did the new evaluations happen
        evaluatedPointMin = np.min(self.__directlyEvaluatedAt)
        evaluatedPointMax = np.max(self.__directlyEvaluatedAt)


        # Reset work variables (doing this here already to avoid spaghetti nesting)
        self.__directEvaluateCount = 0
        self.__directlyEvaluatedAt = []

        if (self.__interpolatedFunction == None):
            ## No existing interpolation table, so just make a new one for some range that seems sensible
            # TODO could use a smarter start range here

            self.newInterpolationTable(evaluatedPointMin, evaluatedPointMax, self._initialInterpolationPointCount)
            return 
    
        # Now we already have a table and need to extend it, but let's not recalculate things for the range that we already have

        if (self.__rangeMin == None or self.__rangeMax == None):
            print(f"Error: Bad interpolation range, should not happen! in class {self.__class__.__name__}")
            print("Unable to update interpolation range, expect bad performance")

        else: 

            ## How many points to append to BOTH ends (if applicable)
            appendPointCount = 0.2 * self._initialInterpolationPointCount

            # what to append to lower end
            if (evaluatedPointMin < self.__rangeMin):
                
                ## Point spacing to use at new lower end 
                spacing = np.abs(self.__rangeMin - evaluatedPointMin) / appendPointCount
                # arange stops one spacing before the max value, which is what we want
                appendPointsMin = np.arange(evaluatedPointMin, self.__rangeMin, spacing)
            else:
                appendPointsMin = np.array([])

            # what to append to upper end
            if (evaluatedPointMax > self.__rangeMax):

                ## Point spacing to use at new upper end 
                spacing = np.abs(evaluatedPointMax - self.__rangeMax) / appendPointCount
                appendPointsMax = np.arange(self.__rangeMax + spacing, evaluatedPointMax + spacing, spacing)
            else:
                appendPointsMax = np.array([])

            appendValuesMin = []
            for x in appendPointsMin:
                appendValuesMin.append(self._functionImplementation(x))

            appendValuesMax = []
            for x in appendPointsMax:
                appendValuesMax.append(self._functionImplementation(x))

            # Ordering is important since interpolation needs the x values to be ordered. 
            # This works, but could be made safer by rearranging the resulting arrays accordingly:
            xRange = np.concatenate( [appendPointsMin, self.__interpolationPoints, appendPointsMax] )
            fxRange = np.concatenate( [appendValuesMin, self.__interpolationValues, appendValuesMax] )


            self.__interpolate(xRange, fxRange)


    # end __updateInterpolationTable()


    ## Reads precalculated values and does cubic interpolation. Stores the interpolated funct to self.values
    def readInterpolationTable(self, fileToRead: str):

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

            print(f"{selfName}: Succesfully read interpolation table from file. Range [{self.__rangeMin}, {self.__rangeMax}]")

        except IOError as ioError:
            print(f"IOError! {selfName} attempted to read interpolation table from file, but got error:")
            print(ioError)
            print(f"This is non-fatal. Interpolation table will not be updated.\n")



    def writeInterpolationTable(self, outputFileName: str):
        

        try:
            ## Write to file, line i is of form: x[i] fx[i]. If our function is vector valued then x[i] fx1[i] fx2[i] ...

            stackedArray = np.column_stack((self.__interpolationPoints, self.__interpolationValues))
            np.savetxt(outputFileName, stackedArray, fmt='%.15g', delimiter=' ')

            print(f"Stored interpolation table for function {self.__class__.__name__}, output file {outputFileName}.")

        except Exception as e:
            print(f"Error from {self.__class__.__name__}, function writeInterpolationTable(): {e}")
    


    ## Test self.values with some input. Result should agree with self.evaluate(x)
    def __validateInterpolationTable(self, x: float, absoluteTolerance: float = 1e-6):
        
        if (self.__interpolatedFunction == None or not ( self.__rangeMin <= x <= self.__rangeMax)):
            print(f"{self.__class__.__name__}: __validateInterpolationTable called, but no valid interpolation table was found.")

        diff = self.__interpolatedFunction(x) - self._functionImplementation(x)
        if (np.any(np.abs(diff) > absoluteTolerance)):
            print(f"{self.__class__.__name__}: Could not validate interpolation table! Value discrepancy was {diff}")
