"""
Class that can be use to evaluate and interpolate functions.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable, Tuple
import logging
import numpy as np
from scipy.interpolate import CubicSpline

from . import helpers

inputType = list[float] | np.ndarray
outputType = list[float | np.ndarray] | np.ndarray


class EExtrapolationType(Enum):
    """
    Enums for extrapolation. Default is NONE, no extrapolation at all.
    """

    ## Throw ValueError if attempting to evaluate out of bounds
    ERROR = auto()
    ## Re-evaluate
    NONE = auto()
    ## Use the boundary value
    CONSTANT = auto()
    ## Extrapolate the interpolated function directly
    FUNCTION = auto()


class InterpolatableFunction(ABC):
    r"""
    This is a totally-not-overengineered base class for defining optimized functions
    :math:`f(x)` that, in addition to normal evaluation, support the following:

    - Producing and using interpolation tables in favor of direct evaluation, where
        applicable.
    - Automatic adaptive updating of the interpolation table.
    - Reading interpolation tables from a file.
    - Producing said file for some range of inputs.
    - Validating that what was read from a file makes sense, ie. matches the result
        given by :py:meth:`_evaluate()`.

    WallGo uses this class for evaluating the free energy as function of the
    temperature. It can also be used for the thermal :math:`J_b, J_f` integrals.

    This also works for functions returning many numbers, ie. vector functions
    :math:`V(x) = [V1, V2, ...]`. In this case each component gets its own interpolation table.

    Works with numpy array input and applying the function element-wise, but it is the
    user's responsibility to ensure that the implementation of :py:meth:`_functionImplementation`
    is compatible with this behavior. The logic is such that if x is an array and idx is
    a index-tuple for an element in `x`, then `fx[idx]` is the value of `f(x)` at `x[idx]`. Note
    that the shapes of `fx` and `x` will NOT match IF `f(x)` is vector valued.

    Special care is needed if the function evaluation fails for some input `x`, eg. if the
    function is evaluated only on some interval. In this case it is the user's
    responsibility to return np.nan from _functionImplementation() for these input
    values; this will mark these points as invalid and they will not be included in
    interpolations. Failure to return np.nan for bad input will likely break the
    interpolation.

    Limitations.
     - If the initial interpolation is bad, then it will remain bad: no functionality to
       improve existing interpolations, only increase of the range is possible.
     - Currently makes sense only for functions of one variable. However, you CAN call
       this with numpy arrays of any shape (see above).
     - Does NOT support piecewise functions as interpolations would break for those.
    """

    def __init__(
        self,
        bUseAdaptiveInterpolation: bool = True,
        initialInterpolationPointCount: int = 1000,
        returnValueCount: int = 1,
    ) -> None:
        """
        Optional argument returnValueCount should be set by the user if using
        list-valued functions.

        Parameters
        ----------
        bUseAdaptiveInterpolation : bool, optional
            Whether or not to use adaptive interpolation. The default is True.
        initialInterpolationPointCount : int, optional
            Initial number of points for the interpolation. The default is 1000.
        returnValueCount : int, optional
            Number of outputs returned by the function. The default is 1.

        """
        ## Vector-like functions can return many values from one input, user needs to
        ## specify this when constructing the object
        assert returnValueCount >= 1
        self._RETURN_VALUE_COUNT = returnValueCount  # pylint: disable=invalid-name

        self._interpolatedFunction: CubicSpline

        ## Will hold list of interpolated derivatives, 1st and 2nd derivatives only
        self._interpolatedDerivatives: list[Callable]

        ## These control out-of-bounds extrapolations.
        ## See toggleExtrapolation() function below.
        self.extrapolationTypeLower = EExtrapolationType.NONE
        self.extrapolationTypeUpper = EExtrapolationType.NONE

        if bUseAdaptiveInterpolation:
            self.enableAdaptiveInterpolation()
        else:
            self.disableAdaptiveInterpolation()

        ### Variables for adaptive interpolation
        # This can safely be changed at runtime and adjusted for different functions
        self._evaluationsUntilAdaptiveUpdate = 500

        ## keep list of values where the function had to be evaluated without
        ## interpolation, allows smart updating of ranges
        self._directEvaluateCount = 0
        self._directlyEvaluatedAt: inputType = []

        ## Range for which we have precalculated data ("x")
        self._interpolationPoints: inputType = []
        ## f(x) for x in self._interpolationPoints
        self._interpolationValues: outputType = []

        """This specifies how many points are calculated the first time an interpolation
        table is constructed. If the interpolation range is changed later
        (adaptive interpolation), more points will be added outside the initial table.
        Point spacing is NOT guaranteed to be uniform in adaptive updating."""
        self._initialInterpolationPointCount = initialInterpolationPointCount

        self._rangeMin: float
        self._rangeMax: float

    @abstractmethod
    def _functionImplementation(self, x: inputType | float) -> outputType:
        """
        Override this with the function return value.
        Do not call this directly, use the __call__ functionality instead.
        If the function value is invalid for whatever reason, you should return np.nan.
        This will guarantee that the invalid values are not included in interpolations

        The return value can be a scalar, or a list if the function is vector valued.
        Can also be a numpy array, in which case the function should be applied
        element-wise. The number of elements returned needs to match
        self._RETURN_VALUE_COUNT; for numpy array input, list length
        self._RETURN_VALUE_COUNT for each x value. A list containing np.nan anywhere in
        the list is interpreted as a failed evaluation, and this input x is not included
        in interpolation
        """

    """ Non abstracts """

    def interpolationRangeMin(self) -> float:
        """Get lower limit of our current interpolation table."""
        return self._rangeMin

    def interpolationRangeMax(self) -> float:
        """Get upper limit of our current interpolation table."""
        return self._rangeMax

    def numPoints(self) -> int:
        """How many input points in our interpolation table."""
        return len(self._interpolationPoints)

    def hasInterpolation(self) -> bool:
        """Returns true if we have an interpolation table."""
        return hasattr(self, "_interpolatedFunction")
        # return self._interpolatedFunction is not None

    def setExtrapolationType(
        self,
        extrapolationTypeLower: EExtrapolationType,
        extrapolationTypeUpper: EExtrapolationType,
    ) -> None:
        """
        Changes extrapolation behavior, default is NONE. See the enum class
        EExtrapolationType.
        NOTE: This will effectively prevent adaptive updates to the interpolation table.
        NOTE 2: Calling this function will force a rebuild of our interpolation table.

        Parameters
        ----------
        extrapolationTypeLower : EExtrapolationType
            Extrapolation type when below the interpolation range.
        extrapolationTypeUpper : EExtrapolationType
            Extrapolation type when above the interpolation range.

        """
        self.extrapolationTypeLower = extrapolationTypeLower
        self.extrapolationTypeUpper = extrapolationTypeUpper

        ## CubicSplines build the extrapolations when initialized,
        ## so reconstruct the interpolation here
        if self.hasInterpolation():
            self.newInterpolationTableFromValues(
                self._interpolationPoints, self._interpolationValues
            )

    def enableAdaptiveInterpolation(self) -> None:
        """
        Enables adaptive interpolation functionality.
        Will clear internal work arrays.
        """
        self._bUseAdaptiveInterpolation = True
        self._directEvaluateCount = 0
        self._directlyEvaluatedAt = []

    def disableAdaptiveInterpolation(self) -> None:
        """Disables adaptive interpolation functionality."""
        self._bUseAdaptiveInterpolation = False

    def newInterpolationTable(
        self,
        xMin: float,
        xMax: float,
        numberOfPoints: int,
    ) -> None:
        """
        Creates a new interpolation table over given range.
        This will purge any existing interpolation information.

        Parameters
        ----------
        xMin : float
            Minimal interpolation point.
        xMax : float
            Maximal interpolation point.
        numberOfPoints : int
            Number of points to use in the interpolation.

        """

        xValues = np.linspace(xMin, xMax, numberOfPoints)

        fx = self._functionImplementation(xValues)

        self._interpolate(xValues, fx)

    def newInterpolationTableFromValues(
        self,
        x: inputType,
        fx: outputType,
    ) -> None:
        """
        Like initializeInterpolationTable but takes in precomputed function values 'fx'

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the function was evaluated.
        fx : list[float | np.ndarray] or np.ndarray
            Value of the function at x.

        """
        self._interpolate(x, fx)

    def scheduleForInterpolation(self, x: inputType, fx: outputType) -> None:
        """
        Add x, f(x) pairs to our pending interpolation table update

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the function was evaluated.
        fx : list[float | np.ndarray] or np.ndarray
            Value of the function at x.

        """

        x = np.asanyarray(x)
        fx = np.asanyarray(fx)

        if np.ndim(x) == 0:
            # Just got 1 input x
            bValidResult = np.all(np.isfinite(fx))

            # put x in array format for consistency with array input
            xValid = np.array([x]) if bValidResult else np.array([])

        else:
            ## Got many input x, keep only x values where f(x) is finite.
            ## For vector-valued f(x), keep x where ALL return values are finite

            if self._RETURN_VALUE_COUNT > 1:
                assert fx.shape == x.shape + (self._RETURN_VALUE_COUNT,), (
                    ""
                    "Incompatable array shapes in scheduleForInterpolation(), "
                    "should not happen!"
                )
                validIndices = np.all(np.isfinite(fx), axis=-1)
            else:
                assert fx.shape == x.shape, (
                    ""
                    "Incompatable array shapes in scheduleForInterpolation(), "
                    "should not happen!"
                )
                validIndices = np.all(np.isfinite(fx))

            xValid = x[validIndices]

            # Avoid unnecessary nested lists. This flattens to a 1D array,
            ## which is fine here since we're just storing x values for later
            xValid = np.ravel(xValid)

        # add x to our internal work list
        if np.size(xValid) > 0:

            xValid = np.unique(xValid)

            self._directEvaluateCount += len(xValid)
            self._directlyEvaluatedAt = np.concatenate(
                (self._directlyEvaluatedAt, xValid)
            )

            if self._directEvaluateCount >= self._evaluationsUntilAdaptiveUpdate:
                self._adaptiveInterpolationUpdate()

    def evaluateInterpolation(self, x: inputType | float) -> np.ndarray:
        """Evaluates our interpolated function at input x"""
        return np.asarray(self._interpolatedFunction(x))

    def _evaluateOutOfBounds(self, x: inputType) -> outputType:
        """
        This gets called when the function is called outside the range of its
        interpolation table. We either extrapolate (different extrapolations are
        possible) or evaluate the function directly based on _functionImplementation().
        """

        x = np.asanyarray(x)

        bErrorExtrapolation = (
            self.extrapolationTypeLower == EExtrapolationType.ERROR
            and self.extrapolationTypeUpper == EExtrapolationType.ERROR
        )
        bNoExtrapolation = (
            self.extrapolationTypeLower == EExtrapolationType.NONE
            and self.extrapolationTypeUpper == EExtrapolationType.NONE
        )

        if bErrorExtrapolation:
            raise ValueError(
                f"Out of bounds: {x} outside [{self._rangeMin}, {self._rangeMax}]"
            )
        if not self.hasInterpolation() or bNoExtrapolation:
            res = self._evaluateDirectly(x)
        else:
            ## Now we have something to extrapolate

            xLower = x <= self._rangeMin
            xUpper = x >= self._rangeMax

            # Figure out shape of the result. If we are vector valued, need an extra axis
            if self._RETURN_VALUE_COUNT > 1:
                resShape = x.shape + (self._RETURN_VALUE_COUNT,)
            else:
                resShape = x.shape
            res = np.empty(resShape)

            ## Lower range
            if np.any(xLower):
                match self.extrapolationTypeLower:
                    case EExtrapolationType.ERROR:
                        # TODO better error message, this is nonsensible if x is array or list
                        raise ValueError(f"Out of bounds: {x} < {self._rangeMin}")
                    case EExtrapolationType.NONE:
                        res[xLower, :] = self._evaluateDirectly(x[xLower])
                    case EExtrapolationType.CONSTANT:
                        res[xLower, :] = self.evaluateInterpolation(self._rangeMin)
                    case EExtrapolationType.FUNCTION:
                        res[xLower, :] = self.evaluateInterpolation(x[xLower])

            ## Upper range
            if np.any(xUpper):
                match self.extrapolationTypeUpper:
                    case EExtrapolationType.ERROR:
                        # TODO better error message, this is nonsensible if x is array or list
                        raise ValueError(f"Out of bounds: {x} > {self._rangeMax}")
                    case EExtrapolationType.NONE:
                        res[xUpper, :] = self._evaluateDirectly(x[xUpper])
                    case EExtrapolationType.CONSTANT:
                        res[xUpper, :] = self.evaluateInterpolation(self._rangeMax)
                    case EExtrapolationType.FUNCTION:
                        res[xUpper, :] = self.evaluateInterpolation(x[xUpper])

        return res

    def __call__(self, x: inputType, bUseInterpolatedValues: bool = True) -> outputType:
        """
        Just calls evaluate()

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the function will be evaluated.
        bUseInterpolatedValues : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the function at x.

        """
        return self.evaluate(x, bUseInterpolatedValues)

    def evaluate(self, x: inputType, bUseInterpolatedValues: bool = True) -> outputType:
        """
        Evaluate the function.

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the function will be evaluated.
        bUseInterpolatedValues : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the function at x.

        """

        x = np.asanyarray(x)

        if not bUseInterpolatedValues or not self.hasInterpolation():
            return self._evaluateDirectly(x)

        # Use interpolated values whenever possible
        canInterpolateCondition, fxShape = self._findInterpolatablePoints(x)

        needsEvaluationCondition = ~canInterpolateCondition

        xInterpolateRegion = x[canInterpolateCondition]
        xEvaluateRegion = x[needsEvaluationCondition]

        results = np.empty(fxShape)
        results[canInterpolateCondition] = self.evaluateInterpolation(
            xInterpolateRegion
        )

        if xEvaluateRegion.size > 0:
            results[needsEvaluationCondition] = self._evaluateOutOfBounds(
                xEvaluateRegion
            )

        return results

    def _evaluateDirectly(
        self,
        x: inputType,
        bScheduleForInterpolation: bool = True,
    ) -> outputType:
        """
        Evaluate the function directly based on _functionImplementation, instead of
        using interpolations. This also accumulates data for the adaptive interpolation
        functionality which is best kept separate from the abstract
        _functionImplementation method.
        """
        fx = self._functionImplementation(x)

        if self._bUseAdaptiveInterpolation and bScheduleForInterpolation:
            self.scheduleForInterpolation(x, fx)

        return fx

    def derivative(
        self,
        x: inputType,
        order: int = 1,
        bUseInterpolation: bool = True,
        epsilon: float = 1e-16,
        scale: float = 1.0,
    ) -> outputType:
        """
        Takes derivative of the function at points x. If bUseInterpolation=True, will
        compute derivatives from the interpolated function (if it exists). nth order
        derivative can be taken with order=n, however we only support interpolated
        derivative of order=1,2 for now. epsilon and scale are parameters for the
        helpers.derivative() routine.

        Parameters
        ----------
        x : list[float] or np.ndarray
            Points where the derivative will be evaluated.
        order : int, optional
            Order of the derivative to take. The default is 1.
        bUseInterpolation : bool, optional
            Whether or not to use interpolation to evaluate the function.
            The default is True.
        epsilon : float, optional
            Relative accuracy at which the function is evaluated. The default is 1e-16.
        scale : float, optional
            Scale at which the function changes by O(1). The default is 1.0.

        Returns
        -------
        list[float | np.ndarray] or np.ndarray
            Value of the derivative at x.

        """
        x = np.asanyarray(x)
        if not bUseInterpolation or not self.hasInterpolation() or order > 2:
            return helpers.derivative(self._evaluateDirectly, x, n=order)

        # Use interpolated values whenever possible
        canInterpolateCondition, fxShape = self._findInterpolatablePoints(x)
        needsEvaluationCondition = ~canInterpolateCondition

        xEvaluateRegion = x[needsEvaluationCondition]

        results = np.empty(fxShape)
        results[canInterpolateCondition] = self._interpolatedDerivatives[order - 1](
            x[canInterpolateCondition]
        )

        ## Outside the interpolation region use whatever extrapolation
        ## type the function uses
        if xEvaluateRegion.size > 0:
            results[needsEvaluationCondition] = helpers.derivative(
                self._evaluateOutOfBounds, x, n=order, epsilon=epsilon, scale=scale
            )

        return results

    def _findInterpolatablePoints(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Finds x values where interpolation can be used. Return tuple is:
        canInterpolateCondition, fxShape where the condition is a numpy bool array and
        fxShape is the resulting shape of f(x).
        """

        canInterpolateCondition = (x <= self._rangeMax) & (x >= self._rangeMin)

        """If x is N-dimensional array and idx is a tuple index for this array,
        we want to return fx so that fx[idx] is the result of function evaluation at
        x[idx]. But if f(x) is vector-valued then necessarily fx shape will not match x
        shape. So figure out the shape here. 
        """
        if self._RETURN_VALUE_COUNT > 1:
            fxShape = x.shape + (self._RETURN_VALUE_COUNT,)
        else:
            fxShape = x.shape

        return canInterpolateCondition, fxShape

    def _interpolate(
        self,
        x: inputType,
        fx: outputType,
    ) -> None:
        """Does the actual interpolation and sets some internal values.
        Input x needs to be 1D, and input fx needs to be at most 2D.
        """

        x = np.asanyarray(x)
        fx = np.asanyarray(fx)
        assert x.ndim == 1 and fx.ndim <= 2, (
            "Shape error in _interpolate(), " "this should not happen!"
        )

        ## Can't specify different extrapolation methods for x > xmax, x < xmin in
        ## CubicSpline! This logic is handled manually in __call__()
        bShouldExtrapolate = EExtrapolationType.FUNCTION in (
            self.extrapolationTypeLower,
            self.extrapolationTypeUpper,
        )

        ## Explicitly drop non-numerics
        xFiltered, fxFiltered = self._dropBadPoints(x, fx)

        ## This works even if f(x) is vector valued
        self._interpolatedFunction = CubicSpline(
            xFiltered, fxFiltered, extrapolate=bShouldExtrapolate, axis=0
        )

        self._rangeMin = np.min(xFiltered)
        self._rangeMax = np.max(xFiltered)
        self._interpolationPoints = xFiltered
        self._interpolationValues = fxFiltered

        """Store a cubic spline for the 1st and 2nd derivatives into a list.
        We do not attempt to spline the higher derivatives as they are not 
        guaranteed to be continuous."""
        self._interpolatedDerivatives = [
            self._interpolatedFunction.derivative(1),
            self._interpolatedFunction.derivative(2),
        ]

    @staticmethod
    def _dropBadPoints(
        x: np.ndarray,
        fx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Removes non-numerical (x, fx) pairs. For 2D fx the check is applied row-wise.
        Input x needs to be 1D, and input fx needs to be at most 2D.
        Output is same shape as input.
        """
        if fx.ndim > 1:
            validIndices = np.all(np.isfinite(fx), axis=1)
            fxValid = fx[validIndices]
        else:
            ## fx is 1D array
            validIndices = np.all(np.isfinite(fx))
            fxValid = np.ravel(fx[validIndices])

        xValid = np.ravel(x[validIndices])

        return xValid, fxValid

    def _adaptiveInterpolationUpdate(self) -> None:
        """
        Handles interpolation table updates for adaptive interpolation.
        """

        ## Where did the new evaluations happen
        evaluatedPointMin = np.min(self._directlyEvaluatedAt)
        evaluatedPointMax = np.max(self._directlyEvaluatedAt)

        # Reset work variables (doing this here already to avoid spaghetti nesting)
        self._directEvaluateCount = 0
        self._directlyEvaluatedAt = []

        if self.hasInterpolation():
            appendPointCount = int(0.2 * self._initialInterpolationPointCount)
        else:
            appendPointCount = int(self._initialInterpolationPointCount / 2)

        self.extendInterpolationTable(
            evaluatedPointMin, evaluatedPointMax, appendPointCount, appendPointCount
        )

    def extendInterpolationTable(
        self,
        newMin: float,
        newMax: float,
        pointsMin: int,
        pointsMax: int,
    ) -> None:
        """
        Extend our interpolation table.
        NB: This will reset internally accumulated data of adaptive interpolation.

        Parameters
        ----------
        newMin : float
            New minimal value at which the interpolation starts.
        newMax : float
            New maximal value at which the interpolation starts.
        pointsMin : int
            Minimal number of points to use.
        pointsMax : int
            Maximal number of points to use.

        """
        if not self.hasInterpolation():
            newPoints = int(pointsMin + pointsMax)
            logging.warning(
                f"Warning: {self.__class__.__name__}.extendInterpolationRange() "
                "called without existing interpolation. "
                f"Creating new table in range [{newMin}, {newMax}] with {newPoints} "
                "points"
            )
            self.newInterpolationTable(newMin, newMax, newPoints)
            return

        # what to append to lower end
        if newMin < self._rangeMin and pointsMin > 0:

            ## Point spacing to use at new lower end
            spacing = np.abs(self._rangeMin - newMin) / pointsMin
            # arange stops one spacing before the max value, which is what we want
            appendPointsMin = np.arange(newMin, self._rangeMin, spacing)
        else:
            appendPointsMin = np.array([])

        # what to append to upper end
        if newMax > self._rangeMax and pointsMax > 0:

            ## Point spacing to use at new upper end
            spacing = np.abs(newMax - self._rangeMax) / pointsMax
            appendPointsMax = np.arange(
                self._rangeMax + spacing, newMax + spacing, spacing
            )
        else:
            appendPointsMax = np.array([])

        appendValuesMin = np.asarray(self._functionImplementation(appendPointsMin))
        appendValuesMax = np.asarray(self._functionImplementation(appendPointsMax))

        # Ordering is important since interpolation needs the x values to be ordered.
        # This works, but could be made safer by rearranging the resulting arrays:
        xRange = np.concatenate(
            (appendPointsMin, self._interpolationPoints, appendPointsMax)
        )
        fxRange: np.ndarray = np.concatenate(
            (appendValuesMin, np.asarray(self._interpolationValues), appendValuesMax)
        )

        self.newInterpolationTableFromValues(xRange, fxRange)

        ## Hacky reset of adaptive routines
        if self._bUseAdaptiveInterpolation:
            self.disableAdaptiveInterpolation()
            self.enableAdaptiveInterpolation()

    def readInterpolationTable(self, fileToRead: str) -> None:
        """
        Reads precalculated values from a file and does cubic interpolation.
        Each line in the file must be of form x f(x).
        For vector valued functions: x f1(x) f2(x)

        Parameters
        ----------
        fileToRead : str
            Path of the file where the interpolation table is stored.

        """

        # for logging
        selfName = self.__class__.__name__

        try:
            ## Each line should be of form x f(x).
            ## For vector valued functions, x f1(x) f2(x) ...
            data = np.genfromtxt(
                fileToRead, delimiter=" ", dtype=float, encoding=None
            )

            columns = data.shape[1]

            # now slice this column-wise. First column is x:
            x = data[:, 0]
            # and for fx we remove the first column,
            # using magic syntax 1: to leave all others
            fx = data[:, 1:]

            ## If f(x) is 1D, this actually gives it in messy format
            ## [ [fx1] [fx2] ...]. So let's fix that
            if columns == 2:
                fx = np.ravel(fx)

            self._interpolate(x, fx)

            ## check that what we read matches our function definition (just evaluate
            ## and compare at a few values)
            self._validateInterpolationTable(self._rangeMin)
            self._validateInterpolationTable(self._rangeMax)
            self._validateInterpolationTable((self._rangeMax - self._rangeMin) / 2.55)

            logging.debug(
                f"{selfName}: Succesfully read interpolation table from file. "
                "Range [{self._rangeMin}, {self._rangeMax}]"
            )

        except IOError as ioError:
            logging.warning(
                f"IOError! {selfName} attempted to read interpolation table from "
                "file, but got error:"
            )
            logging.warning(ioError)
            logging.warning("This is non-fatal. Interpolation table will not be updated.\n")

    def writeInterpolationTable(self, outputFileName: str) -> None:
        """
        Write our interpolation table to file.

        Parameters
        ----------
        outputFileName : str
            Name of the file where the interpolation table will be written.

        """
        try:
            ## Write to file, line i is of form: x[i] fx[i]. If our function is vector
            ## valued then x[i] fx1[i] fx2[i] ...

            stackedArray = np.column_stack(
                (
                    np.asarray(self._interpolationPoints),
                    np.asarray(self._interpolationValues),
                )
            )
            np.savetxt(outputFileName, stackedArray, fmt="%.15g", delimiter=" ")

            logging.debug(
                "Stored interpolation table for function "
                f"{self.__class__.__name__}, output file {outputFileName}."
            )

        except Exception as e:
            logging.warning(
                f"Error from {self.__class__.__name__}, function "
                f"writeInterpolationTable(): {e}"
            )

    def _validateInterpolationTable(
        self,
        x: float,
        absoluteTolerance: float = 1e-6,
    ) -> bool:
        """
        Test the interpolation table with some input.
        Result should agree with self._evaluateDirectly(x).
        """

        if (
            self._interpolatedFunction is None
            or not self._rangeMin <= x <= self._rangeMax
        ):
            logging.warning(
                f"{self.__class__.__name__}: _validateInterpolationTable called, "
                "but no valid interpolation table was found."
            )
            return False

        diff = self.evaluateInterpolation(x) - self._functionImplementation(x)
        if np.any(np.abs(diff) > absoluteTolerance):
            logging.warning(
                f"{self.__class__.__name__}: Could not validate interpolation table!"
                f" Value discrepancy was {diff}"
            )
            return False

        return True
