"""
Classes for Fields and FieldPoint objects
that hold information about the background fields.
"""

from typing import Tuple
import numpy as np


## Dunno if this inheritance is a good idea.
# But this way we can force a common shape for field arrays
# while automatically being numpy compatible
# https://numpy.org/doc/stable/user/basics.subclassing.html


## One point in field space. This is 1D array.
class FieldPoint(np.ndarray):
    """
    FieldPoint is a subclass of numpy's ndarray,
    representing a point in a field with certain constraints.

    Attributes:
        None

    Methods:
        __new__(cls, arr: np.ndarray):
            Constructs a new FieldPoint instance from a 1D numpy array.
            Raises a ValueError if the input array is not 1D.

        numFields():
            Returns the number of fields contained in the FieldPoint.

        getField(i: int):
            Retrieves the field value at the specified index.

        setField(i: int, value: float) -> "FieldPoint":
            Sets the field value at the specified index and
            returns the updated FieldPoint.
    """

    def __new__(cls, arr: np.ndarray) -> "FieldPoint":
        if arr.ndim > 1:
            raise ValueError("FieldPoint must be 1D")

        return arr.view(cls)

    def numFields(self) -> int:
        """
        Calculate the number of background fields.

        This method returns the number of background fields contained within the object.

        Returns:
            int: The number of background fields.
        """
        """Returns how many background fields we contain"""
        return self.shape[0]

    def getField(self, i: int) -> float:
        """
        Retrieve the field value at the specified index.

        Args:
            i (int): The index of the field to retrieve.

        Returns:
            float: The value of the field at the specified index.
        """
        return self[i]

    def setField(self, i: int, value: float) -> "FieldPoint":
        """
        Sets the value of the field at the specified index.

        Args:
            i (int): The index at which to set the value.
            value (float): The value to set at the specified index.

        Returns:
            FieldPoint: The updated FieldPoint object.
        """
        self[i] = value
        return self


## TODO should not have setField(), getField() because numpy already has setfield(), getfield()


class Fields(np.ndarray):
    """
    Simple class for holding collections of background fields in a common format.

    If the theory has N background fields,
    then a field-space point is defined by a list of length N.
    This array describes a collection of field-space points,
    so that the shape is (numPoints, numFields).
    Each row represents one field-space point.
    This is always a 2D array, even if we just have one field-space point.
    """

    """
    Developer note: This is a subclass of np.ndarray, so in principle,
    we can pass this to scipy routines directly,
    e.g., as the initial guess array in `scipy.optimize.minimize(someFunction, array)`.
    However, scipy seems to forcibly convert back to a standard np.ndarray,
    so if someFunction wants to use the extended
    functionality of the Fields class, a wrapper with an explicit cast is needed.
    """

    ## Axis identifier: operate on same field type over different field-space points
    overFieldPoints: int = 0
    ## Axis identifier: operate on over different fields at same field-space point
    overFieldTypes: int = 1

    # Custom constructor that stacks 1D arrays or
    # lists of field-space points into a 2D array
    def __new__(cls, *fieldSpacePoints: Tuple[FieldPoint]) -> "Fields":
        obj = np.vstack(fieldSpacePoints)
        obj = np.atleast_2d(obj)
        return obj.view(cls)

    def __array_finalize__(self, obj: np.ndarray | None) -> None:
        if obj is None:
            return

    @staticmethod
    def castFromNumpy(arr: np.ndarray) -> "Fields":
        """
        Cast a NumPy array to a Fields object.

        Parameters:
        arr (np.ndarray): The input NumPy array. It can be either 1D or 2D.

        Returns:
        Fields: A Fields object created from the input NumPy array.

        Raises:
        AssertionError: If the input array has more than 2 dimensions.
        """
        """ """
        assert len(arr.shape) <= 2
        return np.atleast_2d(arr).view(Fields)

    def numPoints(self) -> int:
        """Returns how many field-space points we contain"""
        return self.shape[0]

    def numFields(self) -> int:
        """Returns how many background fields we contain"""
        return self.shape[1]

    def resizeFields(self, newNumPoints: int, newNumFields: int) -> "Fields":
        """Returns a resized array (uses np.resize internally)"""
        newShape = (newNumPoints, newNumFields)
        return np.resize(self, newShape).view(Fields)

    def getFieldPoint(self, i: int) -> FieldPoint:
        """Get a field space point. It would be a 1D array,
        but we cast to a FieldPoint object for consistency.
        NB: no validation so will error if the index is out of bounds"""
        return self[i].view(FieldPoint)

    def getField(self, i: int) -> np.ndarray:
        """Get field at index i. Ie. if the theory has N background fields f_i,
        this will give all values of field f_i as a 1D array.
        """
        ## Fields are on columns
        return self[:, i].view(np.ndarray)

    def getFieldPreserveShape(self, i: int) -> np.ndarray:
        """
        Retrieve a field while preserving the original shape of the Fields object.

        This method returns an array of the same shape as the original Fields object,
        but with all elements set to zero except for those corresponding to
        the specified field index.

        Args:
            i (int): The index of the field to retrieve.

        Returns:
            np.ndarray: An array of the same shape as the original Fields object,
            with only the values of the specified field index preserved.
        """

        ## Our field i is on column i
        newFields = np.zeros_like(self, dtype=float)
        newFields[:, i] = self[:, i]
        return newFields

    def setField(self, i: int, fieldArray: np.ndarray) -> "Fields":
        """
        Set new values to the field at the specified index.

        Parameters:
        i (int): The index at which to set the new field values.
        fieldArray (np.ndarray): The array containing the new field values.

        Returns:
        Fields: The updated Fields object with the new values set at
        the specified index.
        """
        # Set new values to our field at index i.
        # Operates in place. Is this safe actually...?
        self[:, i] = fieldArray
        return self

    def takeSlice(self, idxStart: int, idxEnd: int, axis: int) -> np.ndarray:
        """Extracts a slice from the array along the specified axis.

        Parameters:
        idxStart (int): The starting index of the slice (inclusive).
        idxEnd (int): The ending index of the slice (inclusive).
        axis (int): The axis along which to take the slice.

        Returns:
        np.ndarray: A sliced portion of the array as a numpy ndarray.

        Notes:
        - The method does not perform range checking on the indices.
        - The output is cast to a Fields object.
        """
        if axis == self.overFieldPoints:
            return self[idxStart:idxEnd, :]
        else:
            return self[:, idxStart:idxEnd]
