import numpy as np
import numpy.typing as npt
from typing import Tuple, Union


## Dunno if this inheritance is a good idea. But this way we can force a common shape for field arrays while automatically being numpy compatible
# https://numpy.org/doc/stable/user/basics.subclassing.html

## One point in field space. This is 1D array.
class FieldPoint(np.ndarray):

    def __new__(cls, arr: np.ndarray):
        if (arr.ndim > 1):
            raise ValueError("FieldPoint must be 1D")
        
        return arr.view(cls)
    
    def NumFields(self):
        """Returns how many background fields we contain
        """
        return self.shape[0]
    
    def GetField(self, i: int):
        return self[i]


## TODO should not have SetField(), GetField() because numpy already has setfield(), getfield() ... lol

class Fields(np.ndarray):
    """Simple class for holding collections of background fields in common format.

    If the theory has N background fields, then a field-space point is defined by a list of length N.
    This array describes a collection of field-space points, so that the shape is (numPoints, numFields). 
    IE. each row is one field-space point. This is always a 2D array, even if we just have one field-space point.
    
    """

    """Developer note! This is subclass of np.ndarray so in principle we can pass this to scipy routines directly, 
    eg. as the initial guess array in ``scipy.optimize.minimize(someFunction, array)``. 
    But scipy seems to forcibly convert back to standard np.ndarray, so if the someFunction wants to use extended functionality
    of the Fields class then a wrapper with explicit cast is needed.
    """

    ## Axis identifier: operate on same field type over different field-space points 
    overFieldPoints: int = 0
    ## Axis identifier: operate on over different fields at same field-space point
    overFieldTypes: int = 1


    # Custom constructor that stacks 1D arrays or lists of field-space points into a 2D array 
    def __new__(cls, *fieldSpacePoints: Tuple[FieldPoint]):
        obj = np.row_stack(fieldSpacePoints)
        obj = np.atleast_2d(obj)
        return obj.view(cls)
    
    def __array_finalize__(self, obj):
        if obj is None: return


    @staticmethod
    def CastFromNumpy(arr: np.ndarray) -> 'Fields':
        """
        """
        assert len(arr.shape) <= 2
        return np.atleast_2d(arr).view(Fields)
    

    def NumPoints(self):
        """Returns how many field-space points we contain
        """
        return self.shape[0]

    def NumFields(self):
        """Returns how many background fields we contain
        """
        return self.shape[1]
    
    def Resize(self, newNumPoints, newNumFields) -> 'Fields':
        """Returns a resized array (uses np.resize internally)
        """
        newShape = (newNumPoints, newNumFields)
        return np.resize(self, newShape).view(Fields)

    def GetFieldPoint(self, i: int) -> FieldPoint:
        """Get a field space point. It would be a 1D array, but we cast to a FieldPoint object for consistency. 
        NB: no validation so will error if the index is out of bounds"""
        return self[i].view(FieldPoint)
    
    def GetField(self, i: int) -> np.ndarray:
        """Get field at index i. Ie. if the theory has N background fields f_i, this will give all values of field f_i
        as a 1D array.
        """
        ## Fields are on columns
        return self[:, i].view(np.ndarray)
    
    
    def GetFieldPreserveShape(self, i: int) -> 'Fields':
        """Like GetField(i), but returns an array of same shape as the original Fields object.
        We leave only values corresponding to the field 'i', setting other elements to zero.
        """

        ## Our field i is on column i
        newFields = np.zeros_like(self, dtype=float)
        newFields[:, i] = self[:, i]
        return newFields


    def SetField(self, i: int, fieldArray: np.ndarray) -> 'Fields':
        # Set new values to our field at index i. Operates in place. Is this safe actually...?
        self[:, i] = fieldArray
        return self
    
    def TakeSlice(self, idx_start: int, idx_end: int, axis: int) -> 'Fields':
        """Take slice along specified axis. Output is as you'd expect from a 2D numpy array,
        but we cast to Fields object. Input indices are INCLUSIVE. 
        NB: no range checking here. 
        """
        if (axis == self.overFieldPoints):
            return self[idx_start:idx_end, :]
        else:
            return self[:, idx_start:idx_end]
