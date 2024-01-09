import numpy as np
import numpy.typing as npt
from typing import Tuple, Union


## Dunno if this inheritance is a good idea. But this way we can force a common shape for field arrays while automatically being numpy compatible
class Fields(np.ndarray):
    """Simple class for holding background fields in common format.

    If the theory has N background fields, then a field-space point is defined by a list of length N.
    This array describes a collection of field-space points, so that the shape is (numPoints, numFields). 
    IE. each row is one field-space point. This is always a 2D array, even if we just have one field-space point.
    
    """

    """Developer note! This is subclass of np.ndarray so in principle we can pass this to scipy routines directly, 
    eg. as the initial guess array in ``scipy.optimize.minimize(someFunction, array)``. 
    But scipy seems to forcibly convert back to standard np.ndarray, so if the someFunction wants to use extended functionality
    of the Fields class then a wrapper with explicit cast is needed.
    """

    # Custom constructor that stacks 1D arrays or lists of field-space points into a 2D array 
    def __new__(cls, *fieldSpacePoints: Tuple[npt.ArrayLike]):
        obj = np.row_stack(fieldSpacePoints)
        obj = np.atleast_2d(obj)
        return obj.view(cls)
    
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

    def GetFieldPoint(self, i: int) -> np.ndarray[float]:
        """Get a field space point. NB: no validation so will error if the index is out of bounds"""
        return self[i]
    
    def GetField(self, i: int) -> np.ndarray[float]:
        """Get field at index i. Ie. if the theory has N background fields f_i, this will give all values of field f_i
        """
        ## Fields are on columns
        return self[:, i]
    