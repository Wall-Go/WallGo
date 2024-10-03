from WallGo import Fields

import pytest
import numpy as np
from typing import Tuple

## Define some points in field space and check that the Fields class interprets them correctly

@pytest.mark.parametrize("fieldSpacePoints, numFields, numPoints", [
    (([1]), 1, 1),
    (([1], [2], [3]), 1, 3),
    (([1, 11], [2, 22], [3, 33]), 2, 3) 
])
def test_FieldsFromTuple(fieldSpacePoints: Tuple[list], numFields: int, numPoints: int):
    fields = Fields(fieldSpacePoints)

    assert fields.numFields() == numFields
    assert fields.numPoints() == numPoints
    assert len(fields.getField(0)) == numPoints
    assert len(fields.getFieldPoint(0)) == numFields


@pytest.mark.parametrize("fieldArray, numFields, numPoints", [
    ([1], 1, 1),
    ([[1], [2], [3]], 1, 3),
    ([[1, 11], [2, 22], [3, 33]], 2, 3) 
])
def test_FieldsFromNumpy(fieldArray: Tuple[list], numFields: int, numPoints: int):
    
    fieldArray = np.asanyarray(fieldArray)
    fields = Fields.castFromNumpy(fieldArray)

    assert fields.numFields() == numFields
    assert fields.numPoints() == numPoints
    assert len(fields.getField(0)) == numPoints
    assert len(fields.getFieldPoint(0)) == numFields


@pytest.mark.parametrize("fieldArrays", [
    ([1, 11], [2, 22], [3]),
    (1, [2], [3]),
])
def test_Fields_invalid(fieldArrays):
    """Test invalid input to Fields. Should raise a ValueError from numpy due to failing array stacking
    """
    with pytest.raises(ValueError):
        Fields(fieldArrays)
    ## Something like Field(1, 2, 3) should fail too, but we need to throw this manually. TODO
        

@pytest.mark.parametrize("fields, fieldNumber, newField", [
    (Fields( [[1, 2, 3]] ), 1, np.asarray([10])) ,
    (Fields( [[1, 2, 3], [4, 5, 6], [7, 8, 9]] ), 1, np.asarray([10, 11, 12]))    
])
def test_setField(fields, fieldNumber, newField):

    fields.setField(fieldNumber, newField)
    np.testing.assert_array_equal( fields.getField(fieldNumber), newField)
    