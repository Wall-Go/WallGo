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

    assert fields.NumFields() == numFields
    assert fields.NumPoints() == numPoints
    assert len(fields.GetField(0)) == numPoints
    assert len(fields.GetFieldPoint(0)) == numFields


@pytest.mark.parametrize("fieldArray, numFields, numPoints", [
    ([1], 1, 1),
    ([[1], [2], [3]], 1, 3),
    ([[1, 11], [2, 22], [3, 33]], 2, 3) 
])
def test_FieldsFromNumpy(fieldArray: Tuple[list], numFields: int, numPoints: int):
    
    fieldArray = np.asanyarray(fieldArray)
    fields = Fields.CastFromNumpy(fieldArray)

    assert fields.NumFields() == numFields
    assert fields.NumPoints() == numPoints
    assert len(fields.GetField(0)) == numPoints
    assert len(fields.GetFieldPoint(0)) == numFields


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