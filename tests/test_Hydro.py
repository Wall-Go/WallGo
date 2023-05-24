import pytest
import TestModel
from WallSpeed.Hydro import *

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

model1 = TestModel.TestModel(0.2,0.1,0.4)

def test_JouguetVelocity():
    res = findJouguetVelocity(model1,0.7)
    assert res == pytest.approx(0.7240,rel=0.01)
