import pytest
from WallSpeed.model import *

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

def test_BM1():
    pot = Model(1,125,160,1.0,1.2)
    res = pot.findMinimum(None,100)
    assert res == pytest.approx([193.033, 0.2785],rel=0.01)

def test_BM2():
    pot = Model(1,125,160,1.0,1.6)
    res = pot.findMinimum(None,100)
    assert res == pytest.approx([4.4e-9, -146.758],rel=0.01)
