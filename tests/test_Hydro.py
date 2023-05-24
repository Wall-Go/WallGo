import pytest
import numpy as np
import TestModel
from WallSpeed.Hydro import *

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

model1 = TestModel.TestModel(0.2,0.1,0.4)

def test_JouguetVelocity():
    res = np.zeros(5)
    for i in range(5):
        res[i] = findJouguetVelocity(model1,0.5+i*0.1)
#    assert res == pytest.approx([0.840948,0.776119,0.7240,0.6836,0.651791],rel=0.01*np.ones(5))
    np.testing.assert_allclose(res,[0.840948,0.776119,0.7240,0.6836,0.651791],rtol = 10**-3,atol = 0)
