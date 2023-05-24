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

def test_DetonationMatch():
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.5),0.5)
    np.testing.assert_allclose(res,(0.925043,0.848164,0.5,0.614381),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.6),0.6)
    np.testing.assert_allclose(res,(0.853731,0.777282,0.6,0.685916),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.7),0.7)
    np.testing.assert_allclose(res,(0.796415,0.737286,0.7,0.763685),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.8),0.8)
    np.testing.assert_allclose(res,(0.751924,0.710458,0.8,0.846123),rtol = 10**-3,atol = 0)
    res = matchDeton(model1,1.1*findJouguetVelocity(model1,0.9),0.9)
    np.testing.assert_allclose(res,(0.71697,0.690044,0.9,0.931932),rtol = 10**-3,atol = 0)
    





                                    #,[(0.853731,0.777282,0.6,0.685916)],
                                    #[(0.796415,0.737286,0.7,0.763685)],[(0.751924,0.710458,0.8,0.846123)],[(0.71697,0.690044,0.9,0.931932)]],rtol = 10**-3,atol = 0)

