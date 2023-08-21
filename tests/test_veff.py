import pytest
from WallSpeed import Particle, FreeEnergy, Model

#model1 = TestModel.TestModel(0.2,0.1,0.4)

#print(model1.eBrok(0.3))

#print(findJouguetVelocity(model1,0.7))

def test_BM1():
    mod = Model(125,160.,1.0,1.2)
    # res = pot.findMinimum(None,100)
    assert mod.Vtot([[110,130]],100) == pytest.approx([-1.15450678e+09],rel=0.01)
    # assert res == pytest.approx([196.734, -0.2536],rel=0.01)

# def test_BM2():
#     pot = Model(1,125,160,1.0,1.6)
#     res = pot.findMinimum(None,100)
#     assert res == pytest.approx([4.4e-9, -146.758],rel=0.01)

