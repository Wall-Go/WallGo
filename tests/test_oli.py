import pytest
from WallSpeed.oli import Oli


def test_main_function():
    res = Oli.main_function()
    assert res == pytest.approx(0.577, rel=0.01)
