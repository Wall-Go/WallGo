import pytest
from WallSpeed.oli import main_function


def test_main_function():
    res = main_function()
    assert res == pytest.approx(0.577, rel=0.01)
