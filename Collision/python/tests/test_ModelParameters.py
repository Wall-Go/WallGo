import pytest

from WallGoCollision import ModelParameters

def test_addAndGet() -> None:
    """"""
    params = ModelParameters()

    assert params.size() == 0
    
    params.add("p1", 1.0)
    
    assert params.size() == 1
    assert params.contains("p1")
    assert params["p1"] == 1.0

    params.add("p2", 2.0)
    assert params.size() == 2
    assert params.contains("p2")
    assert params["p2"] == 2.0

def test_bracketOperator() -> None:
    """Tests that the [] operator can be used in place of add()"""
    params = ModelParameters()

    params["p1"] = 1.0
    
    assert params.contains("p1")
    assert params["p1"] == 1.0
    

def test_modifyParam() -> None:
    """"""
    params = ModelParameters()
    
    params.add("p1", 1.0)
    # modify existing
    params.add("p1", 3.0)
    
    assert params.size() == 1
    # these two are equivalent:
    assert params["p1"] == 3.0
    assert params.at("p1") == 3.0
    

def test_removeParam() -> None:
    """"""
    params = ModelParameters()

    params.add("p1", 1.0)
    params.add("p2", 2.0)
    
    assert params.size() == 2

    params.remove("p1")
    assert params.size() == 1
    assert not params.contains("p1")

    params.remove("p2")
    assert params.size() == 0
    assert not params.contains("p2")
    
def test_invalidAccess() -> None:
    """Checks that invalid access to a parameter raises IndexError"""
    params = ModelParameters()

    with pytest.raises(IndexError):
        params["dumb"]

    with pytest.raises(IndexError):
        params.at("dumb")

    assert not params.contains("dumb")




