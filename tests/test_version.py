""" Tests consistent version numbering"""
import importlib
import WallGo


def test_version() -> None:
    """
    Follows https://packaging.python.org/en/latest/discussions/single-source-version/
    """
    assert WallGo.__version__ == importlib.metadata.version("WallGo")
