"""Basic test of imports and setup"""


def test_importWallGo() -> None:
    """Testing import of WallGo"""
    import WallGo
    print(f"WallGo version {WallGo.__version__}")
