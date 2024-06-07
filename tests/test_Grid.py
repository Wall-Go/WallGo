import pytest
import numpy as np
import WallGo


def test_GridCompactify():
    """
    Tests range is within [-1, 1]
    """
    M = 20
    N = 20
    L = 3
    T = 5
    grid = WallGo.Grid(M, N, L, T)

    chi, rz, rp = grid.getCompactCoordinates()

    min_compact = np.amin([chi, rz, rp])
    max_compact = np.amax([chi, rz, rp])
    assert (min_compact >= -1 and max_compact <= 1)


def test_GridCompactifyInversion():
    """
    Tests compactify inversion to ~machine precision
    """
    M = 20
    N = 20
    L = 3
    T = 5
    grid = WallGo.Grid(M, N, L, T)

    chi, rz, rp = grid.getCompactCoordinates()
    xi, pz, pp = grid.getCoordinates()
    xi_inv, pz_inv, pp_inv = grid.decompactify(chi, rz, rp)

    max_diff = np.amax([abs(xi - xi_inv), abs(pz - pz_inv), abs(pp - pp_inv)])
    assert max_diff == pytest.approx(0, abs=1e-13)
