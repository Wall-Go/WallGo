import pytest
import numpy as np
from WallSpeed.coordinates import compactifyCoordinates, decompactifyCoordinates


def test_compactify_inversion():
    """
    Tests inversion to ~machine precision
    """
    z = 7
    pz = -100
    pp = 120
    L = 3
    T = 25

    z_compact, pz_compact, pp_compact = compactifyCoordinates(z, pz, pp, L, T)
    z_inv, pz_inv, pp_inv = decompactifyCoordinates(
        z_compact, pz_compact, pp_compact, L, T
    )

    max_diff = max(abs(z - z_inv), abs(pz - pz_inv), abs(pp - pp_inv))
    assert max_diff == pytest.approx(0, abs=1e-13)

def test_compactify_range():
    """
    Tests range is within [-1, 1]
    """
    z = np.linspace(-7, 7)
    pz = np.linspace(-100, 100)
    pp = np.linspace(0, 120)
    L = 3
    T = 25

    z_compact, pz_compact, pp_compact = compactifyCoordinates(z, pz, pp, L, T)

    min_compact = np.amin([z_compact, pz_compact, pp_compact])
    max_compact = np.amax([z_compact, pz_compact, pp_compact])
    assert (min_compact >= -1 and max_compact <= 1)
