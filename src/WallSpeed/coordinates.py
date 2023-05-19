import numpy as np

def compactifyCoordinates(z, pz, pp, L, T0):
    """ Transforms coordinates to [-1, 1] interval

    All coordinates are in the wall frame.

    Parameters
    ----------
    z : array_like
        Array of z coordinate positions.
    pz : array_like
        Array of momenta in the z direction.
    pp : array_like
        Array of momenta parallel to the wall.
    L : float
        Length scale determining transform in z direction.
    T0 : float
        Temperature scale determining transform in momentum directions.

    Returns
    -------
    z_compact : array_like
        z mapped to [-1, 1] interval.
    pz_compact : array_like
        pz mapped to [-1, 1] interval.
    pp_compact : array_like
        pp mapped to [-1, 1] interval.
    """
    z_compact = z / np.sqrt(L**2 + z**2)
    pz_compact = np.tanh(pz / 2 / T0)
    pp_compact = 1 - 2 * np.exp(-pp / T0)
    return z_compact, pz_compact, pp_compact


def decompactifyCoordinates(z_compact, pz_compact, pp_compact, L, T0):
    """ Transforms coordinates from [-1, 1] interval
    (inverse of compactifyCoordinates)

    All coordinates are in the wall frame.

    Parameters
    ----------
    z_compact : array_like
        z mapped to [-1, 1] interval.
    pz_compact : array_like
        pz mapped to [-1, 1] interval.
    pp_compact : array_like
        pp mapped to [-1, 1] interval.
    L : float
        Length scale determining transform in z direction.
    T0 : float
        Temperature scale determining transform in momentum directions.

    Returns
    -------
    z : array_like
        Array of z coordinate positions.
    pz : array_like
        Array of momenta in the z direction.
    pp : array_like
        Array of momenta parallel to the wall.
    """
    z = L * z_compact / np.sqrt(1 - z_compact**2)
    pz = 2 * T0 * np.arctanh(pz_compact)
    pp = - T0 * np.log((1 - pp_compact) / 2)
    return z, pz, pp
