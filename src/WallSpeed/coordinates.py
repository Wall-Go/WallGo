import numpy as np

def compactifyCoordinates(z, pz, pp, L, T0):
    r""" Transforms coordinates to [-1, 1] interval

    All coordinates are in the wall frame.

    The barred coordinates are compactified:

    .. math::
        \bar{z} \equiv \frac{z}{\sqrt{z^2 + L^2}}, \qquad
        \bar{p}_{z} \equiv \tanh\left(\frac{p_z}{2 T_0}\right), \qquad
        \bar{p}_{\Vert} \equiv 1 - 2 e^{-p_\Vert/T_0}.


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
    r""" Transforms coordinates from [-1, 1] interval
    (inverse of compactifyCoordinates)

    All coordinates are in the wall frame.

    The barred coordinates are compactified:

    .. math::
        z = \frac{\bar{z} L }{\sqrt{1 - \bar{z}^2}}, \qquad
        p_z = 2 T_0\ \text{atanh}(\bar{p}_z), \qquad
        p_\Vert = - T_0 \log\left(\frac{1-\bar{p}_\Vert}{2}\right).

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
