import pytest  # for tests
import os  # for path names
import numpy as np  # arrays and maths
from WallGo.Grid import Grid
from WallGo.Boltzmann import BoltzmannSolver
from WallGo.CollisionArray import CollisionArray
from WallGo.WallGoUtils import getSafePathToResource
from .conftest import background


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)


@pytest.mark.parametrize(
    "M, N, a, b, c, d, e, f",
    [(25, 19, 1, 2, 3, 4, 5, 6),
    (5, 5, 1, 1, 2, 3, 5, 8)]
)
def test_Delta00(particle, M, N, a, b, c, d, e, f):
    r"""
    Tests that the Delta integral gives a known analytic result for
    :math:`\delta f = E \sqrt{(1 - \rho_z^2)(1 - \rho_\Vert)}`.
    """
    # setting up objects
    bg = background(M)
    grid = Grid(M, N, 1, 100)
    suffix = "hdf5"
    fileName = f"collisions_top_top_N{grid.N}.{suffix}"
    collisionFile = getSafePathToResource("Data/" + fileName)
    collisionArray = CollisionArray(collisionFile, grid.N, 'Cardinal', particle, particle)
    boltzmann = BoltzmannSolver(grid, bg, particle, collisionArray, 'Spectral', 'Cardinal', 'Cardinal')

    # coordinates
    chi, rz, rp = grid.getCompactCoordinates() # compact
    rz = rz[np.newaxis, :, np.newaxis]
    rp = rp[np.newaxis, np.newaxis, :]
    xi, pz, pp = grid.getCoordinates() # non-compact
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # fluctuation mode
    msq = particle.msqVacuum(bg.fieldProfile[:,1:-1])
    msq = msq[:, np.newaxis, np.newaxis]
    E = np.sqrt(msq + pz**2 + pp**2)

    # integrand with known result
    eps = 2e-16
    integrand_analytic = 2 * E * (1 - rz**2) * (1 - rp**2) * np.sqrt((1 - rz**2) * (1 - rp)**2 / (1 - rp**2 + eps)) / (np.log(2 / (1 - rp)) + eps)
    integrand_analytic *= (a + b * rz + c * rz**2)
    integrand_analytic *= (d + e * rp + f * rp**2)

    # doing computation
    Deltas = boltzmann.getDeltas(integrand_analytic)

    # comparing to analytic result
    Delta00_analytic = (4 * a + c) * (4 * d + f) * bg.temperatureProfile**3 / 64

    # asserting result
    np.testing.assert_allclose(Deltas["00"].coefficients, Delta00_analytic[1:-1], rtol=1e-14, atol=0)


@pytest.mark.parametrize("M, N", [(3, 3), (5, 5)])
def test_solution(particle, M, N):
    # setting up objects
    bg = background(M)
    grid = Grid(M, N, 1, 1)
    suffix = "hdf5"
    fileName = f"collisions_top_top_N{grid.N}.{suffix}"
    collisionFile = getSafePathToResource("Data/" + fileName)
    collisionArray = CollisionArray(collisionFile, grid.N, 'Chebyshev', particle, particle)
    boltzmann = BoltzmannSolver(grid, bg, particle, collisionArray)

    # solving Boltzmann equations
    deltaF = boltzmann.solveBoltzmannEquations()

    # building Boltzmann equation terms
    operator, source, liouville, collision = boltzmann.buildLinearEquations()

    # checking difference
    diff = operator @ deltaF.flatten(order="C") - source

    # getting norms
    diffNorm = np.linalg.norm(diff)
    sourceNorm = np.linalg.norm(source)
    ratio = diffNorm / sourceNorm

    # asserting solution works
    assert ratio == pytest.approx(0, abs=1e-14)
