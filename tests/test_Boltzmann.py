import pytest # for tests
import os # for path names
import numpy as np  # arrays and maths
import pathlib 

from WallGo.Grid import Grid
from WallGo.Polynomial2 import Polynomial
from WallGo.Boltzmann import BoltzmannSolver


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)


@pytest.mark.parametrize(
    "M, N, a, b, c, d, e, f",
    [#(10, 10, 1, 0, 0, 1, 0, 0),
    (25, 3, 1, 0, -1, 1, 0, -1)], 
)
def test_Delta00(boltzmannTestBackground, particle, M, N, a, b, c, d, e, f):
    r"""
    Tests that the Delta integral gives a known analytic result for
    :math:`\delta f = E \sqrt{(1 - \rho_z^2)(1 - \rho_\Vert)}`.
    """
    # setting up objects
    ## This is the fixture background constructed with input M. pytest magic that works because argument name here matches that used in fixture def 
    bg = boltzmannTestBackground
    grid = Grid(M, N, 1, 100)
    #poly = Polynomial(grid) # not used so commented out
    boltzmann = BoltzmannSolver(grid, 'Cardinal', 'Cardinal')

    boltzmann.updateParticleList( [particle] )
    boltzmann.setBackground(bg)

    # coordinates
    chi, rz, rp = grid.getCompactCoordinates() # compact
    rz = rz[np.newaxis, :, np.newaxis]
    rp = rp[np.newaxis, np.newaxis, :]
    xi, pz, pp = grid.getCoordinates() # non-compact
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # fluctuation mode
    msq = particle.msqVacuum(bg.fieldProfile)
    ## Drop start and end points in field space
    msq = msq[1:-1, np.newaxis, np.newaxis]
    E = np.sqrt(msq + pz**2 + pp**2)

    # integrand with known result
    eps = 2e-16
    integrand_analytic = E * np.sqrt((1 - rz**2) * (1 - rp)**2/(1-rp**2+eps)) / (np.log(2 / (1 - rp)) + eps)
    integrand_analytic *= (a + b * rz + c * rz**2)
    integrand_analytic *= (d + e * rp + f * rp**2)

    # doing computation
    Deltas = boltzmann.getDeltas(integrand_analytic)

    # comparing to analytic result
    # Delta00_analytic = bg.temperatureProfile**3 * (
    #     2 / (9 * np.pi**2) * (3 * a + c) * (3 * d + f)
    # )
    Delta00_analytic = (2*a + c)*(2*d + f)*bg.temperatureProfile**3/8
    #print(Deltas["00"].coefficients)
    #print(Delta00_analytic[1:-1])
    ratios = Deltas["00"].coefficients / Delta00_analytic[1:-1]

    # asserting result
    np.testing.assert_allclose(ratios, np.ones(M - 1), rtol=1e-2, atol=0)


@pytest.mark.parametrize("M, N", [(3, 3)])
def test_solution(boltzmannTestBackground, particle, M, N):

    # setting up objects
    ## This is the fixture background constructed with input M. pytest magic that works because argument name here matches that used in fixture def 
    bg = boltzmannTestBackground
    grid = Grid(M, N, 1, 1)
    #poly = Polynomial(grid)
    boltzmann = BoltzmannSolver(grid)

    boltzmann.updateParticleList( [particle] )
    boltzmann.setBackground(bg)

    ## Collision test file
    collisionFile = pathlib.Path(__file__).parent.resolve() / "Testdata/collisions_top_top_N3.hdf5" 
    boltzmann.readCollision(collisionFile)

    # solving Boltzmann equations
    deltaF = boltzmann.solveBoltzmannEquations()

    # building Boltzmann equation terms
    operator, source = boltzmann.buildLinearEquations()

    # checking difference
    diff = operator @ deltaF.flatten(order="C") - source

    # getting norms
    diffNorm = np.linalg.norm(diff)
    sourceNorm = np.linalg.norm(source)
    ratio = diffNorm / sourceNorm

    # asserting solution works
    assert ratio == pytest.approx(0, abs=1e-9)
