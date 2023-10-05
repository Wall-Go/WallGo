import pytest # for tests
import os # for path names
import numpy as np  # arrays and maths
from WallSpeed.Grid import Grid
from WallSpeed.Polynomial import Polynomial
from WallSpeed.Boltzmann import BoltzmannSolver
from .conftest import background


real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path)


@pytest.mark.parametrize(
    "M, N, a, b, c, d, e, f",
    [#(10, 10, 1, 0, 0, 1, 0, 0),
    (25, 25, 1, 0, 0, 1, 0, 0)]
)
def test_Delta00(particle, M, N, a, b, c, d, e, f):
    r"""
    Tests that the Delta integral gives a known analytic result for
    :math:`\delta f = E \sqrt{(1 - \rho_z^2)(1 - \rho_\Vert)}`.
    """
    # setting up objects
    bg = background(M)
    grid = Grid(M, N, 1, 1)
    poly = Polynomial(grid)
    boltzmann = BoltzmannSolver(grid, bg, particle)

    # coordinates
    chi, rz, rp = grid.getCompactCoordinates() # compact
    rz = rz[np.newaxis, :, np.newaxis]
    rp = rp[np.newaxis, np.newaxis, :]
    xi, pz, pp = grid.getCoordinates() # non-compact
    pz = pz[np.newaxis, :, np.newaxis]
    pp = pp[np.newaxis, np.newaxis, :]

    # fluctuation mode
    msq = particle.msqVacuum(bg.fieldProfile)
    msq = msq[:, np.newaxis, np.newaxis]
    E = np.sqrt(msq + pz**2 + pp**2)

    # integrand with known result
    eps = 2e-16
    integrand_analytic = E * (1 - rz**2) * (1 - rp) / (np.log(2 / (1 - rp)) + eps)
    integrand_analytic *= (a + b * rz + c * rz**2)
    integrand_analytic *= (d + e * rp + f * rp**2)

    # doing computation
    Deltas = boltzmann.getDeltas(integrand_analytic)

    # comparing to analytic result
    Delta00_analytic = bg.temperatureProfile**3 * (
        2 / (9 * np.pi**2) * (3 * a + c) * (3 * d + f)
    )
    ratios = Deltas["00"] / Delta00_analytic

    # asserting result
    np.testing.assert_allclose(ratios, np.ones(M - 1), rtol=1e-2, atol=0)


@pytest.mark.parametrize("M, N", [(20, 20)])
def test_solution(particle, M, N):
    # setting up objects
    bg = background(M)
    grid = Grid(M, N, 1, 1)
    poly = Polynomial(grid)
    boltzmann = BoltzmannSolver(grid, bg, particle)

    # solving Boltzmann equations
    deltaF = boltzmann.solveBoltzmannEquations()

    # building Boltzmann equation terms
    operator, source = boltzmann.buildLinearEquations()

    # checking difference
    diff = operator @ deltaF.flatten(order="F") - source

    # getting norms
    diffNorm = np.linalg.norm(diff)
    sourceNorm = np.linalg.norm(source)
    ratio = diffNorm / sourceNorm

    # asserting solution works
    assert ratio == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("MN_coarse, MN_fine", [(4, 6)])
def test_convergence(particle, MN_coarse, MN_fine):
    # Boltzmann equation on the coarse grid
    grid_coarse = Grid(MN_coarse, MN_coarse, 1, 1)
    poly_coarse = Polynomial(grid_coarse)
    background_coarse = background(MN_coarse)
    boltzmann_coarse = BoltzmannSolver(grid_coarse, background_coarse, particle)
    deltaF_coarse = boltzmann_coarse.solveBoltzmannEquations()

    # Boltzmann equation on the fine grid
    grid_fine = Grid(MN_fine, MN_fine, 1, 1)
    poly_fine = Polynomial(grid_fine)
    background_fine = background(MN_fine)
    boltzmann_fine = BoltzmannSolver(grid_fine, background_fine, particle)
    deltaF_fine = boltzmann_fine.solveBoltzmannEquations()

    # comparing the results on the two grids
    chi, rz, rp = grid_fine.getCompactCoordinates()
    print(f"{deltaF_coarse.shape=}")
    print(f"{deltaF_fine.shape=}")
    temp1 = poly_coarse.cardinal(chi, MN_coarse, "z")
    print(f"{temp1.shape=}")
    temp2 = poly_coarse.chebyshev(rz, MN_coarse, "pz")
    print(f"{temp2.shape=}")
    temp3 = poly_coarse.chebyshev(rp, MN_coarse, "pp")
    print(f"{temp3.shape=}")
    deltaF_coarse_fine = np.einsum(
        "abc, ai, bj, ck -> ijk",
        boltzmann_coarse,
        poly_coarse.cardinal(chi, MN_coarse, "z"),
        poly_coarse.chebyshev(rz, MN_coarse, restriction="full"),
        poly_coarse.chebyshev(rp, MN_coarse, restriction="partial"),
        optimize=True,
    )
    print(f"{deltaF_coarse_fine.shape=}")

    # getting norms
    pass

    # asserting results are close
