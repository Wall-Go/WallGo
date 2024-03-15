

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
