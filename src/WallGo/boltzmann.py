"""
Classes for solving the Boltzmann equations for out-of-equilibrium particles.
"""

import sys
import typing
from copy import deepcopy
import logging
import numpy as np
import findiff  # finite difference methods
from .containers import BoltzmannBackground, BoltzmannDeltas
from .grid import Grid
from .polynomial import Polynomial
from .particle import Particle
from .collisionArray import CollisionArray
from .results import BoltzmannResults
from .exceptions import CollisionLoadError

if typing.TYPE_CHECKING:
    import importlib


class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    # Static value holding of natural log of the maximum expressible float
    MAX_EXPONENT: typing.Final[float] = sys.float_info.max_exp * np.log(2)

    # Member variables
    grid: Grid
    offEqParticles: list[Particle]
    background: BoltzmannBackground
    collisionArray: CollisionArray

    def __init__(
        self,
        grid: Grid,
        basisM: str = "Cardinal",
        basisN: str = "Chebyshev",
        derivatives: str = "Spectral",
        collisionMultiplier: float = 1.0,
    ):
        """
        Initialisation of BoltzmannSolver

        Parameters
        ----------
        grid : Grid
            An object of the Grid class.
            integrals.
        basisM : str, optional
            The position polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Cardinal'.
        basisN : str, optional
            The momentum polynomial basis type, either 'Cardinal' or 'Chebyshev'.
            Default is 'Chebyshev'.
        derivatives : {'Spectral', 'Finite Difference'}, optional
            Choice of method for computing derivatives. Default is 'Spectral'
            which is expected to be more accurate.
        collisionMultiplier : float, optional
            Factor by which the collision term is multiplied. Can be used for testing.
            Default is 1.0.

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """

        self.grid = grid
        BoltzmannSolver._checkDerivatives(derivatives)
        self.derivatives = derivatives
        BoltzmannSolver._checkBasis(basisM)
        BoltzmannSolver._checkBasis(basisN)
        if derivatives == "Finite Difference":
            assert (
                basisM == "Cardinal" and basisN == "Cardinal"
            ), "Must use Cardinal basis for Finite Difference method"

        # Position polynomial type
        self.basisM = basisM
        # Momentum polynomial type
        self.basisN = basisN
        
        self.collisionMultiplier = collisionMultiplier

        # These are set, and can be updated, by our member functions
        # TODO: are these None types the best way to go?
        self.background = None  # type: ignore[assignment]
        self.collisionArray = None  # type: ignore[assignment]
        self.offEqParticles = []

    def setBackground(self, background: BoltzmannBackground) -> None:
        """
        Setter for the BoltzmannBackground
        """
        self.background = deepcopy(
            background
        )  # do we need a deepcopy? Does this even work generally?
        self.background.boostToPlasmaFrame()

    def setCollisionArray(self, collisionArray: CollisionArray) -> None:
        """
        Setter for the CollisionArray
        """
        self.collisionArray = collisionArray

    def updateParticleList(self, offEqParticles: list[Particle]) -> None:
        """
        Setter for the list of out-of-equilibrium Particle objects
        """
        # TODO: update the collision array as well when one updates the particle list
        for p in offEqParticles:
            assert isinstance(p, Particle)

        self.offEqParticles = offEqParticles

    def getDeltas(self, deltaF: typing.Optional[np.ndarray] = None) -> BoltzmannResults:
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (15) of 2204.13120 [LC22]_.

        Parameters
        ----------
        deltaF : array_like, optional
            The deviation of the distribution function from local thermal
            equilibrium.

        Returns
        -------
        Deltas : BoltzmannDeltas
            Defined in equation (15) of [LC22]_. A collection of 4 arrays,
            each of which is of size :py:data:`len(z)`.
        """
        # checking if result pre-computed
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()

        # getting (optimistic) estimate of truncation error
        truncationError = self.estimateTruncationError(deltaF)

        # getting criteria for validity of linearization
        criterion1, criterion2 = self.checkLinearization(deltaF)

        particles = self.offEqParticles

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            ("Array", self.basisM, self.basisN, self.basisN),
            ("Array", "z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(("Array", "Cardinal", "Cardinal", "Cardinal"))

        # Take all field-space points, but throw the boundary points away
        # TODO: LN: why throw away boundary points?
        field = self.background.fieldProfiles.takeSlice(
            1, -1, axis=self.background.fieldProfiles.overFieldPoints
        )

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues[None, None, :, None]
        pp = self.grid.ppValues[None, None, None, :]
        msq = np.array([particle.msqVacuum(field) for particle in particles])[
            :, :, None, None
        ]
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz[None, None, :, None]
        dppdrp = dppdrp[None, None, None, :]

        # base integrand, for '00'
        integrand = dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)

        Delta00 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), integrand
        )
        Delta02 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), pz**2 * integrand
        )
        Delta20 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), energy**2 * integrand
        )
        Delta11 = deltaFPoly.integrate(  # pylint: disable=invalid-name
            (2, 3), energy * pz * integrand
        )

        Deltas = BoltzmannDeltas(  # pylint: disable=invalid-name
            Delta00=Delta00, Delta02=Delta02, Delta20=Delta20, Delta11=Delta11
        )

        # returning results
        return BoltzmannResults(
            deltaF=deltaF,
            Deltas=Deltas,
            truncationError=truncationError,
            linearizationCriterion1=criterion1,
            linearizationCriterion2=criterion2,
        )

    def solveBoltzmannEquations(self) -> np.ndarray:
        r"""
        Solves Boltzmann equation for :math:`\delta f`, equation (32) of [LC22].

        Equations are of the form

        .. math::
            \mathcal{O}_{ijk,abc} \delta f_{abc} = \mathcal{S}_{ijk},

        where letters from the middle of the alphabet denote points on the
        coordinate lattice :math:`\{\xi_i,p_{z,j},p_{\Vert,k}\}`, and letters
        from the beginning of the alphabet denote elements of the basis of
        spectral functions :math:`\{\bar{T}_a, \bar{T}_b, \tilde{T}_c\}`.

        Parameters
        ----------

        Returns
        -------
        delta_f : array_like
            The deviation from equilibrium, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.

        References
        ----------
        .. [LC22] B. Laurent and J. M. Cline, First principles determination
            of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
            doi:10.1103/PhysRevD.106.023501
        """

        # contructing the various terms in the Boltzmann equation
        operator, source, _, _ = self.buildLinearEquations()

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # returning result
        deltaFShape = (
            len(self.offEqParticles),
            self.grid.M - 1,
            self.grid.N - 1,
            self.grid.N - 1,
        )
        deltaF = np.reshape(deltaF, deltaFShape, order="C")

        return deltaF

    def estimateTruncationError(self, deltaF: np.ndarray) -> float:
        r"""
        Quick estimate of the polynomial truncation error using
        John Boyd's Rule-of-thumb-2: the last coefficient of a Chebyshev
        polynomial expansion is the same order-of-magnitude as the truncation
        error.

        Parameters
        ----------
        deltaF : array_like
            The solution for which to estimate the truncation error,
            a rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.

        Returns
        -------
        truncationError : float
            Estimate of the relative trucation error.
        """
        # constructing Polynomial
        basisTypes = ("Array", self.basisM, self.basisN, self.basisN)
        basisNames = ("Array", "z", "pz", "pp")
        deltaFPoly = Polynomial(deltaF, self.grid, basisTypes, basisNames, False)

        # sum(|deltaF|) as the norm
        deltaFPoly.changeBasis(("Array", "Chebyshev", "Chebyshev", "Chebyshev"))
        deltaFMeanAbs = np.sum(
            np.abs(deltaFPoly.coefficients),
            axis=(1, 2, 3),
        )

        # estimating truncation errors in each direction
        truncationErrorChi = np.sum(
            np.abs(deltaFPoly.coefficients[:, -1, :, :]),
            axis=(1, 2),
        )
        truncationErrorPz = np.sum(
            np.abs(deltaFPoly.coefficients[:, :, -1, :]),
            axis=(1, 2),
        )
        truncationErrorPp = np.sum(
            np.abs(deltaFPoly.coefficients[:, :, :, -1]),
            axis=(1, 2),
        )

        # estimating the total truncation error as the maximum of these three
        return max(  # type: ignore[no-any-return]
            np.max(truncationErrorChi / deltaFMeanAbs),
            np.max(truncationErrorPz / deltaFMeanAbs),
            np.max(truncationErrorPp / deltaFMeanAbs),
        )

    def checkLinearization(
        self, deltaF: typing.Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute two criteria to verify the validity of the linearization of the
        Boltzmann equation: :math:`\delta f/f_{eq}` and :math:`C[\delta f]/L[\delta f]`.
        To be valid, at least one of the two criteria must be small for each particle.

        Parameters
        ----------
        deltaF : array-like, optional
            Solution of the Boltzmann equation. The default is None.

        Returns
        -------
        deltaFCriterion : tuple
        collCriterion : tuple
            Criteria for the validity of the linearization.

        """
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()

        particles = self.offEqParticles

        # constructing Polynomial class from deltaF array
        deltaFPoly = Polynomial(
            deltaF,
            self.grid,
            ("Array", self.basisM, self.basisN, self.basisN),
            ("z", "z", "pz", "pp"),
            False,
        )
        deltaFPoly.changeBasis(("Array", "Cardinal", "Cardinal", "Cardinal"))

        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )
        fieldPoly = Polynomial(
            np.sum(self.background.fieldProfiles, axis=1),
            self.grid,
            "Cardinal",
            "z",
            True,
        )
        dfielddChi = fieldPoly.derivative(0).coefficients[None, 1:-1, None, None]

        # adding new axes, to make everything rank 3 like deltaF (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        pz = self.grid.pzValues[None, None, :, None]
        pp = self.grid.ppValues[None, None, None, :]
        msq = msqFull[:, 1:-1, None, None]
        # constructing energy with (z, pz, pp) axes
        energy = np.sqrt(msq + pz**2 + pp**2)

        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        fEq = BoltzmannSolver._feq(energy / temperature, statistics)
        fEqPoly = Polynomial(
            fEq,
            self.grid,
            ("Array", "Cardinal", "Cardinal", "Cardinal"),
            ("z", "z", "pz", "pp"),
            False,
        )

        _, dpzdrz, dppdrp = self.grid.getCompactificationDerivatives()
        dpzdrz = dpzdrz[None, None, :, None]
        dppdrp = dppdrp[None, None, None, :]

        # base integrand, for '00'
        integrand = dfielddChi * dpzdrz * dppdrp * pp / (4 * np.pi**2 * energy)

        # The first criterion is to require that pressureOut/pressureEq is small
        pressureOut = deltaFPoly.integrate((1, 2, 3), integrand).coefficients
        pressureEq = fEqPoly.integrate((1, 2, 3), integrand).coefficients
        deltaFCriterion = pressureOut / pressureEq

        # If criterion1 is large, we need C[deltaF]/L[deltaF] to be small
        _, _, liouville, collision = self.buildLinearEquations()
        collisionDeltaF = np.sum(
            collision * deltaF[None, None, None, None, ...], axis=(4, 5, 6, 7)
        )
        liouvilleDeltaF = np.sum(
            liouville * deltaF[None, None, None, None, ...], axis=(4, 5, 6, 7)
        )
        collisionDeltaFPoly = Polynomial(
            collisionDeltaF,
            self.grid,
            ("Array", "Cardinal", "Cardinal", "Cardinal"),
            ("z", "z", "pz", "pp"),
            False,
        )
        lioviilleDeltaFPoly = Polynomial(
            liouvilleDeltaF,
            self.grid,
            ("Array", "Cardinal", "Cardinal", "Cardinal"),
            ("z", "z", "pz", "pp"),
            False,
        )
        collisionDeltaFIntegrated = collisionDeltaFPoly.integrate(
            (1, 2, 3), integrand
        ).coefficients
        liovilleDeltaFIntegrated = lioviilleDeltaFPoly.integrate(
            (1, 2, 3), integrand
        ).coefficients
        collCriterion = collisionDeltaFIntegrated / liovilleDeltaFIntegrated

        return deltaFCriterion, collCriterion

    def buildLinearEquations(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """

        particles = self.offEqParticles

        # coordinates
        xi, pz, pp = self.grid.getCoordinates()  # non-compact
        # adding new axes, to make everything rank 3 like deltaF, (z, pz, pp)
        # for fast multiplication of arrays, using numpy's broadcasting rules
        xi = xi[None, :, None, None]
        pz = pz[None, None, :, None]
        pp = pp[None, None, None, :]

        # compactified coordinates
        # chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)

        # background profiles
        temperatureFull = self.background.temperatureProfile
        vFull = self.background.velocityProfile
        msqFull = np.array(
            [
                particle.msqVacuum(self.background.fieldProfiles)
                for particle in particles
            ]
        )
        velocityWall = self.background.velocityWall

        # expanding to be rank 3 arrays, like deltaF
        temperature = self.background.temperatureProfile[None, 1:-1, None, None]
        v = vFull[None, 1:-1, None, None]
        msq = msqFull[:, 1:-1, None, None]
        energy = np.sqrt(msq + pz**2 + pp**2)

        # fluctuation mode
        statistics = np.array(
            [-1 if particle.statistics == "Fermion" else 1 for particle in particles]
        )[:, None, None, None]

        # building parts which depend on the 'derivatives' argument
        if self.derivatives == "Spectral":
            # fit the background profiles to polynomials
            temperaturePoly = Polynomial(
                temperatureFull,
                self.grid,
                "Cardinal",
                "z",
                True,
            )
            vPoly = Polynomial(vFull, self.grid, "Cardinal", "z", True)
            msqPoly = Polynomial(
                msqFull, self.grid, ("Array", "Cardinal"), ("Array", "z"), True
            )
            # intertwiner matrices
            intertwinerChiMat = temperaturePoly.matrix(self.basisM, "z")
            intertwinerRzMat = temperaturePoly.matrix(self.basisN, "pz")
            intertwinerRpMat = temperaturePoly.matrix(self.basisN, "pp")
            # derivative matrices
            derivMatrixChi = temperaturePoly.derivMatrix(self.basisM, "z")[1:-1]
            derivMatrixRz = temperaturePoly.derivMatrix(self.basisN, "pz")[1:-1]
            # spatial derivatives of profiles
            dTemperaturedChi = temperaturePoly.derivative(0).coefficients[
                None, 1:-1, None, None
            ]
            dvdChi = vPoly.derivative(0).coefficients[None, 1:-1, None, None]
            dMsqdChi = msqPoly.derivative(1).coefficients[:, 1:-1, None, None]
        else:  # self.derivatives == "Finite Difference"
            # intertwiner matrices are simply unit matrices
            # as we are in the (Cardinal, Cardinal) basis
            intertwinerChiMat = np.identity(self.grid.M - 1)
            intertwinerRzMat = np.identity(self.grid.N - 1)
            intertwinerRpMat = np.identity(self.grid.N - 1)
            # derivative matrices
            chiFull, rzFull, _ = self.grid.getCompactCoordinates(endpoints=True)
            derivOperatorChi = findiff.FinDiff((0, chiFull, 1), acc=2)
            derivMatrixChi = derivOperatorChi.matrix((self.grid.M + 1,))
            derivOperatorRz = findiff.FinDiff((0, rzFull, 1), acc=2)
            derivMatrixRz = derivOperatorRz.matrix((self.grid.N + 1,))
            # spatial derivatives of profiles, endpoints used for taking
            # derivatives but then dropped as deltaF fixed at 0 at endpoints
            dTemperaturedChi = (derivMatrixChi @ temperatureFull)[
                None, 1:-1, None, None
            ]
            dvdChi = (derivMatrixChi @ temperatureFull)[None, 1:-1, None, None]
            # the following is equivalent to:
            # dMsqdChiEinsum = np.einsum(
            #   "ij,aj->ai", derivMatrixChi.toarray(), msqFull
            # )[:, 1:-1, None, None]
            dMsqdChi = np.sum(
                derivMatrixChi.toarray()[None, :, :] * msqFull[:, None, :],
                axis=-1,
            )[:, 1:-1, None, None]
            # restructuring derivative matrices to appropriate forms for
            # Liouville operator
            derivMatrixChi = derivMatrixChi.toarray()[1:-1, 1:-1]
            derivMatrixRz = derivMatrixRz.toarray()[1:-1, 1:-1]

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - velocityWall**2)
        momentumWall = gammaWall * (pz - velocityWall * energy)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        energyPlasma = gammaPlasma * (energy - v * pz)
        momentumPlasma = gammaPlasma * (pz - v * energy)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (velocityWall - v)

        # (exact) derivatives of compactified coordinates
        dxidchi, dpzdrz, _ = self.grid.getCompactificationDerivatives()
        dchidxi = 1 / dxidchi[None, :, None, None]
        drzdpz = 1 / dpzdrz[None, None, :, None]

        # derivative of equilibrium distribution
        dfEq = BoltzmannSolver._dfeq(energyPlasma / temperature, statistics)

        ##### source term #####
        # Given by S_i on the RHS of Eq. (5) in 2204.13120, with further details
        # given in Eq. (6).
        source = (
            (dfEq / temperature)
            * dchidxi
            * (
                momentumWall * momentumPlasma * gammaPlasma**2 * dvdChi
                + momentumWall * energyPlasma * dTemperaturedChi / temperature
                + 1 / 2 * dMsqdChi * uwBaruPl
            )
        )

        ##### liouville operator #####
        # Given in the LHS of Eq. (5) in 2204.13120, with further details given
        # by the second line of Eq. (32).
        identityParticles = np.identity(len(particles))[
            :, None, None, None, :, None, None, None
        ]
        liouville = identityParticles * (
            dchidxi[:, :, :, :, None, None, None, None]
            * momentumWall[:, :, :, :, None, None, None, None]
            * derivMatrixChi[None, :, None, None, None, :, None, None]
            * intertwinerRzMat[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
            - dchidxi[:, :, :, :, None, None, None, None]
            * drzdpz[:, :, :, :, None, None, None, None]
            * (gammaWall / 2)
            * dMsqdChi[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * derivMatrixRz[None, None, :, None, None, None, :, None]
            * intertwinerRpMat[None, None, None, :, None, None, None, :]
        )
        """
        An alternative, but slower, implementation is given by the following:
        liouville = (
            np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                dchidxi * PWall,
                derivChi,
                TRzMat,
                TRpMat,
                optimize=True,
            )
            - np.einsum(
                "ijk, ia, jb, kc -> ijkabc",
                gammaWall / 2 * dchidxi * drzdpz * dmsqdChi,
                TChiMat,
                derivRz,
                TRpMat,
                optimize=True,
            )
        )
        """

        # including factored-out T^2 in collision integrals
        collision = self.collisionMultiplier * (
            (temperature**2)[:, :, :, :, None, None, None, None]
            * intertwinerChiMat[None, :, None, None, None, :, None, None]
            * self.collisionArray[:, None, :, :, :, None, :, :]
        )
        ##### total operator #####
        operator = liouville + collision

        # reshaping indices
        totalSize = (
            len(particles) * (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        )
        source = np.reshape(source, totalSize, order="C")
        operator = np.reshape(operator, (totalSize, totalSize), order="C")

        # returning results
        return operator, source, liouville, collision

    def loadCollisions(self, directoryPath: "pathlib.Path") -> None:
        """
        Loads collision files for use with the Boltzmann solver.

        Args:
            directoryPath (pathlib.Path): Directory containing the .hdf5 collision data.

        Returns:
            None

        Raises:
            CollisionLoadError
        """
        try:
            self.collisionArray = CollisionArray.newFromDirectory(
                directoryPath,
                self.grid,
                self.basisN,
                self.offEqParticles,
            )
            logging.debug(f"Loaded collision data from directory {directoryPath}")
        except CollisionLoadError as e:
            raise

    @staticmethod
    def _checkBasis(basis: str) -> None:
        """
        Check that basis is recognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, f"BoltzmannSolver error: unkown basis {basis}"

    @staticmethod
    def _checkDerivatives(derivatives: str) -> None:
        """
        Check that derivative option is recognised
        """
        derivativesOptions = ["Spectral", "Finite Difference"]
        assert (
            derivatives in derivativesOptions
        ), f"BoltzmannSolver error: unkown derivatives option {derivatives}"

    @staticmethod
    def _feq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Thermal distribution functions, Bose-Einstein and Fermi-Dirac
        """
        x = np.asarray(x)
        return np.where(
            x > BoltzmannSolver.MAX_EXPONENT,
            0,
            1 / (np.exp(x) - statistics),
        )

    @staticmethod
    def _dfeq(x: np.ndarray, statistics: int | np.ndarray) -> np.ndarray:
        """
        Temperature derivative of thermal distribution functions
        """
        x = np.asarray(x)
        return np.where(
            x > BoltzmannSolver.MAX_EXPONENT,
            -0,
            -1 / (np.exp(x) - 2 * statistics + np.exp(-x)),
        )
