import os
import warnings
import numpy as np
from copy import deepcopy
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from .Grid import Grid
from .Polynomial2 import Polynomial
from .model import Particle
from .helpers import boostVelocity
from .WallGoUtils import getSafePathToResource

class BoltzmannBackground:
    def __init__(
        self,
        velocityMid,
        velocityProfile,
        fieldProfile,
        temperatureProfile,
        polynomialBasis="Cardinal",
    ):
        # assumes input is in the wall frame
        self.vw = 0
        self.velocityProfile = np.asarray(velocityProfile)
        self.fieldProfile = np.asarray(fieldProfile)
        self.temperatureProfile = np.asarray(temperatureProfile)
        self.polynomialBasis = polynomialBasis
        self.vMid = velocityMid
        self.TMid = 0.5 * (temperatureProfile[0] + temperatureProfile[-1])

    def boostToPlasmaFrame(self):
        """
        Boosts background to the plasma frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.vMid)
        self.vw = boostVelocity(self.vw, self.vMid)

    def boostToWallFrame(self):
        """
        Boosts background to the wall frame
        """
        self.velocityProfile = boostVelocity(self.velocityProfile, self.vw)
        self.vw = 0


class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    def __init__(
        self,
        grid,
        background,
        particle,
        basisM="Cardinal",
        basisN="Chebyshev",
    ):
        """
        Initialsation of BoltzmannSolver

        Parameters
        ----------
        grid : Grid
            An object of the Grid class.
        background : Background
            An object of the Background class.
        mode : Mode
            An object of the Mode class.

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """
        self.grid = grid
        self.background = deepcopy(background)
        self.background.boostToPlasmaFrame()
        self.particle = particle
        BoltzmannSolver.__checkBasis(basisM)
        BoltzmannSolver.__checkBasis(basisN)
        self.basisM = basisM
        self.basisN = basisN

        ##### collision operator #####
        collisionFile = self.__collisionFilename()
        self.readCollision(collisionFile, self.particle)

    def getDeltas(self, deltaF=None):
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (15) of [LC22]_.

        Parameters
        ----------
        deltaF : array_like, optional
            The deviation of the distribution function from local thermal
            equilibrium.

        Returns
        -------
        Deltas : array_like
            Defined in equation (15) of [LC22]_. A list of 4 arrays, each of
            which is of size :py:data:`len(z)`.
        """
        # checking if result pre-computed
        if deltaF is None:
            deltaF = self.solveBoltzmannEquations()

        # dict to store results
        Deltas = {"00": 0, "02": 0, "20": 0, "11": 0}

        # constructing polynomial representation and changing to Cardinal basis
        basisTypes = (self.basisM, self.basisN, self.basisN)
        basisNames = ('z','pz','pp')
        deltaFPoly = Polynomial(
            deltaF, self.grid, basisTypes, basisNames, False
        )
        deltaFPoly.changeBasis('Cardinal')

        # introducing some shorthands for equations below
        field = self.background.fieldProfile[:, 1:-1]
        msq = self.particle.msqVacuum(field)[:, None, None]
        pz = self.grid.pzValues[None, :, None]
        pp = self.grid.ppValues[None, None, :]
        TMid = self.background.TMid
        rz = self.grid.rzValues[None, :, None]
        rp = self.grid.rpValues[None, None, :]

        # base integrand
        E = np.sqrt(msq + pz**2 + pp**2)
        dpzdrz = 2 * TMid / (1 - rz**2)
        dppdrp = TMid / (1 - rp)
        integrand = dpzdrz * dppdrp * pp / (4 * np.pi**2 * E)

        # integrations
        Deltas['00'] = deltaFPoly.integrate((1,2), integrand)
        Deltas['20'] = deltaFPoly.integrate((1,2), E**2 * integrand)
        Deltas['02'] = deltaFPoly.integrate((1,2), pz**2 * integrand)
        Deltas['11'] = deltaFPoly.integrate((1,2), E * pz * integrand)

        # returning results
        return Deltas

    def solveBoltzmannEquations(self):
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
        print("Starting solveBoltzmannEquations")
        operator, source, liouville, collision = self.buildLinearEquations()

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # returning result
        deltaFShape = (self.grid.M - 1, self.grid.N - 1, self.grid.N - 1)
        deltaF = np.reshape(deltaF, deltaFShape, order="C")

        # testing result
        source = np.reshape(source, deltaFShape, order="C")
        truncationError = self.estimateTruncationError(deltaF)
        print(f"{truncationError=}")
        finiteDifferenceError = self.estimateFiniteDifferenceError(
                deltaF, source, collision
        )
        print(f"{finiteDifferenceError=}")

        return deltaF

    def estimateTruncationError(self, deltaF):
        r"""
        Quick estimate of the polynomial truncation error using
        John Boyd's Rule-of-thumb-2:
            the last coefficient of a Chebyshev polynomial expansion
            is the same order-of-magnitude as the truncation error.

        Returns
        -------
        truncationError : float
            Estimate of the relative trucation error.
        """
        # constructing Polynomial
        basisTypes = (self.basisM, self.basisN, self.basisN)
        basisNames = ('z', 'pz', 'pp')
        deltaFPoly = Polynomial(
            deltaF, self.grid, basisTypes, basisNames, False
        )

        # mean(|deltaF|) in the Cardinal basis as the norm
        deltaFPoly.changeBasis('Cardinal')
        deltaFMeanAbs = np.mean(abs(deltaFPoly.coefficients[:, :, :]))

        # last coefficient in Chebyshev basis estimates error
        deltaFPoly.changeBasis('Chebyshev')

        # estimating truncation errors in each direction
        truncationErrorChi = np.mean(abs(deltaFPoly.coefficients[-1, :, :]))
        truncationErrorPz = np.mean(abs(deltaFPoly.coefficients[:, -1, :]))
        truncationErrorPp = np.mean(abs(deltaFPoly.coefficients[:, :, -1]))

        # estimating the total truncation error as the maximum of these three
        return (
            max((truncationErrorChi, truncationErrorPz, truncationErrorPp))
            / deltaFMeanAbs
        )

    def estimateFiniteDifferenceError(self, deltaF, source, collision):
        r"""
        Tests the validity of a solution of the Boltzmann equation.

        Parameters
        ----------
        solution : array_like
            The solution to be tested, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.

        Returns
        -------

        """
        # the right hand side, to reproduce, with collision term removed
        rhs = source - np.einsum(
            "ijkabc, abc -> ijk",
            collision,
            deltaF,
        )

        # moving into Cardinal basis
        basisTypes = (self.basisM, self.basisN, self.basisN)
        basisNames = ('z', 'pz', 'pp')
        deltaFPoly = Polynomial(
            deltaF, self.grid, basisTypes, basisNames, False
        )
        deltaFPoly.changeBasis('Cardinal')

        # starting computation of the left hand side, with finite differences
        chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)
        xi, pz, pp = self.grid.getCoordinates(endpoints=False)
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]
        dchidxi, drzdpz, drpdpp = self.grid.getCompactificationDerivatives()

        vw = self.background.vw
        gammaWall = 1 / np.sqrt(1 - vw**2)

        field = self.background.fieldProfile[:, 1:-1, np.newaxis, np.newaxis]
        msq = self.particle.msqVacuum(field)
        E = np.sqrt(msq + pz**2 + pp**2)
        msqpoly = Polynomial(
            self.particle.msqVacuum(self.background.fieldProfile),
            self.grid,
            'Cardinal',
            'z',
            True,
        )
        PWall = gammaWall * (pz - vw * E)

        # msq finite difference derivative
        h = 1e-4
        dMsqdChi = (
            -1 / 12 * msqpoly.evaluate(np.array([chi + 2 * h]))
            + 2 / 3 * msqpoly.evaluate(np.array([chi + h]))
            - 2 / 3 * msqpoly.evaluate(np.array([chi - h]))
            + 1 / 12 * msqpoly.evaluate(np.array([chi - 2 * h]))
        ) / h

        # d(deltaF)/d(chi) finite difference derivative
        dFdChi = np.zeros_like(deltaF)
        # OG: Is there a better way to write this than with the for loop?
        for i_rz in range(self.grid.N - 1):
            for i_rp in range(self.grid.N - 1):
                # OG: This seems a clunky construction
                rz_to_stack = [rz[i_rz]] * len(rz)
                rp_to_stack = [rp[i_rp]] * len(rp)
                # OG: Can we use derivative() from helpers.py here?
                dFdChi[:, i_rz, i_rp] = (
                    -1 / 12 * deltaFPoly.evaluate(np.stack((chi + 2 * h, rz_to_stack, rp_to_stack), axis=0))
                    + 2 / 3 * deltaFPoly.evaluate(np.stack((chi + h, rz_to_stack, rp_to_stack), axis=0))
                    - 2 / 3 * deltaFPoly.evaluate(np.stack((chi - h, rz_to_stack, rp_to_stack), axis=0))
                    + 1 / 12 * deltaFPoly.evaluate(np.stack((chi - 2 * h, rz_to_stack, rp_to_stack), axis=0))
                ) / h

        # d(deltaF)/d(rz) finite difference derivative
        dFdRz = np.zeros_like(deltaF)
        for i_chi in range(self.grid.M - 1):
            for i_rp in range(self.grid.N - 1):
                chi_to_stack = [chi[i_chi]] * len(chi)
                rp_to_stack = [rp[i_rp]] * len(rp)
                dFdRz[i_chi, :, i_rp] = (
                    -1 / 12 * deltaFPoly.evaluate(np.stack((chi_to_stack, rz + 2 * h, rp_to_stack), axis=0))
                    + 2 / 3 * deltaFPoly.evaluate(np.stack((chi_to_stack, rz + h, rp_to_stack), axis=0))
                    - 2 / 3 * deltaFPoly.evaluate(np.stack((chi_to_stack, rz - h, rp_to_stack), axis=0))
                    + 1 / 12 * deltaFPoly.evaluate(np.stack((chi_to_stack, rz - 2 * h, rp_to_stack), axis=0))
                ) / h

        lhs = (
            dchidxi[:, np.newaxis, np.newaxis] * (
                PWall * dFdChi
                - gammaWall / 2 * (
                    drzdpz[np.newaxis, :, np.newaxis]
                    * dMsqdChi[:, np.newaxis, np.newaxis]
                    * dFdRz
                )
            )
        )

        # quantifying discrepancy with finite difference approximation
        return np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)


    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        # coordinates
        xi, pz, pp = self.grid.getCoordinates(endpoints=False)  # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # compactified coordinates
        chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False) # compact

        # background profiles
        T = self.background.temperatureProfile[1:-1, np.newaxis, np.newaxis]
        field = self.background.fieldProfile[:, 1:-1, np.newaxis, np.newaxis]
        v = self.background.velocityProfile[1:-1, np.newaxis, np.newaxis]
        vw = self.background.vw

        # fluctuation mode
        statistics = -1 if self.particle.statistics == "Fermion" else 1
        # TODO: indices order not consistent across different functions.
        msq = self.particle.msqVacuum(field)
        E = np.sqrt(msq + pz**2 + pp**2)

        # fit the background profiles to polynomials
        Tpoly = Polynomial(
            self.background.temperatureProfile,
            self.grid,
            'Cardinal',
            'z',
            True,
        )
        msqpoly = Polynomial(
            self.particle.msqVacuum(self.background.fieldProfile),
            self.grid,
            'Cardinal',
            'z',
            True,
        )
        vpoly = Polynomial(
            self.background.velocityProfile,
            self.grid,
            'Cardinal',
            'z',
            True,
        )

        # intertwiner matrices
        TChiMat = Tpoly.matrix(self.basisM, "z")
        TRzMat = Tpoly.matrix(self.basisN, "pz")
        TRpMat = Tpoly.matrix(self.basisN, "pp")

        # derivative matrices
        derivChi = Tpoly.derivMatrix(self.basisM, "z")[1:-1]
        derivRz = Tpoly.derivMatrix(self.basisN, "pz")[1:-1]

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - vw**2)
        PWall = gammaWall * (pz - vw * E)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        EPlasma = gammaPlasma * (E - v * pz)
        PPlasma = gammaPlasma * (pz - v * E)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (vw - v)

        # spatial derivatives of profiles
        dTdChi = Tpoly.derivative(0).coefficients[1:-1, None, None]
        dvdChi = vpoly.derivative(0).coefficients[1:-1, None, None]
        dmsqdChi = msqpoly.derivative(0).coefficients[1:-1, None, None]

        # derivatives of compactified coordinates
        dchidxi, drzdpz, drpdpp = self.grid.getCompactificationDerivatives()
        dchidxi = dchidxi[:, np.newaxis, np.newaxis]
        drzdpz = drzdpz[np.newaxis, :, np.newaxis]

        # equilibrium distribution, and its derivative
        warnings.filterwarnings("ignore", message="overflow encountered in exp")
        fEq = BoltzmannSolver.__feq(EPlasma / T, statistics)
        dfEq = BoltzmannSolver.__dfeq(EPlasma / T, statistics)
        warnings.filterwarnings(
            "default", message="overflow encountered in exp"
        )

        ##### source term #####
        source = (dfEq / T) * dchidxi * (
            PWall * PPlasma * gammaPlasma**2 * dvdChi
            + PWall * EPlasma * dTdChi / T
            + 1 / 2 * dmsqdChi * uwBaruPl
        )

        ##### liouville operator #####
        liouville = (
            dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * PWall[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * derivChi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
            - dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * drzdpz[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * gammaWall / 2
                * dmsqdChi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * derivRz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        )

        # including factored-out T^2 in collision integrals
        collision = (
            (T ** 2)[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * self.collisionArray[np.newaxis, :, :, np.newaxis, :, :]
        )

        ##### total operator #####
        operator = liouville + collision

        # reshaping indices
        N_new = (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        source = np.reshape(source, N_new, order="C")
        operator = np.reshape(operator, (N_new, N_new), order="C")

        # n = self.grid.N-1
        # Nnew = n**2
        # eigs = np.linalg.eigvals(self.background.TMid**2*((self.collisionArray/PWall[-1,:,:,None,None]).reshape((n,n,Nnew))).reshape((Nnew,Nnew)))
        # eigs2 = np.linalg.eigvals(self.background.TMid**2*((self.collisionArray).reshape((n,n,Nnew))).reshape((Nnew,Nnew)))
        # print(np.sort(1/np.abs(np.real(eigs)))[-4:],np.sort(1/np.abs(np.real(eigs2)))[-4:])

        # returning results
        return operator, source, liouville, collision

    def readCollision(self, collisionFile, particle):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22]_.
        """
        try:
            with h5py.File(collisionFile, "r") as file:
                metadata = file["metadata"]
                basisSize = metadata.attrs["Basis Size"]
                basisType = codecs.decode(
                    metadata.attrs["Basis Type"], 'unicode_escape',
                )
                BoltzmannSolver.__checkBasis(basisType)

                # LN: currently the dataset names are of form
                # "particle1, particle2". Here it's just "top, top" for now.
                datasetName = particle.name + ", " + particle.name
                collisionArray = np.array(file[datasetName][:])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise

        # converting between conventions
        self.collisionArray = np.transpose(
            np.flip(collisionArray, (2, 3)),
            (2, 3, 0, 1),
        )

        if self.basisN != basisType:
            print("--------------------")
            print(f"Changing basis of collision integrals in BoltzmannSolver from {basisType} to {self.basisN}")
            print("Testing two different implementations")

            # OG: The following is equivalent to Benoit's original implementation
            collisionPoly = Polynomial(
                self.collisionArray,
                self.grid,
                ("Cardinal", "Cardinal", basisType, basisType),
                ("pz", "pp", "pz", "pp"),
                False,
            )
            Tn1 = collisionPoly.matrix(basisType, "pz", endpoints=False)
            Tn2 = collisionPoly.matrix(basisType, "pp", endpoints=False)
            self.collisionArray = np.einsum(
                "ec, fd, abef -> abcd",
                np.linalg.inv(Tn1),
                np.linalg.inv(Tn2),
                self.collisionArray,
                optimize=True,
            )

            # OG: Why doesn't the following work?
            collisionPoly.changeBasis(
                ("Cardinal", "Cardinal", self.basisN, self.basisN)
            )
            normOriginal = np.linalg.norm(self.collisionArray)
            normAlt = np.linalg.norm(np.asarray(collisionPoly))
            normDiff = np.linalg.norm(self.collisionArray - np.asarray(collisionPoly))
            print(f"norm(original) = {normOriginal}")
            print(f"norm(alt)      = {normAlt}, should equal norm(original)")
            print(f"norm(diff)     = {normDiff / normOriginal}, should be << 1")
            print("--------------------")

    def __collisionFilename(self):
        """
        A filename convention for collision integrals.
        """
        # LN: This will need generalization. And do we want just one gargantuan
        # file with all out-of-eq pairs, or are individual files better?

        suffix = "hdf5"
        fileName = f"collisions_top_top_N{self.grid.N}.{suffix}"
        return getSafePathToResource("Data/" + fileName)


    def __checkBasis(basis):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "BoltzmannSolver error: unkown basis %s" % basis

    @staticmethod
    def __feq(x, statistics):
        if np.isclose(statistics, 1, atol=1e-14):
            return 1 / np.expm1(x)
        else:
            return 1 / (np.exp(x) + 1)

    @staticmethod
    def __dfeq(x, statistics):
        x = np.asarray(x)
        if np.isclose(statistics, 1, atol=1e-14):
            return np.where(x > 100, -np.exp(-x), -np.exp(x) / np.expm1(x) ** 2)
        else:
            return np.where(
                x > 100, -np.exp(-x), -1 / (np.exp(x) + 2 + np.exp(-x))
            )
