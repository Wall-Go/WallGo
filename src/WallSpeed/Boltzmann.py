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
from .WallGoUtils import getPackagedDataPath

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
        source = np.reshape(source, ((self.grid.M - 1), (self.grid.N - 1), (self.grid.N - 1)), order="C")
        self.testSolution(deltaF, source, collision)

        return result

    def testSolution(self, deltaF, source, collision):
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
        print("\nStarting testSolution")
        # the right hand side, to reproduce
        rhs = np.einsum(
            "ijkabc, abc -> ijk",
            collision,
            deltaF,
        )
        rhs = rhs - source

        # moving into Cardinal basis
        print("moving into Cardinal basis")
        basisTypes = (self.basisM, self.basisN, self.basisN)
        basisNames = ('z', 'pz', 'pp')
        deltaFPoly = Polynomial(
            deltaF, self.grid, basisTypes, basisNames, False
        )
        deltaFPoly.changeBasis('Cardinal')

        # the left hand side, finite differences
        print("the left hand side, finite differences")
        chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)
        xi, pz, pp = self.grid.getCoordinates(endpoints=False)
        xi = xi[:, np.newaxis, np.newaxis]
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

        # checking msq derivatives
        print("\ntesting msq derivatives")
        h = 1e-4
        deriv1 = (
            msqpoly.evaluate(np.array([chi + h]))
            - msqpoly.evaluate(np.array([chi - h]))
        ) / (2 * h)
        deriv2 = np.asarray(msqpoly.derivative(axis=0))[1:-1]
        diffMsqDeriv = np.linalg.norm(deriv1 - deriv2) / np.linalg.norm(deriv1)
        print(f"diff dMsq/dchi = {diffMsqDeriv}")

        # checking field derivatives
        print("\ntesting field derivatives")
        deriv1 = (
            deltaFPoly.evaluate(np.array([chi + h, rz, rp]))
            - deltaFPoly.evaluate(np.array([chi - h, rz, rp]))
        ) / (2 * h)
        deriv2 = np.asarray(deltaFPoly.derivative(axis=0))[1:-1, :, :]
        diffFieldDeriv = np.linalg.norm(deriv1 - deriv2) / np.linalg.norm(deriv1)
        print(f"diff dField/dchi = {diffFieldDeriv}")
        
        deriv1 = (
            deltaFPoly.evaluate(np.array([chi, rz + h, rp]))
            - deltaFPoly.evaluate(np.array([chi, rz - h, rp]))
        ) / (2 * h)
        deriv2 = np.asarray(deltaFPoly.derivative(axis=1))[:, 1:-1, :]
        diffFieldDeriv = np.linalg.norm(deriv1 - deriv2) / np.linalg.norm(deriv1)
        print(f"diff dField/drz = {diffFieldDeriv}")

        # OG: should use derivative() from helpers.py, for more accurate results
        print("\nputting the left hand side together")
        print(f"Input shapes: {np.array([chi + h, rz, rp]).shape}")
        lhs = (
            dchidxi[:, np.newaxis, np.newaxis] * (
                PWall * (
                    deltaFPoly.evaluate(np.array([chi + h, rz, rp]))
                    - deltaFPoly.evaluate(np.array([chi - h, rz, rp]))
                ) / (2 * h)
                - gammaWall / 2 * drzdpz * (
                    msqpoly.evaluate(np.array([chi + h]))
                    - msqpoly.evaluate(np.array([chi - h]))
                ) / (2 * h) * (
                    deltaFPoly.evaluate(np.array([chi, rz + h, rp]))
                    - deltaFPoly.evaluate(np.array([chi, rz - h, rp]))
                ) / (2 * h)
            )
        )

        # quantifying discrepancy with finite difference approximation
        print("\n quantifying differences")
        lhsNorm = np.linalg.norm(lhs)
        rhsNorm = np.linalg.norm(rhs)
        diffFiniteDifference = np.linalg.norm(lhs - rhs) / rhsNorm
        print(f"{lhsNorm = }")
        print(f"{rhsNorm = }")
        print(f"{diffFiniteDifference = }")

        exit()

        # testing result
        # optimistic estimate of convergence in momentum directions
        convergenceOptimistic = (
            np.linalg.norm(result[:, -1, -1]) / np.linalg.norm(result[:, 0, 0])
        )
        print(f"{convergenceOptimistic = }")
        convergencePz = np.linalg.norm(result, axis=(0, 2))
        convergencePp = np.linalg.norm(result, axis=(0, 1))
        x = self.grid.rzValues
        y = np.log(convergencePz)
        popt, pcov = np.polyfit(x, y, 1, cov=True)
        print(f"fit parameters {popt[0]}({np.sqrt(pcov[0][0])}), {popt[1]}({np.sqrt(pcov[1][1])})")

        # testing other stuff
        basisTypes = (self.basisM, self.basisN, self.basisN)
        basisNames = ('z', 'pz', 'pp')
        deltaFPoly = Polynomial(
            result, self.grid, basisTypes, basisNames, False
        )
        deltaFPoly.changeBasis('Cardinal')
        exit()

        return result


    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        print("Starting buildLinearEquations")
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
        print("Building liouville operator")
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
        print("Building collision operator")
        collision = (
            (T ** 2)[:, :, :, np.newaxis, np.newaxis, np.newaxis]
            * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * self.collisionArray[np.newaxis, :, :, np.newaxis, :, :]
        )

        ##### total operator #####
        print("Summing operators")
        print(f"Shapes: {liouville.shape=}, {collision.shape=}")
        operator = liouville + collision

        # reshaping indices
        print("reshaping indices")
        N_new = (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        source = np.reshape(source, N_new, order="C")
        operator = np.reshape(operator, (N_new, N_new), order="C")

        # n = self.grid.N-1
        # Nnew = n**2
        # eigs = np.linalg.eigvals(self.background.TMid**2*((self.collisionArray/PWall[-1,:,:,None,None]).reshape((n,n,Nnew))).reshape((Nnew,Nnew)))
        # eigs2 = np.linalg.eigvals(self.background.TMid**2*((self.collisionArray).reshape((n,n,Nnew))).reshape((Nnew,Nnew)))
        # print(np.sort(1/np.abs(np.real(eigs)))[-4:],np.sort(1/np.abs(np.real(eigs2)))[-4:])

        # returning results
        print("Ending buildLinearEquations")
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
        return getPackagedDataPath("WallSpeed.Data", fileName)


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
