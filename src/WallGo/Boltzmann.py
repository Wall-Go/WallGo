import warnings
import numpy as np
from copy import deepcopy
import h5py  # read/write hdf5 structured binary data file format
import codecs  # for decoding unicode string from hdf5 file
import findiff  # finite difference methods
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
        collisionArray,
        derivatives="Spectral",
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
        particle : Particle
            An object of the Particle class
        collisionArray : CollisionArray
            An object of the CollisionArray class containing the collision
            integrals.
        derivatives : {'Spectral', 'Finite Difference'}
            Choice of method for computing derivatives. Default is 'Spectral'
            which is expected to be more accurate.
        basisM :  {'Chebyshev', 'Cardinal'}
            If `derivatives='Spectral'`, the Polynomial basis used
            in the chi direction.
        basisN :  {'Chebyshev', 'Cardinal'}
             If `derivatives='Spectral'`, the Polynomial basis used
            in the pz and pp directions.

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """
        self.grid = grid
        self.background = deepcopy(background)
        self.background.boostToPlasmaFrame()
        self.particle = particle
        self.collisionArray = collisionArray
        BoltzmannSolver.__checkDerivatives(derivatives)
        self.derivatives = derivatives
        BoltzmannSolver.__checkBasis(basisM)
        BoltzmannSolver.__checkBasis(basisN)
        if derivatives == "Finite Difference":
            assert basisM == "Cardinal" and basisN == "Cardinal", \
                "Must use Cardinal basis for Finite Difference method"
        self.basisM = basisM
        self.basisN = basisN

        ##### collision operator #####
        # collisionFile = self.__collisionFilename()
        # self.readCollision(collisionFile, self.particle)

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
        print("Starting solveBoltzmannEquations")

        # contructing the various terms in the Boltzmann equation
        operator, source, liouville, collision = self.buildLinearEquations()

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # returning result
        deltaFShape = (self.grid.M - 1, self.grid.N - 1, self.grid.N - 1)
        deltaF = np.reshape(deltaF, deltaFShape, order="C")

        # testing result
        source = np.reshape(source, deltaFShape, order="C")
        truncationError = self.estimateTruncationError(deltaF)
        print(f"(optimistic) estimate of truncation error = {truncationError}")

        return deltaF

    def estimateTruncationError(self, deltaF):
        r"""
        Quick estimate of the polynomial truncation error using
        John Boyd's Rule-of-thumb-2:
            the last coefficient of a Chebyshev polynomial expansion
            is the same order-of-magnitude as the truncation error.

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

    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """

        # initial coordinates
        xi, pz, pp = self.grid.getCoordinates(endpoints=False)

        # expanding to be rank 3 arrays, like deltaF
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # compactified coordinates
        chi, rz, rp = self.grid.getCompactCoordinates(endpoints=False)

        # background profiles
        TFull = self.background.temperatureProfile
        vFull = self.background.velocityProfile
        msqFull = self.particle.msqVacuum(self.background.fieldProfile)
        vw = self.background.vw

        # expanding to be rank 3 arrays, like deltaF
        T = TFull[1:-1, np.newaxis, np.newaxis]
        v = vFull[1:-1, np.newaxis, np.newaxis]
        msq = msqFull[1:-1, np.newaxis, np.newaxis]
        E = np.sqrt(msq + pz**2 + pp**2)

        # fluctuation mode
        statistics = -1 if self.particle.statistics == "Fermion" else 1

        # building parts which depend on the 'derivatives' argument
        if self.derivatives == "Spectral":
            # fit the background profiles to polynomials
            TPoly = Polynomial(TFull, self.grid, 'Cardinal', 'z', True)
            vPoly = Polynomial(vFull, self.grid, 'Cardinal', 'z', True)
            msqPoly = Polynomial(msqFull, self.grid, 'Cardinal', 'z', True)
            # intertwiner matrices
            TChiMat = TPoly.matrix(self.basisM, "z")
            TRzMat = TPoly.matrix(self.basisN, "pz")
            TRpMat = TPoly.matrix(self.basisN, "pp")
            # derivative matrices
            derivMatrixChi = TPoly.derivMatrix(self.basisM, "z")[1:-1]
            derivMatrixRz = TPoly.derivMatrix(self.basisN, "pz")[1:-1]
            # spatial derivatives of profiles
            dTdChi = TPoly.derivative(0).coefficients[1:-1, None, None]
            dvdChi = vPoly.derivative(0).coefficients[1:-1, None, None]
            dMsqdChi = msqPoly.derivative(0).coefficients[1:-1, None, None]
        elif self.derivatives == "Finite Difference":
            # intertwiner matrices are simply unit matrices
            # as we are in the (Cardinal, Cardinal) basis
            TChiMat = np.identity(self.grid.M - 1)
            TRzMat = np.identity(self.grid.N - 1)
            TRpMat = np.identity(self.grid.N - 1)
            # derivative matrices
            chiFull, rzFull, rpFull = self.grid.getCompactCoordinates(
                endpoints=True
            )
            derivOperatorChi = findiff.FinDiff((0, chiFull, 1), acc=2)
            derivMatrixChi = derivOperatorChi.matrix((self.grid.M + 1,))
            derivOperatorRz = findiff.FinDiff((0, rzFull, 1), acc=2)
            derivMatrixRz = derivOperatorRz.matrix((self.grid.N + 1,))
            # spatial derivatives of profiles, endpoints used for taking
            # derivatives but then dropped as deltaF fixed at 0 at endpoints
            dTdChi = (derivMatrixChi @ TFull)[1:-1, np.newaxis, np.newaxis]
            dvdChi = (derivMatrixChi @ vFull)[1:-1, np.newaxis, np.newaxis]
            dMsqdChi = (derivMatrixChi @ msqFull)[1:-1, np.newaxis, np.newaxis]
            # restructuring derivative matrices to appropriate forms for
            # Liouville operator
            derivMatrixChi = np.asarray(derivMatrixChi.todense())[1:-1, 1:-1]
            derivMatrixRz = np.asarray(derivMatrixRz.todense())[1:-1, 1:-1]

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - vw**2)
        PWall = gammaWall * (pz - vw * E)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        EPlasma = gammaPlasma * (E - v * pz)
        PPlasma = gammaPlasma * (pz - v * E)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (vw - v)

        # (exact) derivatives of compactified coordinates
        dchidxi, drzdpz, drpdpp = self.grid.getCompactificationDerivatives()
        dchidxi = dchidxi[:, np.newaxis, np.newaxis]
        drzdpz = drzdpz[np.newaxis, :, np.newaxis]

        # (exact) temperature derivative of equilibrium distribution
        warnings.filterwarnings("ignore", message="overflow encountered in exp")
        dfEq = BoltzmannSolver.__dfeq(EPlasma / T, statistics)
        warnings.filterwarnings(
            "default", message="overflow encountered in exp"
        )

        ##### source term #####
        source = (dfEq / T) * dchidxi * (
            PWall * PPlasma * gammaPlasma**2 * dvdChi
            + PWall * EPlasma * dTdChi / T
            + 1 / 2 * dMsqdChi * uwBaruPl
        )

        ##### liouville operator #####
        liouville = (
            dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * PWall[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * derivMatrixChi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
            - dchidxi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * drzdpz[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * gammaWall / 2
                * dMsqdChi[:, :, :, np.newaxis, np.newaxis, np.newaxis]
                * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * derivMatrixRz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
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
                assert basisSize == self.grid.N, \
                    f"Collision basis size {basisSize} != {self.grid.N}"
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
        Check that basis is recognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "BoltzmannSolver error: unkown basis %s" % basis

    def __checkDerivatives(derivatives):
        """
        Check that derivative option is recognised
        """
        derivativesOptions = ["Spectral", "Finite Difference"]
        assert derivatives in derivativesOptions, \
            f"BoltzmannSolver error: unkown derivatives option {derivatives}"

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
