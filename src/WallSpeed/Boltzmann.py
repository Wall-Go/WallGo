import numpy as np
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from .Grid import Grid
from .Polynomial import Polynomial
from .model import Particle
from .helpers import boostVelocity


class BoltzmannBackground:
    def __init__(
        self,
        vw,
        velocityProfile,
        fieldProfile,
        temperatureProfile,
        polynomialBasis="Cardinal",
    ):
        self.vw = vw
        self.velocityProfile = np.asarray(velocityProfile)
        self.fieldProfile = np.asarray(fieldProfile)
        self.temperatureProfile = np.asarray(temperatureProfile)
        self.polynomialBasis = polynomialBasis

    def boostToPlasmaFrame(self):
        """
        Boosts background to the plasma frame
        """
        vPlasma = self.velocityProfile
        v0 = self.velocityProfile[0]
        vw = self.vw
        self.velocityProfile = boostVelocity(vPlasma, v0)
        self.vw = boostVelocity(vw, v0)

    def boostToWallFrame(self):
        """
        Boosts background to the wall frame
        """
        vPlasma = self.velocityProfile
        vPlasma0 = self.velocityProfile[0]
        vw = self.vw
        self.velocityProfile = boostVelocity(vPlasma, vw)
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
        self.background = background
        self.background.boostToPlasmaFrame()
        self.particle = particle
        BoltzmannSolver.__checkBasis(basisM)
        BoltzmannSolver.__checkBasis(basisN)
        self.basisM = basisM
        self.basisN = basisN
        self.poly = Polynomial(self.grid)

    def getDeltas(self, deltaF=None):
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (15) of [LC22]_.

        Parameters
        ----------

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

        # coordinates
        chi, rz, rp = self.grid.getCompactCoordinates() # compact
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # background
        vFixed = self.background.velocityProfile[0]
        T0 = self.background.temperatureProfile[0]

        # fluctuation mode
        msq = self.particle.msqVacuum(self.background.fieldProfile)
        msq = msq[:, np.newaxis, np.newaxis]
        E = np.sqrt(msq + pz**2 + pp**2)

        # dot products with fixed plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - vFixed**2)
        EPlasma = gammaPlasma * (E - vFixed * pz)
        PPlasma = gammaPlasma * (pz - vFixed * E)

        # weights for Gauss-Lobatto quadrature (endpoints plus extrema)
        sin_arg_Pz = np.pi / self.grid.N * np.arange(1, self.grid.N)
        weightsPz = np.pi / self.grid.N * np.sin(np.flip(sin_arg_Pz))**2
        weightsPz /= np.sqrt(1 - rz**2)
        # note, we drop the point at rp=-1, to avoid an apparent divergence.
        # should think further about this another day.
        sin_arg_Pp = np.pi / (self.grid.N - 1) * np.arange(1, self.grid.N - 1)
        weightsPp = np.pi / (self.grid.N - 1) * np.sin(np.flip(sin_arg_Pp))**2
        weightsPp /= np.sqrt(1 - rp[1:]**2)
        weights = weightsPz[:, np.newaxis] * weightsPp[np.newaxis, :]
        # measure, including Jacobian from coordinate compactification
        measurePz = (2 * T0) / (1 - rz**2)
        measurePp = T0**2 / (1 - rp[1:]) * np.log(2 / (1 - rp[1:]))
        measurePzPp = measurePz[:, np.newaxis] * measurePp[np.newaxis, :]
        measurePzPp /= (2 * np.pi)**2

        # evaluating integrals with Gaussian quadrature
        measureWeight = measurePzPp * weights
        arg00 = deltaF[:, :, 1:] / E[:, :, 1:]
        Deltas["00"] = np.einsum(
            "jk, ijk -> i",
            measureWeight,
            deltaF[:, :, 1:] / E[:, :, 1:],
            optimize=True,
        )
        Deltas["11"] = np.einsum(
            "jk, ijk -> i",
            measureWeight,
            arg00 * EPlasma[:, :, 1:] * PPlasma[:, :, 1:],
            optimize=True,
        )
        Deltas["20"] = np.einsum(
            "jk, ijk -> i",
            measureWeight,
            arg00 * EPlasma[:, :, 1:]**2,
            optimize=True,
        )
        Deltas["02"] = np.einsum(
            "jk, ijk -> i",
            measureWeight,
            arg00 * PPlasma[:, :, 1:]**2,
            optimize=True,
        )

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
        operator, source = self.buildLinearEquations()

        # solving the linear system: operator.deltaF = source
        deltaF = np.linalg.solve(operator, source)

        # returning result
        deltaFShape = (self.grid.M - 1, self.grid.N - 1, self.grid.N - 1)
        return np.reshape(deltaF, deltaFShape, order="F")


    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        # derivative matrices
        derivChi = self.poly.deriv(self.basisM, "z")
        derivRz = self.poly.deriv(self.basisN, "pz")

        # coordinates
        xi, pz, pp = self.grid.getCoordinates()  # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # intertwiner matrices
        TChiMat = self.poly.matrix(self.basisM, "z")
        TRzMat = self.poly.matrix(self.basisN, "pz")
        TRpMat = self.poly.matrix(self.basisN, "pp")

        # background profiles
        T = self.background.temperatureProfile[:, np.newaxis, np.newaxis]
        field = self.background.fieldProfile[..., np.newaxis, np.newaxis]
        v = self.background.velocityProfile[:, np.newaxis, np.newaxis]
        vw = self.background.vw

        # fluctuation mode
        statistics = -1 if self.particle.statistics == "Fermion" else 1
        # TODO: indices order not consistent across different functions.
        msq = self.particle.msqVacuum(field)
        E = np.sqrt(msq + pz**2 + pp**2)

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - vw**2)
        EWall = gammaWall * (E - vw * pz)
        PWall = gammaWall * (pz - vw * E)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        EPlasma = gammaPlasma * (E - v * pz)
        PPlasma = gammaPlasma * (pz - v * E)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (vw - v)

        # spatial derivatives of profiles
        dTdChi = np.einsum("ij,jbc->ibc", derivChi, T, optimize=True)
        dvdChi = np.einsum("ij,jbc->ibc", derivChi, v, optimize=True)
        dmsqdChi = np.einsum("ij,jbc->ibc", derivChi, msq, optimize=True)

        # derivatives of compactified coordinates
        dchidxi, drzdpz, drpdpp = self.grid.getCompactificationDerivatives()
        dchidxi = dchidxi[:, np.newaxis, np.newaxis]
        drzdpz = drzdpz[np.newaxis, :, np.newaxis]

        # equilibrium distribution, and its derivative
        fEq = 1 / (np.exp(EPlasma / T) - statistics * 1)
        dfEq = -np.exp(EPlasma / T) * fEq**2

        ##### source term #####
        source = (dfEq / T) * dchidxi * (
            PWall * PPlasma * gammaPlasma**2 * dvdChi
            + PWall * EPlasma * dTdChi / T
            + 1 / 2 * dmsqdChi * uwBaruPl
        )

        ##### liouville operator #####
        liouville = (
            dchidxi * PWall
                * derivChi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
            - dchidxi * drzdpz * gammaWall / 2 * dmsqdChi
                * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * derivRz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        )

        ##### collision operator #####
        collisionFile = self.__collisionFilename()
        collisionArray, collisionBasis, collisionN = BoltzmannSolver.readCollision(
            collisionFile,
            self.particle,
        )
        assert collisionN == self.grid.N, \
            f"Collision basis size error {collisionN=}"
        if self.basisN != collisionBasis:
            TInvRzMat = self.poly.intertwiner(collisionBasis, self.basisN)
            TInvRpMat = TInvRzMat
            collisionArray = np.einsum(
                "ac,bd,ijcd->ijab",
                TInvRzMat,
                TInvRpMat,
                collisionArray,
                optimize=True,
            )

        ##### total operator #####
        operator = (
            liouville
            + TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
            * collisionArray[np.newaxis, :, :, np.newaxis, :, :]
        )

        # reshaping indices
        N_new = (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        source = np.reshape(source, N_new)
        operator = np.reshape(operator, (N_new, N_new), order="F")

        # returning results
        return operator, source

    def readCollision(collisionFile, particle):
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
                collisionArray = np.array(file[particle.name][:])
                print("Multiplying by first particle collision prefactor.")
                collisionArray *= particle.collisionPrefactors[0]
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise
        return collisionArray, basisType, basisSize

    def __collisionFilename(self):
        """
        A filename convention for collision integrals.
        """
        dir = "../data"
        suffix = "hdf5"
        return f"{dir}/collisions_N{self.grid.N}.{suffix}"

    def __checkBasis(basis):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "BoltzmannSolver error: unkown basis %s" % basis
