import numpy as np
import h5py # read/write hdf5 structured binary data file format
from .Grid import Grid
from .Polynomial import Polynomial

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
        self.particle = particle
        BoltzmannSolver.__checkBasis(basisM)
        BoltzmannSolver.__checkBasis(basisN)
        self.basisM = basisM
        self.basisN = basisN

    def solveBoltzmannEquations(self):
        r"""
        Solves Boltzmann equation for :math:`\delta f`, equation (32) of [LC22].

        Equations are of the form

        .. math::
            \mathcal{O}_{ijk,abc} \delta f_{abc} = \mathcal{S}_{ijk},

        where letters from the middle of the alphabet denote points on the
        coordinate lattice :math:`\{\xi_i,p_{z,j},p_{\Vert,k}\}`, and letters from the
        beginning of the alphabet denote elements of the basis of spectral
        functions :math:`\{\bar{T}_a, \bar{T}_b, \tilde{T}_c\}`.

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

    def getDeltas(self):
        """
        Computes Deltas necessary for solving the Higgs equation of motion.

        These are defined in equation (20) of [LC22]_.

        Parameters
        ----------

        Returns
        -------
        Deltas : array_like
            Defined in equation (20) of [LC22]_. A list of 4 arrays, each of
            which is of size :py:data:`len(z)`.
        """
        pass

    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        # polynomial tool
        poly = Polynomial(self.grid)
        derivChi = poly.derivatives(self.grid.chiValues, basis=self.basisM)
        derivPz = poly.derivatives(self.grid.pzValues, basis=self.basisN)
        derivXi = poly.derivatives(self.grid.xiValues, basis=self.basisM)
        derivRz = poly.derivatives(self.grid.rzValues, basis=self.basisN)

        # coordinates
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # intertwiner matrices
        #TChiMat = np.identity(self.grid.M - 1)
        #TRzMat = np.identity(self.grid.N - 1)
        #TRpMat = np.identity(self.grid.N - 1)
        TChiMat = poly.intertwiner("Coordinate", self.basisM)
        TRzMat = poly.intertwiner("Coordinate", self.basisN)
        TRpMat = TRzMat
        DTChiMat = np.dot(derivChi, TChiMat)
        DTRzMat = np.dot(derivRz, TRzMat)

        # background profiles
        T = self.background.temperatureProfile[:, np.newaxis, np.newaxis]
        field = self.background.fieldProfile[:, np.newaxis, np.newaxis]
        v = self.background.velocityProfile[:, np.newaxis, np.newaxis]
        vw = self.background.vw

        # fluctuation mode
        statistics = self.particle.statistics
        msq = self.particle.msqVacuum(field)
        E = np.sqrt(msq + pz**2 + pp**2)

        # dot products with wall velocity
        gammaWall = 1 / np.sqrt(1 - vw**2)
        EWall = gammaWall * (pz - vw * E)
        PWall = gammaWall * (E - vw * pz)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        EPlasma = gammaPlasma * (pz - v * E)
        PPlasma = gammaPlasma * (E - v * pz)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (vw - v)

        # spatial derivatives of profiles
        #dTdxi = np.einsum("ij,jbc", derivXi, T, optimize=True)
        #dvdxi = np.einsum("ij,jbc", derivXi, v, optimize=True)
        #dmsqdxi = np.einsum("ij,jbc", derivXi, msq, optimize=True)
        dTdxi = np.dot(derivXi, T[:, 0, 0])[:, np.newaxis, np.newaxis]
        dvdxi = np.dot(derivXi, v[:, 0, 0])[:, np.newaxis, np.newaxis]
        dmsqdxi = np.dot(derivXi, msq[:, 0, 0])[:, np.newaxis, np.newaxis]

        # derivatives of compactified coordinates
        dchidxi, drzdpz, drpdpp = self.grid.getCompactificationDerivatives()
        dchidxi = dchidxi[:, np.newaxis, np.newaxis]
        drzdpz = drzdpz[np.newaxis, :, np.newaxis]

        # equilibrium distribution, and its derivative
        fEq = 1 / (np.exp(EPlasma / T) - statistics * 1)
        dfEq = -np.exp(EPlasma / T) * fEq**2

        ##### source term #####
        source = (dfEq / T) * (
            PWall * PPlasma * gammaPlasma**2 * dvdxi
            + PWall * EPlasma * dTdxi / T
            + 1 / 2 * dmsqdxi * uwBaruPl
        )

        ##### liouville operator #####
        liouville = (
            dchidxi * PWall
                * DTChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * TRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
            - dchidxi * drzdpz * gammaWall / 2 * dmsqdxi
                * TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * DTRzMat[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * TRpMat[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        )

        ##### collision operator #####
        collisionFile = self.__collisionFilename()
        collisionArray, collision = BoltzmannSolver.readCollision(collisionFile, "top")
        if self.basisN != collisionBasis:
            TInvRzMat = poly.intertwiner(collisionBasis, self.basisN)
            TInvRpMat = TInvRzMat
            collisionArray = np.einsum(
                "ac,bd,ijcd",
                TInvRzMat,
                TInvRpMat,
                collisionArray,
                optimize=True,
            )

        ##### total operator #####
        operator = (
            liouville
            + TChiMat[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * collision[np.newaxis, :, :, np.newaxis, :, :]
        )

        # reshaping indices
        N_new = (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        source = np.reshape(source, N_new)
        operator = np.reshape(operator, (N_new, N_new), order="F")

        # returning results
        return operator, source

    def readCollision(collisionFile):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22]_.
        """
        try:
            with h5py.File(collisionFile, "r") as file:
                collisionArray = np.array(file["Array"])
                collisionBasis = np.array(file["Basis"])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise
        return collisionArray, collisionBasis

    def __collisionFilename(self):
        """
        A filename convention for collision integrals.
        """
        return "data/collision_N_20.hdf5" ##### a temporary hack
        dir = "."
        suffix = "hdf5"
        filename = "%s/collision_%s_N_%i.%s" % (
            dir, self.basisN, self.grid.N, suffix
        )
        return filename

    def __checkBasis(basis):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "BoltzmannSolver error: unkown basis %s" % basis
