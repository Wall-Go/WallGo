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
        self.poly = Polynomial(self.grid)
        print("NOTE: should boost frames for input velocities from Joonas's to mine")

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

    def getDeltas(self, deltaF):
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
        # dict to store results
        Delta = {"00": 0, "02": 0, "20": 0, "11": 0}

        # coordinates
        chi, rz, rp = self.grid.getCompactCoordinates() # compact

        # fluctuation mode
        msq = self.particle.msqVacuum(field)
        E = np.sqrt(msq + pz**2 + pp**2)

        # dot products with fixed plasma profile velocity
        vFixed = v[0]
        gammaPlasma = 1 / np.sqrt(1 - vFixed**2)
        EPlasma = gammaPlasma * (E - vFixed * pz)
        PPlasma = gammaPlasma * (pz - vFixed * E)

        # weights for Gauss-Legendre quadrature
        weightsPz = np.pi / self.grid.N * np.sin(np.pi / self.grid.N * np.arange(1, self.grid.N))
        weightsPz /= np.sqrt(1 - rz**2)
        weightsPp = ... ###### this isn't true
        weightsPp *= ...
        weights = weightsPz[:, np.newaxis] *  weightsPp[np.newaxis, :]
        # measure, including Jacobian from coordinate compactification
        measurePz = (2 * T0) / (1 - rz**2)
        measurePp = T0**2 / (1 - rp) * np.log(2 / (1 - rp)) ####### looks like this is going to hit a singularity on the grid
        measurePzPp = 1 / (2 * np.pi)**2 / E * measurePz * measurePp
        # evaluating with Gauss-Chebyshev quadrature
        Delta["00"] = np.einsum("jk, ijk", measurePzPp * weights, deltaF, optimize=True)
        print("NOTE: should boost frames for output velocities from mine to Joonas's")



    def buildLinearEquations(self):
        """
        Constructs matrix and source for Boltzmann equation.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        derivChi = self.poly.deriv(self.basisM, "z")
        derivRz = self.poly.deriv(self.basisN, "pz")

        # coordinates
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # intertwiner matrices
        TChiMat = self.poly.matrix(self.basisM, "z")
        TRzMat = self.poly.matrix(self.basisN, "pz")
        TRpMat = self.poly.matrix(self.basisN, "pp")

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
        EWall = gammaWall * (E - vw * pz)
        PWall = gammaWall * (pz - vw * E)

        # dot products with plasma profile velocity
        gammaPlasma = 1 / np.sqrt(1 - v**2)
        EPlasma = gammaPlasma * (E - v * pz)
        PPlasma = gammaPlasma * (pz - v * E)

        # dot product of velocities
        uwBaruPl = gammaWall * gammaPlasma * (vw - v)

        # spatial derivatives of profiles
        #dTdxi = np.einsum("ij,jbc", derivXi, T, optimize=True)
        #dvdxi = np.einsum("ij,jbc", derivXi, v, optimize=True)
        #dmsqdxi = np.einsum("ij,jbc", derivXi, msq, optimize=True)
        dTdChi = np.dot(derivChi, T[:, 0, 0])[:, np.newaxis, np.newaxis]
        dvdChi = np.dot(derivChi, v[:, 0, 0])[:, np.newaxis, np.newaxis]
        dmsqdChi = np.dot(derivChi, msq[:, 0, 0])[:, np.newaxis, np.newaxis]

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
        collisionArray, collisionBasis = BoltzmannSolver.readCollision(
            collisionFile,
            "top",
        )
        if self.basisN != collisionBasis and False:
            print("I am confused:", [self.basisN, collisionBasis])
            TInvRzMat = self.poly.intertwiner(collisionBasis, self.basisN)
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
                * collisionArray[np.newaxis, :, :, np.newaxis, :, :]
        )

        # reshaping indices
        N_new = (self.grid.M - 1) * (self.grid.N - 1) * (self.grid.N - 1)
        source = np.reshape(source, N_new)
        operator = np.reshape(operator, (N_new, N_new), order="F")

        # returning results
        return operator, source

    def readCollision(collisionFile, particle="top"):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22]_.
        """
        try:
            with h5py.File(collisionFile, "r") as file:
                dataset = file[particle]
                collisionArray = np.array(dataset)
                collisionBasis = str(dataset.attrs["Basis"])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise
        return collisionArray, collisionBasis

    def __collisionFilename(self):
        """
        A filename convention for collision integrals.
        """
        return "src/Collision/build/collisions_Chebyshev_20.hdf5" ##### hack
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
