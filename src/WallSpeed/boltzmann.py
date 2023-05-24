import numpy as np
import h5py # read/write hdf5 structured binary data file format
from .Grid import Grid

class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    def __init__(self, grid, background, mode):
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
        if background.vw < 0:
            # fixing convention so that bubble moves outwards
            self.background.vw *= -1
            self.background.vProfile *= -1
        self.mode = mode
        self.dotPs = self.__getDotProducts()
        self.derivs = self.__getDerivatives()

    def solveBoltzmannEquations():
        """
        Solves Boltzmann equation for :math:`\delta f`, equation (32) of [LC22].

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
        source, operator = self.buildLinearEquations()

        # solving the linear system: operator.delta_f = source
        delta_f = np.linalg.solve(operator, source)

        # returning result
        return delta_f

    def getDeltas():
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

    def buildLinearEquations():
        """
        Constructs matrix and source for equation of form :math:`M x = s`.

        Note, we make extensive use of numpy's broadcasting rules.
        """
        # polynomial tool
        poly = Polynomial(self.grid)

        # coordinates
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # identity matrices
        unitXi = np.identity(self.grid.M)
        unitRz = np.identity(self.grid.N)
        unitRp = np.identity(self.grid.N)

        # background profiles
        T = self.background.temperatureProfile[:, np.newaxis, np.newaxis]
        field = self.background.fieldProfile[:, np.newaxis, np.newaxis]
        v = self.background.vProfile[:, np.newaxis, np.newaxis]
        vw = self.background.vw

        # fluctuation mode
        statistics = self.mode.statistics
        msq = self.mode.msq(field)
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
        derivXi = poly.derivativesChebyshev(...)
        derivPz = poly.derivativesChebyshev(...)
        dTdxi = np.dot(derivM, T) #np.einsum("ij,jbc", deriv, T, optimize=True)
        dvdxi = np.dot(derivM, v) #np.einsum("ij,jbc", deriv, v, optimize=True)
        dmsqdxi = np.dot(derivM, msq) #np.einsum("ij,jbc", deriv, msq, optimize=True)

        # derivatives of compactified coordinates
        dchidxi, drzdpz, drpdpp = grid.getCompactificationDerivatives()
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
                * derivXi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * unitPz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * unitPp[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
            - dchidxi * drzdpz * gammaWall / 2 * dmsqdxi
                * unitXi[:, np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                * derivPz[np.newaxis, :, np.newaxis, np.newaxis, :, np.newaxis]
                * unitPp[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :]
        ) * Tai * Tbj * Tck
        should be a better way of doing the above, without introducing so many unit operators, I guess these are the Tai in the cardinal basis

        ##### collision operator #####
        collisionFile = self.__collisionFilename("top")
        collision = BoltzmannSolver.readCollision(collisionFile)

        ##### total operator #####
        operator = liouville + collision[np.newaxis, :, :, np.newaxis, :, :]

        # reshaping indices
        N_new = self.grid.M * self.grid.N**2
        source = np.reshape(source, N_new)
        operator = np.reshape(operator, (N_new, N_new), order="F")

        # returning results
        return operator, srouce

    def readCollision(collisionFile):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22]_.
        """
        datasetName = "chebyshev"
        try:
            with h5py.File(collisionFile, "r") as file:
                collision = np.array(file[datasetName])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise
        return collision

    def __collisionFilename(self, particle):
        """
        A filename convention for collision integrals.
        """
        dir = "."
        suffix = "hdf5"
        filename = "%s/collision_%s_M_%i_N_%i.%s" % (
            dir, particle, self.grid.M, self.grid.N, suffix
        )
        return filename
