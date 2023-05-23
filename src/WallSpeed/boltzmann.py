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
            # changing convention so that bubble moves outwards
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
        source = self.source()
        liouville = self.liouville()
        collision = self.collision()

        # constructing the full rank 6 tensor operator
        operator = liouville + collision[np.newaxis, :, :, np.newaxis, :, :]

        # reshaping indices
        N_new = self.grid.M * self.grid.N**2
        source = np.reshape(source, N_new)
        operator = np.reshape(operator, (N_new, N_new), order="F")

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

    def __getDotProducts(self):
        """
        Returns dict containing various useful dot products
        """
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        v = self.background.vProfile
        vw = self.background.vw

        # dict to store results
        dotPs = {}

        # fluctuation mode
        statistics = self.mode.statistics
        msq = self.mode.msq(field)
        E = np.sqrt(msq + pz**2 + pp**2)

        # dot products with wall velocity
        dotPs["gammaWall"] = 1 / np.sqrt(1 - vw**2)
        dotPs["EWall"] = gammaWall * (pz - vw * E)
        dotPs["PWall"] = gammaWall * (E - vw * pz)

        # dot products with plasma profile velocity
        dotPs["gammaPlasma"] = 1 / np.sqrt(1 - v**2)
        dotPs["EPlasma"] = gammaPlasma * (pz - v * E)
        dotPs["PPlasma"] = gammaPlasma * (E - v * pz)

        # dot product of velocities
        dotPs["uwBaruPl"] = gammaWall * gammaPlasma * (vw - v)

        return dotPs

    def __getDerivatives(self):
        """
        Returns dict containing various useful derivatives
        """
        # dict to store results
        derivs = {}

        # polynomial tool
        poly = Polynomial(self.grid)

        # coordinates
        xi, pz, pp = self.grid.getCoordinates() # non-compact

        # background profiles
        T = self.background.temperatureProfile
        field = self.background.fieldProfile
        v = self.background.vProfile

        # fluctuation mode
        msq = self.mode.msq(field)

        # spatial derivatives of profiles
        deriv = poly.derivativesChebyshev()
        derivs["dTdxi"] = np.dot(deriv, T)
        derivs["dvdxi"] = np.dot(deriv, v)
        derivs["dmsqdxi"] = np.dot(deriv, msq)

        # derivatives of compactified coordinates
        dchi, drz, drp = grid.getCompactificationDerivatives()
        derivs["dchidxi"] = dchi
        derivs["drzdpz"] = drz
        derivs["drpdpp"] = drp

        # equilibrium distribution, and its derivative
        fEq = 1 / (np.exp(EPlasma / T) - statistics * 1)
        derivs["dfEq"] = -np.exp(EPlasma / T) * fEq**2

        return derivs

    def source():
        """
        Local equilibrium source term for non-equilibrium deviations, a
        rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.
        """
        # coordinates
        xi, pz, pp = self.grid.getCoordinates() # non-compact
        chi, rz, rp = self.grid.getCompactCoordinates() # compact
        xi = xi[:, np.newaxis, np.newaxis]
        pz = pz[np.newaxis, :, np.newaxis]
        pp = pp[np.newaxis, np.newaxis, :]

        # polynomial tool
        poly = Polynomial(self.grid)

        # background profiles
        T = self.background.temperatureProfile
        field = self.background.fieldProfile
        v = self.background.vProfile
        vw = self.background.vw



        # putting it all together
        return (dfEq / T) * (
            PWall * PPlasma * gammaPlasma**2 * dvdxi
            + PWall * EPlasma * dTdxi / T
            + 1 / 2 * dmsqdxi * uwBaruPl
        )


    def liouville():
        """
        Lioville operator, a rank 6 array, with shape
        :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.
        """
        # putting it together
        return (
            dchidxi * PWall * deriv * unit * unit
            - dchidxi * drzdpz * gammaWall / 2 * dmsqdxi * unit * deriv * unit
        ) * Tai * Tbj * Tck


    def collision():
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22]_.
        """
        collisionFile = BoltzmannSolver.__collisionFilename(z, pp, pp, "top")
        datasetName = "chebyshev"
        try:
            with h5py.File(collisionFile, "r") as file:
                collision = np.array(file[datasetName])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise
        return collision

    def __collisionFilename(z, pz, pp, particle):
        """
        A filename convention for collision integrals.
        """
        dir = "."
        suffix = "hdf5"
        filename = "%s/collision_%s_Nz_%i_Npz_%i_Npp_%i.%s" % (
            dir, particle, len(z), len(pz), len(pp), suffix
        )
        return filename
