import numpy as np
import h5py # read/write hdf5 structured binary data file format
from .Grid import Grid

class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    def __init__(self, grid, background, mode, collisionFile):
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
        collisionFile : string
            File name where collision integrals are stored.

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """
        self.grid = grid
        self.background = background
        self.mode = mode
        try:
            with h5py.File(collisionFile, "r") as file:
                self.collision = np.array(file["random"])
        except FileNotFoundError:
            print("BoltzmannSolver error: %s not found" % collisionFile)
            raise


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
        source = self.source(z, pz, pp)
        liouville = self.liouville(z, pz, pp)
        collision = self.collision(z, pz, pp)

        # constructing the full rank 6 tensor operator
        operator = liouville + collision[np.newaxis, :, :, np.newaxis, :, :]

        # reshaping indices
        N_new = len(z) * len(pz) * len(pp)
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

    def source(vw):
        """
        Local equilibrium source term for non-equilibrium deviations, a
        rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.
        """
        # coordinates
        chi, rz, rp = self.grid.getCompactCoordinates()

        # background profiles
        T = self.background.temperatureProfile
        field = self.background.fieldProfile

        # fluctuation mode
        statistics = self.mode.statistics
        msq = self.mode.msq(field)

        # evaluating dot products with the plasma 4-velocity
        gamma = 1 / np.sqrt(1 - vw**2)
        E_wall = np.sqrt(msq + rz**2 + rp**2)
        E_plasma = gamma * (E_wall - vw * rz)
        P_plasma = gamma * (- vw * E_wall + rz)

        # equilibrium distribution, and its derivative
        f_eq = 1 / (np.exp(E_plasma / T) - statistics * 1)
        df_eq = -np.exp(E_plasma / T) * f_eq**2

        # pz d/dz term

        # mass derivative term

        # putting it together
        pass

    def liouville(z, pz, pp):
        """
        Lioville operator, a rank 6 array, with shape
        :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.
        """
        pass


    def collision(z, pz, pp):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`.

        See equation (30) of [LC22].
        """
        pass

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
