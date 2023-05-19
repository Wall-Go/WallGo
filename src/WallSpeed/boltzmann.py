import numpy as np
import h5py # read/write hdf5 structured binary data file format
from .coordinates import ...

class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.

    """

    def __init__():
        """
        Initialsation of BoltzmannSolver

        Parameters
        ----------

        Returns
        -------
        cls : BoltzmannSolver
            An object of the BoltzmannSolver class.
        """
        cls = BoltzmannSolver()

        return cls

    def solveBoltzmannEquations():
        """
        Solves Boltzmann equation for :math:`\delta f`.

        Parameters
        ----------

        Returns
        -------
        delta_f : array_like
            The deviation from equilibrium, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.
        """
        # contructing the various terms in the Boltzmann equation
        source = self.__source(z, pz, pp)
        liouville = self.__liouville(z, pz, pp)
        collision = self.__collision(z, pz, pp)

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

        These are defined in equation (20) or [LC22]_.

        Parameters
        ----------

        Returns
        -------
        Deltas : array_like
            Defined in equation (20) or [LC22]_. A list of 4 arrays, each of
            which is of size :py:data:`len(z)`.

        References
        ----------
        .. [LC22] B. Laurent and J. M. Cline, First principles determination
            of bubble wall velocity, Phys. Rev. D 106 (2022) no.2, 023501
            doi:10.1103/PhysRevD.106.023501
        """
        pass

    def __source(z, pz, pp):
        """
        Local equilibrium source term for non-equilibrium deviations, a
        rank 3 array, with shape :py:data:`(len(z), len(pz), len(pp))`.
        """
        pass

    def __liouville(z, pz, pp):
        """
        Lioville operator, a rank 6 array, with shape
        :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`
        """
        pass


    def __collision(z, pz, pp):
        """
        Collision integrals, a rank 4 array, with shape
        :py:data:`(len(pz), len(pp), len(pz), len(pp))`
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
