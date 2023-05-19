import numpy as np
import h5py # read/write hdf5 structured binary data file format

class BoltzmannSolver:
    """
    Class for solving Boltzmann equations for small deviations from equilibrium.
    """

    def __init__():
        """ Initialsation of BoltzmannSolver

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
        """ Solves Boltzmann equation for :math:`\delta f`.
        
        Parameters
        ----------

        Returns
        -------
        delta_f : array_like
            The deviation from equilibrium, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.
        """
        pass

    def getDeltas():
        """ Computes Deltas necessary for solving the Higgs equation of motion.

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
        """ Local equilibrium source for non-equilibrium deviations

        All coordinates are in the wall frame.

        Parameters
        ----------
        z : array_like
            Array of z coordinate positions.
        pz : array_like
            Array of momenta in the z direction.
        pp : array_like
            Array of momenta parallel to the wall.

        Returns
        -------
        source : array_like
            The source term, a rank 3 array, with shape
            :py:data:`(len(z), len(pz), len(pp))`.
        """
        pass

    def __liouville(z, pz, pp):
        """ Lioville operator

        All coordinates are in the wall frame.

        Parameters
        ----------
        z : array_like
            Array of z coordinate positions.
        pz : array_like
            Array of momenta in the z direction.
        pp : array_like
            Array of momenta parallel to the wall.

        Returns
        -------
        liouville : array_like
            The liouville operator, a rank 6 array, with shape
            :py:data:`(len(z), len(pz), len(pp), len(z), len(pz), len(pp))`.
        """
        pass


    def __collision(z, pz, pp):
        """ Collision integrals

        All coordinates are in the wall frame.

        Parameters
        ----------
        z : array_like
            Array of z coordinate positions.
        pz : array_like
            Array of momenta in the z direction.
        pp : array_like
            Array of momenta parallel to the wall.

        Returns
        -------
        collision : array_like
            The collision integrals, a rank 4 array, with shape
            :py:data:`(len(pz), len(pp), len(pz), len(pp))`.
        """
        pass
