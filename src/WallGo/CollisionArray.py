import numpy as np
from scipy.special import eval_chebyt
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from .model import Particle

class CollisionArray:
    """
    Class used to load, transform, interpolate and hold the collision array 
    which is needed in Boltzmann.
    """
    def __init__(self, collisionFilename: str, N: int, particle1: Particle, particle2: Particle):
        """
        Initialization of CollisionArray

        Parameters
        ----------
        collisionFilename : str
            Path of the file containing the collision array.
        N : int
            Desired order of the polynomial expansion. The resulting collision
            array will have a shape (N-1, N-1)
        particle1 : Particle
            Particle object describing the first out-of-equilibrium particle.
        particle2 : Particle
            Particle object describing the second out-of-equilibrium particle.

        Returns
        -------
        None.

        """
        self.N = N
        self.particle1 = particle1
        self.particle2= particle2
        self.loadFile(collisionFilename)
        
    def loadFile(self, filename: str):
        """
        Load the collision array and transform it to be used by Boltzmann.

        Parameters
        ----------
        filename : str
            Path of the file containing the collision array.

        Returns
        -------
        None.

        """
        try:
            with h5py.File(filename, "r") as file:
                metadata = file["metadata"]
                basisSize = metadata.attrs["Basis Size"]
                basisType = codecs.decode(
                    metadata.attrs["Basis Type"], 'unicode_escape',
                )
                CollisionArray.__checkBasis(basisType)

                # LN: currently the dataset names are of form
                # "particle1, particle2". Here it's just "top, top" for now.
                datasetName = self.particle1.name + ", " + self.particle2.name
                collisionArray = np.array(file[datasetName][:])
        except FileNotFoundError:
            print("CollisionArray error: %s not found" % filename)
            raise
            
        # Need to make sure basisSize has the same meaning as self.N!!!
        assert basisSize <= self.N, f"CollisionArray error: Basis size of {filename} too small to generate a collision array of size N = {self.N}."

        # converting between conventions
        self.collisionArray = np.transpose(
            np.flip(collisionArray, (2, 3)),
            (2, 3, 0, 1),
        )
        
    def __checkBasis(basis):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "BoltzmannSolver error: unkown basis %s" % basis