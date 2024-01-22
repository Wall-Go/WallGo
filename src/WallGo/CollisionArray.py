import numpy as np
from scipy.special import eval_chebyt
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from .model import Particle

class CollisionArray:
    def __init__(self, collisionFilename: str, N: int, particle: Particle):
        self.N = N
        self.particle = particle
        self.loadFile(collisionFilename)
        
    def loadFile(self, filename: str):
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
                datasetName = self.particle.name + ", " + self.particle.name
                collisionArray = np.array(file[datasetName][:])
        except FileNotFoundError:
            print("CollisionArray error: %s not found" % filename)
            raise
            
        # Need to make sure basisSize has the same meaning as self.N!!!
        assert basisSize <= self.N, f"Basis size of {filename} too small to generate a collision array of size N = {self.N}."

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