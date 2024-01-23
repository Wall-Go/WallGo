import numpy as np
from scipy.special import eval_chebyt
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from .model import Particle
from .Polynomial2 import Polynomial
from .Grid import Grid

class CollisionArray:
    """
    Class used to load, transform, interpolate and hold the collision array 
    which is needed in Boltzmann.
    """
    def __init__(self, collisionFilename: str, N: int, basis: str, particle1: Particle, particle2: Particle):
        """
        Initialization of CollisionArray

        Parameters
        ----------
        collisionFilename : str
            Path of the file containing the collision array.
        N : int
            Desired order of the polynomial expansion. The resulting collision
            array will have a shape (N-1, N-1)
        basis : str
            Basis in which the Boltzmann equation is solved.
        particle1 : Particle
            Particle object describing the first out-of-equilibrium particle.
        particle2 : Particle
            Particle object describing the second out-of-equilibrium particle.

        Returns
        -------
        None.

        """
        self.N = N
        CollisionArray.__checkBasis(basis)
        self.basis = basis
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
        collisionArray = np.transpose(
            np.flip(collisionArray, (2, 3)),
            (2, 3, 0, 1),
        )
        
        # If collisionArray is not in the right basis, change it.
        if basisType != self.basis:
            self.__changeBasis(collisionArray, self.basis)
        
    def __changeBasis(self, collisionArray: np.ndarray, newBasis: str):
        """
        Transforms the basis of collisionArray to 'Cardinal'.

        Parameters
        ----------
        collisionArray : np.ndarray
            Array containing the collision term.
        newBasis : str
            Desired basis for collisionArray.

        Returns
        -------
        None.

        """
        
        CollisionArray.__checkBasis(newBasis)
        
        N = collisionArray.shape[0]+1
        grid = Grid(N,N,1,1)
        
        # Computing the transformation matrix
        n1 = np.arange(2,N+1)
        n2 = np.arange(1,N)
        M1 = np.cos(n1[:,None]*np.arccos(grid.rzValues[None,:])) - np.where(n1[:,None]%2==0,1,grid.rzValues[None,:])
        M2 = np.cos(n2[:,None]*np.arccos(grid.rpValues[None,:])) - 1
        if newBasis == 'Cardinal':
            M1 = np.linalg.inv(M1)
            M2 = np.linalg.inv(M2)
        collisionArray = np.sum(M1[None,None,:,None,:,None]*M2[None,None,None,:,None,:]*collisionArray[:,:,None,None,:,:],(-1,-2))
        
        
        
    def __checkBasis(basis: str):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "CollisionArray error: unkown basis %s" % basis