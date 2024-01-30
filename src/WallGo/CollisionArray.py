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
        self.particle2 = particle2
        
        # Load the collision file
        self.loadFile(collisionFilename)
        
        # Interpolates the collision file
        self.interpolateArray(N)
        
        # Change the basis
        self.interpolatedPoly.changeBasis(("Cardinal", "Cardinal", basis, basis), inverseTranspose=True)
        
        # Extract the collision array
        self.collisionArray = self.interpolatedPoly.coefficients
        
    def __getitem__(self, key):
        return self.collisionArray[key]
        
    def loadFile(self, filename: str):
        """
        Load the collision array and store it in a Polynomial object.

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
                basisSizeFile = metadata.attrs["Basis Size"]
                basisTypeFile = codecs.decode(
                    metadata.attrs["Basis Type"], 'unicode_escape',
                )
                CollisionArray.__checkBasis(basisTypeFile)

                # LN: currently the dataset names are of form
                # "particle1, particle2". Here it's just "top, top" for now.
                datasetName = self.particle1.name + ", " + self.particle2.name
                collisionFileArray = np.array(file[datasetName][:])
        except FileNotFoundError:
            print("CollisionArray error: %s not found" % filename)
            raise
            
        # converting between conventions
        collisionFileArray = np.transpose(
            np.flip(collisionFileArray, (2, 3)),
            (2, 3, 0, 1),
        )
        
        # Create a Grid object to be used by Polynomial
        gridFile = Grid(basisSizeFile, basisSizeFile, 1, 1)
            
        # Create the Polynomial object 
        self.collisionFilePoly = Polynomial(
            collisionFileArray,
            gridFile,
            ("Cardinal", "Cardinal", basisTypeFile, basisTypeFile),
            ("pz", "pp", "pz", "pp"),
            False,
        ) 
        
    def interpolateArray(self, newBasisSize: int):
        """
        Interpolate the collision file to get a collision array of a smaller 
        size.

        Parameters
        ----------
        newBasisSize : int
            Basis size of the desired collision array.

        Returns
        -------
        None.

        """
        assert newBasisSize <= self.collisionFilePoly.grid.N, "CollisionArray error: newBasisSize must be smaller than the collision file's basisType."
        
        # Make sure the collision file is in the Chebyshev basis which is 
        # required for the interpolation.
        self.collisionFilePoly.changeBasis(("Cardinal", "Cardinal", "Chebyshev", "Chebyshev"), inverseTranspose=True)  
        
        # Create a Grid object for the interpolated collision array
        newGrid = Grid(newBasisSize, newBasisSize, 1, 1)
        
        # Generate a grid of points to give as input to Polynomial.evaluate.
        gridPoints = np.array(np.meshgrid(newGrid.rzValues,newGrid.rpValues,indexing='ij')).reshape((2,(newGrid.N-1)**2))
        
        # Evaluate the collision file on that grid,truncate it and reshape it.
        interpolatedArray = self.collisionFilePoly.evaluate(gridPoints, (0,1))[:,:newGrid.N-1,:newGrid.N-1].reshape(4*(newGrid.N-1,))
        
        # Create a new Polynomial object containing the interpolated array
        self.interpolatedPoly = Polynomial(
            interpolatedArray, newGrid,
            ("Cardinal", "Cardinal", "Chebyshev", "Chebyshev"),
            ("pz","pp","pz","pp"), False)
          
    def __checkBasis(basis: str):
        """
        Check that basis is reckognised
        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "CollisionArray error: unkown basis %s" % basis