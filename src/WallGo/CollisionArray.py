import numpy as np
from scipy.special import eval_chebyt
import h5py # read/write hdf5 structured binary data file format
import codecs # for decoding unicode string from hdf5 file
from typing import Tuple
import copy ## for deepcopy
from pathlib import Path

from .Particle import Particle
from .Polynomial import Polynomial
from .Grid import Grid

class CollisionArray:
    """
    Class used to load, transform, interpolate and hold the collision array 
    which is needed in Boltzmann. Internally the collision array is represented by a Polynomial object.
    Specifically, this describes a rank-4 tensor C[P_k(pZ_i) P_l(pPar_j)] where the p are momenta on grid.
    Index ordering is hardcoded as: ijkl.
    Right now we have one CollisionArray for each pair of off-eq particles 
    """

    """Hardcode axis types and their meaning in correct ordering. 
    Our order is ijklmn, as given in AXIS_LABELS.
    """
    AXIS_TYPES = ("Array", "pz", "pp", "Array", "pz", "pp")
    AXIS_LABELS = ("particles", "pz", "pp", "particles", "polynomial1", "polynomial2")

    def __init__(self, grid: Grid, basisType: str, particles: list):
        """
        Initializes a CollisionArray for a given grid and basis. Collision data will be set to zero.

        Parameters
        ----------
        grid : Grid
            Grid object that the collision array lives on (non-owned).
        basisType: str
            Basis to use for the polynomials. Note that unlike in the Polynomial class, our basis is just a string.
            We always use "Cardinal" basis on momentum axes and basisType on polynomial axes. 
            We do NOT support different basis types for the two polynomials.
        particles: list
            List of the Particle objects this collision array describes.

        Returns
        -------
        None.
        """

        self.grid = grid

        ## Our actual data size is N-1 in each direction
        self.size = grid.N - 1

        self.grid = grid
        self.basisType = basisType
        self.particles = particles
        
        ## Setup the actual collision data. We will use "Cardinal" basis on momentum axes and default to "Chebyshev" for polynomial axes.
        bases = ("Array", "Cardinal", "Cardinal", "Array", basisType, basisType)

        ## Default to zero but correct size
        data = np.empty( (len(particles), self.size, self.size, len(particles), self.size, self.size))
        self.polynomialData = Polynomial(data, grid, bases, CollisionArray.AXIS_TYPES, endpoints=False)



    def __getitem__(self, key):
        """
        Retrieve the value at the specified key.

        Parameters:
            key (int): The index of the value to retrieve.

        Returns:
            Any: The value at the specified key.

        Raises:
            IndexError: If the key is out of range.
        """
        return self.polynomialData.coefficients[key]
    
    def getBasisSize(self) -> int:
            """
            Returns the size of the basis.

            Returns:
                int: The size of the basis.
            """
            return self.size
    
    def getBasisType(self) -> str:
            """
            Returns the basis type of the CollisionArray.

            Returns:
                str: The basis type of the CollisionArray.
            """
            return self.basisType
    
    @staticmethod
    def newFromPolynomial(inputPolynomial: Polynomial, particles: list) -> 'CollisionArray':
        """Creates a new CollisionArray object from polynomial data (which contains a grid reference).
        This only makes sense if the polynomial is already in correct shape.

        Args:
            inputPolynomial (Polynomial): The input polynomial data.
            particles (list): The list of particles.

        Returns:
            CollisionArray: The newly created CollisionArray object.

        Raises:
            AssertionError: If the input polynomial does not meet the required conditions.

        """
        bases = inputPolynomial.basis

        assert inputPolynomial.rank == 6
        assert bases[1] == "Cardinal" and bases[2] == "Cardinal"
        assert bases[0] == "Array" and bases[3] == "Array"
        assert bases[4] == bases[5] ## polynomial axes need to be in same basis   

        basisType = bases[4]

        newCollision = CollisionArray(inputPolynomial.grid, basisType, particles)
        newCollision.polynomialData = inputPolynomial
        return newCollision



    ## This will fail with assert or exception if something goes wrong. If we don't want to abort, consider denoting failure by return value instead
    @staticmethod
    def newFromDirectory(directoryname: str, grid: Grid, basisType: str, particles: list, bInterpolate: bool = True) -> 'CollisionArray':
        """
        Create a new CollisionArray object from a directory containing collision files.

        Parameters
        ----------
        directoryname : str
            Path of the directory containing the collision files. The collision files must have names with the form "collisions_particle1_particle2.hdf5".

        grid : Grid
            The grid object representing the computational grid.

        basisType : str
            The basis type for the CollisionArray object.

        particles : list
            The list of particles involved in the collisions.

        bInterpolate : bool, optional
            Interpolate the data to match the grid size. Extrapolation is not possible. Default is True.

        Returns
        -------
        CollisionArray
            The new CollisionArray object created from the collision files.

        Raises
        ------
        FileNotFoundError
            If any of the collision files are not found.

        RuntimeError
            If there is a grid size mismatch and bInterpolate is set to False.

        """
        collisionFileArray = None
        basisSizeFile = grid.N 
        basisTypeFile = None
        
        for i, particle1 in enumerate(particles):
            for j, particle2 in enumerate(particles):

                # file names are hardcoded
                filename = Path(directoryname) / f"collisions_{particle1.name}_{particle2.name}.hdf5"
                try:
                    with h5py.File(str(filename), "r") as file:

                        metadata = file["metadata"]
                        size = metadata.attrs["Basis Size"]
                        btype = codecs.decode(
                            metadata.attrs["Basis Type"], 'unicode_escape',
                        )
                        CollisionArray.__checkBasis(btype)

                        # Dataset names are hardcoded, eg. "top, top"
                        datasetName = particle1.name + ", " + particle2.name
                        collision = np.array(file[datasetName][:])

                        ## TODO error handling, what happens if the dataset is not found?

                        if collisionFileArray is None:
                            collisionFileArray = np.zeros((len(particles),size-1,size-1,len(particles),size-1,size-1))
                            basisSizeFile = size
                            # assert size == basisSizeFile, "CollisionArray error: Collision file has different basis size than initial value."
                            basisTypeFile = btype
                        else:
                            ## TODO throw WallGo error?
                            assert size == basisSizeFile, "CollisionArray error: All the collision files must have the same basis size."
                            assert btype == basisTypeFile, "CollisionArray error: All the collision files must have the same basis type."

                        # HACK. converting between conventions because collision file was computed with different index ordering
                        collisionFileArray[i,:,:,j,:,:] = np.transpose(
                            np.flip(collision, (2, 3)),
                            (2, 3, 0, 1),
                        )
                except FileNotFoundError:
                    print("CollisionArray error: %s not found" % filename)
                    raise

        collisionFileArray = collisionFileArray.reshape((len(particles), basisSizeFile-1,
                                                         basisSizeFile-1,len(particles),basisSizeFile-1,basisSizeFile-1))

        """We want to compute Polynomial object from the loaded data and put it on the input grid.
        This is straightforward if the grid size matches that of the data, if not we either abort
        or attempt interpolation to smaller N. In latter case we need a dummy grid of read size,
        create a dummy CollisionArray living on the dummy grid, and finally downscale that. 
        """

        if (basisSizeFile == grid.N):
            polynomialData = Polynomial(collisionFileArray, grid, ("Array", "Cardinal", "Cardinal", "Array", basisTypeFile, basisTypeFile),
                                        CollisionArray.AXIS_TYPES, endpoints=False)
            newCollision = CollisionArray.newFromPolynomial(polynomialData, particles)

        else:   
            ## Grid sizes don't match, attempt interpolation
            if (not bInterpolate):
                raise RuntimeError("Grid size mismatch when loading collision directory: ", directoryname, \
                                   "Consider using bInterpolate=True in CollisionArray.loadFromFile()." )

            dummyGrid = Grid(grid.M, basisSizeFile, grid.L_xi, grid.momentumFalloffT, grid.spacing)
            dummyPolynomial = Polynomial(collisionFileArray, dummyGrid, ("Array", "Cardinal", "Cardinal", "Array", basisTypeFile, basisTypeFile),
                                         CollisionArray.AXIS_TYPES, endpoints=False)

            dummyCollision = CollisionArray.newFromPolynomial(dummyPolynomial, particles)
            newCollision = CollisionArray.interpolateCollisionArray(dummyCollision, grid)

        ## Change to the requested basis
        return newCollision.changeBasis(basisType)


    def changeBasis(self, newBasisType: str) -> 'CollisionArray':
        """Changes the basis in our polynomial indices.

        Args:
            newBasisType (str): The new basis type to be used.

        Returns:
            CollisionArray: The modified CollisionArray object.

        Notes:
            - Momentum indices always use the Cardinal basis.
            - This method modifies the object in place.
        """

        if self.basisType == newBasisType:
            return self

        CollisionArray.__checkBasis(newBasisType)

        # NEEDS to take inverse transpose because of magic
        self.polynomialData.changeBasis(("Array", "Cardinal", "Cardinal", "Array", newBasisType, newBasisType), inverseTranspose=True)
        self.basisType = newBasisType
        return self

        
    @staticmethod
    def interpolateCollisionArray(srcCollision: 'CollisionArray', targetGrid: Grid) -> 'CollisionArray':
        """
        Interpolate collision array to match a target grid size.

        Parameters
        ----------
        srcCollision : CollisionArray
            The source collision array to be interpolated.
        targetGrid : Grid
            The target grid to match the size of the interpolated collision array.

        Returns
        -------
        CollisionArray
            The interpolated collision array.

        Raises
        ------
        AssertionError
            If the target grid size is larger than or equal to the source grid size.

        Notes
        -----
        This function interpolates a collision array to match the size of a target grid. It takes the source collision array and the target grid as input, and returns the interpolated collision array.

        The interpolation is performed by evaluating the original collisions on the interpolated grid points. The resulting data is used to create a new polynomial, which is then used to create a new CollisionArray object.

        The source collision array must be in the Chebyshev basis for interpolation. The target grid should have a size smaller than the source grid size.

        Example
        -------
        >>> srcCollision = CollisionArray(...)
        >>> targetGrid = Grid(...)
        >>> interpolatedCollision = interpolateCollisionArray(srcCollision, targetGrid)
        """

        assert targetGrid.N <= srcCollision.getBasisSize(), "CollisionArray interpolation error: target grid size must be smaller than the source grid size."
        
        ## Take deepcopy to avoid modifying the input
        source = copy.deepcopy(srcCollision)

        # Source needs to be in the Chebyshev basis for interpolation
        source.changeBasis("Chebyshev")  
        
        # Generate a grid of points to give as input to Polynomial.evaluate.
        gridPoints = np.array(np.meshgrid(targetGrid.rzValues, targetGrid.rpValues,indexing='ij')).reshape((2,(targetGrid.N-1)**2))
        
        # Evaluate the original collisions on the interpolated grid, create a new polynomial from the result and finally a new CollisionArray from the polynomial data
        newShape = 2*(len(source.particles),targetGrid.N-1,targetGrid.N-1,)
        interpolatedData = source.polynomialData.evaluate(gridPoints, (1,2))[...,:targetGrid.N-1,:targetGrid.N-1].reshape(newShape)

        interpolatedPolynomial = Polynomial(interpolatedData, targetGrid, 
                                            ("Array", "Cardinal", "Cardinal", "Array", "Chebyshev", "Chebyshev"), 
                                            ("z","pz","pp","z","pz","pp"), endpoints=False)
        
        newCollision = CollisionArray.newFromPolynomial(interpolatedPolynomial, source.particles)

        ## Change back to the original basis
        newCollision.changeBasis(srcCollision.getBasisType())
        return newCollision
    

    def estimateLxi(self, v: float, T1: float, T2: float, msq1: float, msq2: float) -> Tuple[float, float]:
        """
        Estimate the decay length of the solution by computing the eigenvalues
        of the collision array.

        Parameters
        ----------
        v : float
            Wall velocity in the plasma frame.
        T1 : float
            Temperature in the symmetric phase.
        T2 : float
            Temperature in the broken phase.
        msq1 : float
            Squared mass in the symmetric phase.
        msq2 : float
            Squared mass in the broken phase.

        Returns
        -------
        Tuple(float,float)
            Approximate decay length in the symmetric and broken phases, 
            respectively.

        """
        # Compute the grid of momenta
        _, pz, pp = self.grid.getCoordinates() 
        pz = pz[:, np.newaxis]
        pp = pp[np.newaxis, :]
        E1 = np.sqrt(msq1 + pz**2 + pp**2)
        E2 = np.sqrt(msq2 + pz**2 + pp**2)
        
        gamma = 1 / np.sqrt(1 - v**2)
        PWall1 = gamma * (pz - v * E1)
        PWall2 = gamma * (pz - v * E2)

        # Compute the eigenvalues
        size = self.grid.N-1
        eigvals1 = np.linalg.eigvals(T1**2*((self.polynomialData.coefficients / PWall1[:,:,None,None]).reshape((size,size,size**2))).reshape((size**2,size**2)))
        eigvals2 = np.linalg.eigvals(T2**2*((self.polynomialData.coefficients / PWall2[:,:,None,None]).reshape((size,size,size**2))).reshape((size**2,size**2)))
        
        # Compute the decay length
        return np.max(-1/np.real(eigvals1)),np.max(1/np.real(eigvals2))
        
    @staticmethod
    def __checkBasis(basis: str):
        """
        Check that basis is recognized.

        Parameters:
        basis (str): The basis to be checked.

        Raises:
        AssertionError: If the basis is unknown.

        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, "CollisionArray error: unknown basis %s" % basis
