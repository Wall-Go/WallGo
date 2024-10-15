"""
Class for loading and storing the collision files used in boltzmann.py.
"""

import codecs  # for decoding unicode string from hdf5 file
import copy  ## for deepcopy
from pathlib import Path
import tempfile
import typing
import numpy as np
import h5py  # read/write hdf5 structured binary data file format

from .particle import Particle
from .grid import Grid
from .polynomial import Polynomial
from .exceptions import CollisionLoadError


class CollisionArray:
    r"""
    Class used to load, transform, interpolate and hold the collision array
    which is needed in :py:data:`WallGo.BoltzmannSolver`. Internally the collision array
    is represented by a :py:data:`WallGo.Polynomial` object. Specifically, this
    describes a rank-4 tensor :math:`C[P_k(p_{z}^{(i)}) P_l(p_{\parallel}^{(j)})]` where
    the :math:`p` are momenta on the grid. Index ordering is hardcoded as: :math:`ijkl`.
    Currently we have one CollisionArray object for each pair of off-equilibrium
    particles.
    """

    AXIS_TYPES: typing.Final[tuple[str, ...]] = (
        "Array",
        "pz",
        "pp",
        "Array",
        "pz",
        "pp",
    )
    r"""Static axis types in correct order."""

    AXIS_LABELS: typing.Final[tuple[str, ...]] = (
        "particles",
        "pz",
        "pp",
        "particles",
        "polynomial1",
        "polynomial2",
    )
    r"""Static axis labels in correct order, :math:`ijklmn`."""

    def __init__(self, grid: Grid, basisType: str, particles: list[Particle]):
        """
        Initializes a CollisionArray for a given grid and basis. Collision data will be
        set to zero.

        Parameters
        ----------
        grid : Grid
            Grid object that the collision array lives on (non-owned).
        basisType: str
            Basis to use for the polynomials. Note that unlike in the
            :py:data:`WallGo.Polynomial` class, our basis is just a string. We always
            use `"Cardinal"` basis on momentum axes and `basisType` on polynomial axes.
            We do NOT support different basis types for the two polynomials.
        particles: list[Particle]
            List of the :py:data:`WallGo.Particle` objects this collision array
            describes.

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

        ## Setup the actual collision data. We will use "Cardinal" basis on momentum
        ## axes and default to "Chebyshev" for polynomial axes.
        bases = ("Array", "Cardinal", "Cardinal", "Array", basisType, basisType)

        ## Default to zero but correct size
        data = np.empty(
            (len(particles), self.size, self.size, len(particles), self.size, self.size)
        )
        self.polynomialData = Polynomial(
            data, grid, bases, CollisionArray.AXIS_TYPES, endpoints=False
        )

    def __getitem__(self, key: int | slice) -> float | np.ndarray:
        """
        Retrieve the value at the specified key.

        Parameters
        ----------
        key : int or slice
            The index of the value to retrieve.

        Returns
        -------
        float or np.ndarray
            The value at the specified key.

        Raises
        ------
        IndexError
            If the key is out of range.

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

        Returns
        -------
        str
            The basis type of the CollisionArray.
        """
        return self.basisType

    @staticmethod
    def newFromPolynomial(
        inputPolynomial: Polynomial, particles: list[Particle]
    ) -> "CollisionArray":
        """
        Creates a new CollisionArray object from polynomial data (which contains a grid
        reference). This only makes sense if the polynomial is already in correct shape.

        Parameters
        ----------
        inputPolynomial : Polynomial
            The input polynomial data.
        particles : list[Particle]
            The list of particles.

        Returns
        -------
        "CollisionArray"
            The newly created CollisionArray object.

        Raises
        ------
        AssertionError
            If the input polynomial does not meet the required conditions.

        """
        bases = inputPolynomial.basis

        assert inputPolynomial.rank == 6
        assert bases[1] == "Cardinal" and bases[2] == "Cardinal"
        assert bases[0] == "Array" and bases[3] == "Array"
        assert bases[4] == bases[5]  ## polynomial axes need to be in same basis

        basisType = bases[4]

        newCollision = CollisionArray(inputPolynomial.grid, basisType, particles)
        newCollision.polynomialData = inputPolynomial
        return newCollision

    ## This will fail with assert or exception if something goes wrong.
    # If we don't want to abort, consider denoting failure by return value instead
    @staticmethod
    def newFromDirectory(
        directoryPath: Path,
        grid: Grid,
        basisType: str,
        particles: list[Particle],
        bInterpolate: bool = True,
    ) -> "CollisionArray":
        """
        Create a new CollisionArray object from a directory containing collision files.

        Parameters
        ----------
        collision : Collision
            Collision class that holds path of the directory containing the collision
            files. The collision files must have names with the form
            "collisions_particle1_particle2.hdf5".
        grid : Grid
            The grid object representing the computational grid.
        basisType : str
            The basis type for the CollisionArray object.
        particles : list[Particle]
            The list of particles involved in the collisions.
        bInterpolate : bool, optional
            Interpolate the data to match the grid size. Extrapolation is not possible.
            Default is True.

        Returns
        -------
        CollisionArray
            The new CollisionArray object created from the collision files.

        Raises
        ------
        CollisionLoadError
            If any of the collision files are not found,
            or if there is a grid size mismatch and bInterpolate is set to False.

        """
        collisionFileArray: np.ndarray
        basisSizeFile: int
        basisTypeFile: str

        for i, particle1 in enumerate(particles):
            for j, particle2 in enumerate(particles):

                # file names are hardcoded
                filename = (
                    directoryPath / f"collisions_{particle1.name}_{particle2.name}.hdf5"
                )
                try:
                    with h5py.File(str(filename), "r") as file:

                        metadata = file["metadata"]
                        size = metadata.attrs["Basis Size"]

                        if grid.N > size:
                            raise CollisionLoadError(
                                f"""Target collision grid size ({grid.N}) must be smaller
                                or equal to the grid size recorded in collision files ({size}).
                                """
                            )

                        btype = codecs.decode(
                            metadata.attrs["Basis Type"],
                            "unicode_escape",
                        )
                        CollisionArray._checkBasis(btype)

                        # Dataset names are hardcoded, eg. "top, top"
                        datasetName = particle1.name + ", " + particle2.name
                        collisionDataset = np.array(file[datasetName][:])

                        if not "collisionFileArray" in locals():
                            collisionFileArray = np.zeros(
                                (
                                    len(particles),
                                    size - 1,
                                    size - 1,
                                    len(particles),
                                    size - 1,
                                    size - 1,
                                )
                            )
                            basisSizeFile = size
                            basisTypeFile = btype
                        else:
                            assert (
                                size == basisSizeFile
                            ), """CollisionArray error: All the collision files must
                            have the same basis size."""
                            assert (
                                btype == basisTypeFile
                            ), """CollisionArray error: All the collision files must
                            have the same basis type."""

                        collisionFileArray[i, :, :, j, :, :] = collisionDataset
                        
                except FileNotFoundError:
                    raise CollisionLoadError(
                        f"CollisionArray error: {filename} not found."
                    )

        collisionFileArray = collisionFileArray.reshape(
            (
                len(particles),
                basisSizeFile - 1,
                basisSizeFile - 1,
                len(particles),
                basisSizeFile - 1,
                basisSizeFile - 1,
            )
        )

        """We want to compute Polynomial object from the loaded data and put it on the
        input grid. This is straightforward if the grid size matches that of the data,
        if not we either abort or attempt interpolation to smaller N. In latter case we
        need a dummy grid of read size, create a dummy CollisionArray living on the
        dummy grid, and finally downscale that. 
        """

        if basisSizeFile == grid.N:
            polynomialData = Polynomial(
                collisionFileArray,
                grid,
                (
                    "Array",
                    "Cardinal",
                    "Cardinal",
                    "Array",
                    basisTypeFile,
                    basisTypeFile,
                ),
                CollisionArray.AXIS_TYPES,
                endpoints=False,
            )
            newCollision = CollisionArray.newFromPolynomial(polynomialData, particles)

        else:
            ## Grid sizes don't match, attempt interpolation
            if not bInterpolate:
                raise CollisionLoadError(
                    f"""Grid size mismatch when loading collision directory: {directoryPath}.
                    Consider using bInterpolate=True in CollisionArray.loadFromFile().""",
                )

            dummyGrid = Grid(
                grid.M,
                basisSizeFile,
                grid.positionFalloff,
                grid.momentumFalloffT,
                grid.spacing,
            )
            dummyPolynomial = Polynomial(
                collisionFileArray,
                dummyGrid,
                (
                    "Array",
                    "Cardinal",
                    "Cardinal",
                    "Array",
                    basisTypeFile,
                    basisTypeFile,
                ),
                CollisionArray.AXIS_TYPES,
                endpoints=False,
            )

            dummyCollision = CollisionArray.newFromPolynomial(
                dummyPolynomial, particles
            )
            newCollision = CollisionArray.interpolateCollisionArray(
                dummyCollision, grid
            )

        ## Change to the requested basis
        return newCollision.changeBasis(basisType)

    def changeBasis(self, newBasisType: str) -> "CollisionArray":
        """
        Changes the basis in our polynomial indices.

        Parameters
        ----------
        newBasisType : str
            The new basis type to be used.

        Returns
        -------
        CollisionArray
            The modified CollisionArray object.

        Notes
        -----
            - Momentum indices always use the Cardinal basis.
            - This method modifies the object in place.

        """

        if self.basisType == newBasisType:
            return self

        CollisionArray._checkBasis(newBasisType)

        # NEEDS to take inverse transpose because of magic
        self.polynomialData.changeBasis(
            ("Array", "Cardinal", "Cardinal", "Array", newBasisType, newBasisType),
            inverseTranspose=True,
        )
        self.basisType = newBasisType
        return self

    @staticmethod
    def interpolateCollisionArray(
        srcCollision: "CollisionArray", targetGrid: Grid
    ) -> "CollisionArray":
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
        This function interpolates a collision array to match the size of a target grid.
        It takes the source collision array and the target grid as input, and returns
        the interpolated collision array.

        The interpolation is performed by evaluating the original collisions on the
        interpolated grid points. The resulting data is used to create a new polynomial,
        which is then used to create a new CollisionArray object.

        The source collision array must be in the Chebyshev basis for interpolation.
        The target grid should have a size smaller than the source grid size.

        Example
        -------
        >>> srcCollision = CollisionArray(...)
        >>> targetGrid = Grid(...)
        >>> interpolatedCollision = interpolateCollisionArray(srcCollision, targetGrid)

        """

        assert (
            targetGrid.N <= srcCollision.getBasisSize()
        ), f"""CollisionArray interpolation error: target grid size ({targetGrid.N})
        must be smaller than or equal to the source grid size
        ({srcCollision.getBasisSize()+1})."""

        ## Take deepcopy to avoid modifying the input
        source = copy.deepcopy(srcCollision)

        # Source needs to be in the Chebyshev basis for interpolation
        source.changeBasis("Chebyshev")

        # Generate a grid of points to give as input to Polynomial.evaluate.
        gridPoints = np.array(
            np.meshgrid(targetGrid.rzValues, targetGrid.rpValues, indexing="ij")
        ).reshape((2, (targetGrid.N - 1) ** 2))

        # Evaluate the original collisions on the interpolated grid, create a new
        # polynomial from the result and finally a new CollisionArray from the
        # polynomial data
        newShape = 2 * (
            len(source.particles),
            targetGrid.N - 1,
            targetGrid.N - 1,
        )
        interpolatedData = np.array(source.polynomialData.evaluate(gridPoints, (1, 2)))[
            ..., : targetGrid.N - 1, : targetGrid.N - 1
        ].reshape(newShape)

        interpolatedPolynomial = Polynomial(
            interpolatedData,
            targetGrid,
            ("Array", "Cardinal", "Cardinal", "Array", "Chebyshev", "Chebyshev"),
            ("z", "pz", "pp", "z", "pz", "pp"),
            endpoints=False,
        )

        newCollision = CollisionArray.newFromPolynomial(
            interpolatedPolynomial, source.particles
        )

        ## Change back to the original basis
        newCollision.changeBasis(srcCollision.getBasisType())
        return newCollision

    @staticmethod
    def _checkBasis(basis: str) -> None:
        """
        Check that basis is recognized.

        Parameters
        ----------
        basis : str
            The basis to be checked.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the basis is unknown.

        """
        bases = ["Cardinal", "Chebyshev"]
        assert basis in bases, f"collisionarray error: unknown basis {basis}"
