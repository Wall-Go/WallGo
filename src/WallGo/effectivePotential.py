"""
Class for describing the field- and temperature-dependent effective potential.
"""

from typing import Tuple
from abc import ABC, abstractmethod  # Abstract Base Class
import numpy as np
import scipy.optimize
import scipy.interpolate

from .helpers import derivative, gradient, hessian
from .Fields import Fields, FieldPoint


class EffectivePotential(ABC):
    """
    Base class for the effective potential Veff. WallGo uses this to identify phases and
    their temperature dependence, and to compute free energies
    (pressures) in the two phases.

    Hydrodynamical routines in WallGo need the full pressure in the plasma, which in
    principle is p = -Veff(phi) if phi is a local minimum. However for phase transitions
    it is common to neglect field-independent parts of Veff, for example one may choose
    normalization so that Veff(0) = 0. Meanwhile for hydrodynamics we require knowledge
    of all temperature-dependent parts. With this in mind, WallGo requires that the
    effective potential is defined with full T-dependence included.

    The final technicality you should be aware of is the variable fieldLowerBound,
    which is used as a cutoff for avoiding spurious behavior at phi = 0.
    You may need to adjust this to suit your needs, especially if using
    a complicated 2-loop potential.
    """

    # TODO we could optimize some routines that only depend on free-energy differences
    # ( dV/dField, findTc ) by separating the field-dependent parts of Veff(phi, T)
    # and the T-dependent constant terms. This was done in intermediate commits but,
    # scrapped because it was too error prone (Veff appears in too many places).
    # But let's keep this possibility in mind. If attempting this,
    # keep full Veff as the default and use the field-only part internally when needed.

    # How many background fields. This is explicitly required so that we can have
    # better control over array shapes
    fieldCount: int

    # Lower bound for field values, used in normalize(). Using a small but nonzero value
    # to avoid spurious divergences from eg. logarithms
    fieldLowerBound: float = 1e-8

    # Typical relative accuracy at which the effective potential can be computed.
    # Is set close to the machine precision here which is appropriate
    # when the potential can be computed in terms of simple functions.
    effectivePotentialError: float = 1e-15

    # Typical temperature scale over which the effective potential changes by O(1).
    # A reasonable value would be of order Tc-Tn.
    temperatureScale: float

    # Field scale over which the potential changes by O(1).
    # A good value would be similar to the field VEV.
    fieldScale: np.ndarray

    # In practice we'll get the model params from a GenericModel subclass
    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        self.modelParameters = modelParameters
        self.fieldCount = fieldCount

        # HACK! This intitializes fieldScale and temperatureScale to 1s.
        # Should be overriden by self.setScales, but used in some tests.
        self.fieldScale = np.ones(fieldCount)
        self.temperatureScale = 1.0
        self.combinedScales = np.append(self.fieldScale, self.temperatureScale)

    @abstractmethod
    def evaluate(
        self,
        fields: Fields | FieldPoint,
        temperature: np.ndarray,
        checkForImaginary: bool = False,
    ) -> np.ndarray:
        """
        Implement the actual computation of Veff(phi) here. The return value should be
        (the UV-finite part of) Veff at the input field configuration and temperature.

        Normalization of the potential DOES matter: You have to ensure that the full
        T-dependence is included. Pay special attention to field-independent "constant"
        terms such as (minus the) pressure from light fermions.

        Parameters
        ----------
        fields: Fields | FieldPoint
            input field configuration.
        temperature: np.ndarray
            temperature at which the potential gets evaluated
        checkForImaginary: bool
            setting to handle imaginary parts in the effective potential.

        """
        raise NotImplementedError(
            "You are required to give an expression for the effective potential."
        )

    # Non-abstract stuff from here on

    def setPotentialError(self, potentialError: float) -> None:
        """
        Sets self.effectivePotentialError to potentialError.

        Parameters
        ----------
        potentialError: float
            new value of effectivePotentialError.
        """
        self.effectivePotentialError = potentialError

    def setScales(
        self, temperatureScale: float, fieldScale: float | np.ndarray
    ) -> None:
        """
        Sets self.temperatureScale to temperatureScale and
        self.fieldScale to fieldScale.

        Parameters
        ----------
        temperatureScale : float
            new temperature scale
        fieldScale : float or np.ndarray
            new field scale

        Returns
        -------
        """
        self.temperatureScale = temperatureScale

        if isinstance(fieldScale, float):
            self.fieldScale = fieldScale * np.ones(self.fieldCount)
        else:
            self.fieldScale = np.asanyarray(fieldScale)
            assert (
                self.fieldScale.size == self.fieldCount
            ), "EffectivePotential error: fieldScale must have size of self.fieldCount."
        self.combinedScales = np.append(self.fieldScale, self.temperatureScale)

    def findLocalMinimum(
        self, initialGuess: Fields, temperature: np.ndarray, tol: float | None = None
    ) -> Tuple[Fields, np.ndarray]:
        """
        Finds a local minimum starting from a given initial configuration
        of background fields. Feel free to override this if your model requires more
        delicate minimization.
        The returned values will be arrays of the same length as temperature.

        Parameters
        ----------
        initialguess: Fields
            Initial guess for the position of the minimum.
        temperature: np.ndarray
            Temperature(s) at which the minimum should be found.
        tol: float
            Tolerance for the minimizer.


        Returns
        -------
        minimum, functionValue : tuple.
            minimum: list[float] is the location x of the minimum in field space.
            functionValue: float is Veff(x) evaluated at the minimum.
        """

        # We need to manually vectorize this in case we get
        # many field/temperature points
        T = np.atleast_1d(temperature)

        numPoints = max(T.shape[0], initialGuess.NumPoints())

        # Reshape for broadcasting
        guesses = initialGuess.Resize(numPoints, initialGuess.NumFields())
        T = np.resize(T, (numPoints))

        resValue = np.empty_like(T)
        resLocation = np.empty_like(guesses)

        for i in range(0, numPoints):

            """
            Numerically minimize the potential wrt. fields.
            We can pass a fields array to scipy routines normally, but scipy
            seems to forcibly convert back to standard ndarray causing issues in the
            Veff evaluate function if it uses extended functionality from the
            Fields class. So we need a wrapper that casts back to Fields type. 
            It also needs to fix the temperature, and we only minimize the real part
            """

            def evaluateWrapper(fieldArray: np.ndarray) -> float | np.ndarray:
                """
                Wrapper that casts the potential back to Fields type.
                """
                fields = Fields.CastFromNumpy(fieldArray)
                return self.evaluate(fields, T[i]).real

            guess = guesses.GetFieldPoint(i)

            res = scipy.optimize.minimize(evaluateWrapper, guess, tol=tol)

            resLocation[i] = res.x
            resValue[i] = res.fun

            # Check for presenece of imaginary parts at minimum
            self.evaluate(Fields((res.x)), T[i], checkForImaginary=True)

        # Need to cast the field location
        return Fields.CastFromNumpy(resLocation), resValue

    def _wrapperPotential(self, fieldsAndTemperature: np.ndarray) -> float | np.ndarray:
        """
        Calls self.evaluate from a single array fieldsAndTemperature
        that contains both the fields and temperature.
        """
        fields = Fields(fieldsAndTemperature[..., :-1])
        temperature = fieldsAndTemperature[..., -1]
        return self.evaluate(fields, temperature)

    def _combineInputs(self, fields: Fields | FieldPoint, temperature: float | np.ndarray) -> np.ndarray:
        """
        Combines the fields and temperature in a single array.
        """
        shape = list(fields.shape)
        shape[-1] += 1
        combinedInput = np.empty(shape)
        combinedInput[..., :-1] = fields
        combinedInput[..., -1] = temperature
        return combinedInput

    def derivT(self, fields: Fields | FieldPoint, temperature: float | np.ndarray) -> np.ndarray:
        """Calculate derivative of (real part of) the effective potential with
        respect to temperature.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : array_like
            The temperature

        Returns
        ----------
        dVdT : array_like
            Temperature derivative of the potential, evaluated at each
            point of the input temperature array.
        """
        der = derivative(
            lambda T: self.evaluate(fields, T).real,
            temperature,
            n=1,
            order=4,
            epsilon=self.effectivePotentialError,
            scale=self.temperatureScale,
            bounds=(0, np.inf),
        )
        return der

    def derivField(self, fields: Fields | FieldPoint, temperature: np.ndarray):
        """Compute field-derivative of the effective potential with respect to
        all background fields, at given temperature.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : array_like
            The temperature

        Returns
        ----------
        dVdField : list[Fields]
            Field derivatives of the potential, one Fields object for each
            temperature. They are of Fields type since the shapes match nicely.
        """

        return gradient(
            self._wrapperPotential,
            self._combineInputs(fields, temperature),
            epsilon=self.effectivePotentialError,
            scale=self.combinedScales,
            axis=np.arange(self.fieldCount).tolist(),
        )

    def deriv2FieldT(
        self, fields: Fields | FieldPoint, temperature: np.ndarray
    ) -> list[Fields]:
        r"""Computes :math:`d^2V/(d\text{Field} dT)`.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : array_like
            The temperature

        Returns
        ----------
        d2fdFielddT : list[Fields]
            Field derivatives of the potential, one Fields object for each
            temperature. They are of Fields type since the shapes match nicely.
        """

        res = hessian(
            self._wrapperPotential,
            self._combineInputs(fields, temperature),
            epsilon=self.effectivePotentialError,
            scale=self.combinedScales,
            xAxis=np.arange(self.fieldCount).tolist(),
            yAxis=-1,
        )[..., 0]

        return res

    def deriv2Field2(self, fields: Fields | FieldPoint, temperature: np.ndarray) -> list[Fields]:
        r"""Computes the Hessian, :math:`d^2V/(d\text{Field}^2)`.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : np.ndarray
            Temperatures. Either scalar or a 1D array of same length
            as fields.NumPoints()

        Returns
        ----------
        d2VdField2 : list[Fields]
            Field Hessian of the potential. For each temperature, this is
            a matrix of the same size as Fields.
        """

        axis = np.arange(self.fieldCount).tolist()
        return hessian(
            self._wrapperPotential,
            self._combineInputs(fields, temperature),
            epsilon=self.effectivePotentialError,
            scale=self.combinedScales,
            xAxis=axis,
            yAxis=axis,
        )

    def allSecondDerivatives(
        self, fields: Fields | FieldPoint, temperature: np.ndarray | float
    ) -> Tuple[list[Fields], list[Fields], np.ndarray]:
        r"""Computes :math:`d^2V/(d\text{Field}^2)`, :math:`d^2V/(d\text{Field} dT)`
        and :math:`d^2V/(dT^2)` at the same time. This function is more efficient
        than calling the other functions one at a time.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : array_like
            The temperature

        Returns
        ----------
        d2VdField2 : list[Fields]
            Field Hessian of the potential. For each temperature, this is
            a matrix of the same size as Fields.
        d2fdFielddT : list[Fields]
            Field derivatives of the potential, one Fields object for each
            temperature. They are of Fields type since the shapes match nicely.
        d2VdT2 : array-like
            Temperature second derivative of the potential.
        """

        res = hessian(
            self._wrapperPotential,
            self._combineInputs(fields, temperature),
            epsilon=self.effectivePotentialError,
            scale=self.combinedScales,
        )

        hess = res[..., :-1, :-1]
        dgraddT = res[..., -1, :-1]
        d2VdT2 = res[..., -1, -1]

        return hess, dgraddT, d2VdT2
