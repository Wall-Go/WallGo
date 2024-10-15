"""
EffectivePotential class definition.
"""
from typing import Tuple, Type, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import inspect # check for ABC
import numpy as np
import numpy.typing as npt
import scipy.optimize
import scipy.interpolate

from .helpers import derivative, gradient, hessian

from .fields import Fields, FieldPoint

@dataclass
class VeffDerivativeSettings:
    """Parameters used to estimate the optimal value of dT used
    for the finite difference derivatives of the effective potential."""

    temperatureVariationScale: float
    """Temperature scale (in GeV) over which the potential changes by O(1).
    A good value would be of order Tc-Tn."""

    fieldValueVariationScale: float | list[float] | np.ndarray
    """Field scale (in GeV) over which the potential changes by O(1). A good value
    would be similar to the field VEV.
    Can either be a single float, in which case all the fields have the
    same scale, or an array of floats (one element for each classical field in the model)."""

class EffectivePotential(ABC):
    r"""Base class for the effective potential Veff. WallGo uses this to identify phases and their temperature dependence, 
    and computing free energies (pressures) in the two phases.
    
    Hydrodynamical routines in WallGo need the full pressure in the plasma, which in principle is :math:`p = -V_{\rm eff}(\phi)` if :math:`\phi` is a local minimum.
    However for phase transitions it is common to neglect field-independent parts of :math:`V_{\rm eff}`, for example one may choose normalization so that :math:`V_{\rm eff}(0) = 0`.
    Meanwhile for hydrodynamics we require knowledge of all temperature-dependent parts.
    With this in mind, you should ensure that your effective potential is defined with full T-dependence included.

    The effective potential defined here is assumed to be real. That is, the
    :py:meth:`evaluate()` function is assumed to return a real value.

    The user must call :py:meth:`configureDerivatives()` before evaluating the derivatives to set
    temperature and field scales of your potential, and optionally update the effectivePotentialError attribute.
    These quantities are used to estimate the optimal step size when computing derivatives with finite
    differences. It is done by requiring that the potential error and the error from
    finite difference calculation contribute similarly to the derivative error.
    See the :py:class:`VeffDerivativeSettings` dataclass.
    """
    
    fieldCount: int
    """
    Number of background fields in your potential.
    IMPORTANT: YOUR CONCRETE POTENTIAL MUST SET THIS TO A NONZERO POSITIVE INTEGER.
    """

    effectivePotentialError: float
    """
    Typical relative accuracy at which the effective potential can be computed.
    For simple polynomial potentials this is probably close to machine precision of Python floats (1e-15).
    For loop-corrected potentials a limited factor can be the eg. accuracy of numerical integration.
    IMPORTANT: YOUR CONCRETE POTENTIAL MUST SET THIS TO A NONZERO POSITIVE FLOAT.
    """

    derivativeSettings: VeffDerivativeSettings

    __combinedScales: np.ndarray
    # Used in derivatives, combines field/temperature scales into one array

    @abstractmethod
    def evaluate(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike) -> npt.ArrayLike:
        r"""Implement the actual computation of :math:`V_{\rm eff}(\phi)` here. The return value should be (the UV-finite part of)  :math:`V_{\rm eff}` 
        at the input field configuration and temperature. 
        
        Normalization of the potential DOES matter: You have to ensure that full T-dependence is included.
        Pay special attention to field-independent "constant" terms such as (minus the) pressure from light fermions. 
        """
        raise NotImplementedError("You are required to give an expression for the effective potential.")

    def __init_subclass__(cls: Type["EffectivePotential"], **kwargs: Any) -> None:
        """Called whenever a subclass is initialized.
        """
        super().__init_subclass__(**kwargs)

        # Check that fieldCount is valid, but skip the check for subclasses that are still abstract
        if inspect.isabstract(cls):
            return
        elif not hasattr(cls, 'fieldCount') or cls.fieldCount < 1:
            raise NotImplementedError("EffectivePotential subclasses must define a class variable 'fieldCount' with value > 0.")
    
        
    def configureDerivatives(self, settings: VeffDerivativeSettings) -> None:
        """
        Sets the temperature and field scales.
        These quantities are used together with the 'effectivePotentialError' attribute
        to estimate the optimal step size when computing
        derivatives with finite differences. It is done by requiring that the potential
        error and the error from finite difference calculation contribute similarly to
        the derivative error.
        """

        self.derivativeSettings = copy.copy(settings)

        # Interpret the field scale input and make it correct shape
        if isinstance(settings.fieldValueVariationScale, float):
            self.derivativeSettings.fieldValueVariationScale = settings.fieldValueVariationScale * np.ones(self.fieldCount)
        else:
            self.derivativeSettings.fieldValueVariationScale = np.asanyarray(settings.fieldValueVariationScale)
            assert self.derivativeSettings.fieldValueVariationScale.size == self.fieldCount, "EffectivePotential error: fieldValueVariationScale must have a size of self.fieldCount."
        self.__combinedScales = np.append(self.derivativeSettings.fieldValueVariationScale, self.derivativeSettings.temperatureVariationScale)

    def areDerivativesConfigured(self) -> bool:
        """True if derivative routines are ready to use."""
        return hasattr(self, 'derivativeSettings')
    
    def getInherentRelativeError(self) -> float:
        """"""
        return self.effectivePotentialError


    def findLocalMinimum(self, initialGuess: Fields, temperature: npt.ArrayLike, tol: float = None) -> Tuple[Fields, np.ndarray]:
        """
        Finds a local minimum starting from a given initial configuration of background fields.
        Feel free to override this if your model requires more delicate minimization.

        Returns
        -------
        minimum, functionValue : tuple. 
        minimum: list[float] is the location x of the minimum in field space.
        functionValue: float is Veff(x) evaluated at the minimum.
        If the input temperature is a numpy array, the returned values will be arrays of same length. 
        """

        # I think we'll need to manually vectorize this in case we got many field/temperature points
        T = np.atleast_1d(temperature)

        numPoints = max(T.shape[0], initialGuess.numPoints())

        ## Reshape for broadcasting
        guesses = initialGuess.resizeFields(numPoints, initialGuess.numFields())
        T = np.resize(T, (numPoints))

        resValue = np.empty_like(T)
        resLocation = np.empty_like(guesses)

        for i in range(0, numPoints):

            """Numerically minimize the potential wrt. fields. 
            We can pass a fields array to scipy routines normally, but scipy seems to forcibly convert back to standard ndarray
            causing issues in the Veff evaluate function if it uses extended functionality from the Fields class. 
            So we need a wrapper that casts back to Fields type. It also needs to fix the temperature.
            """

            def evaluateWrapper(fieldArray: np.ndarray):
                fields = Fields.castFromNumpy(fieldArray)
                return self.evaluate(fields, T[i])

            guess = guesses.getFieldPoint(i)

            res = scipy.optimize.minimize(evaluateWrapper, guess, tol=tol)

            resLocation[i] = res.x
            resValue[i] = res.fun

            # Check for presenece of imaginary parts at minimum
            self.evaluate(Fields((res.x)), T[i])

        ## Need to cast the field location
        return Fields.castFromNumpy(resLocation), resValue
    
    def __wrapperPotential(self, X):
        """
        Calls self.evaluate from a single array X that contains both the fields and temperature.
        """
        fields = Fields(X[...,:-1])
        temperature = X[...,-1]
        return self.evaluate(fields, temperature)
    
    def __combineInputs(self, fields, temperature):
        """
        Combines the fields and temperature in a single array.
        """
        shape = list(fields.shape)
        shape[-1] += 1
        combinedInput = np.empty(shape)
        combinedInput[...,:-1] = fields
        combinedInput[...,-1] = temperature
        return combinedInput

    def derivT(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike):
        """Calculate derivative of the effective potential with
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
        assert self.areDerivativesConfigured(), "EffectivePotential Error: configureDerivatives() must be "\
                                    "called before computing a derivative."
        der = derivative(
            lambda T: self.evaluate(fields, T),
            temperature,
            n=1,
            order=4,
            epsilon=self.effectivePotentialError,
            scale=self.derivativeSettings.temperatureVariationScale,
            bounds=(0,np.inf),
        )
        return der

    def derivField(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike):
        """ Compute field-derivative of the effective potential with respect to
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
        assert self.areDerivativesConfigured(), "EffectivePotential Error: configureDerivatives() must be "\
                                    "called before computing a derivative."
        return gradient(self.__wrapperPotential, self.__combineInputs(fields, temperature), epsilon=self.effectivePotentialError, 
                        scale=self.__combinedScales, axis=np.arange(self.fieldCount).tolist())

    def deriv2FieldT(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike):
        r""" Computes :math:`d^2V/(d\text{Field} dT)`.

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
        assert self.areDerivativesConfigured(), "EffectivePotential Error: configureDerivatives() must be "\
                                    "called before computing a derivative."
        res = hessian(self.__wrapperPotential, self.__combineInputs(fields, temperature), epsilon=self.effectivePotentialError, 
                      scale=self.__combinedScales, xAxis=np.arange(self.fieldCount).tolist(), yAxis=-1)[...,0]
        
        return res

    def deriv2Field2(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike):
        r""" Computes the Hessian, :math:`d^2V/(d\text{Field}^2)`.

        Parameters
        ----------
        fields : Fields
            The background field values (e.g.: Higgs, singlet)
        temperature : npt.ArrayLike
            Temperatures. Either scalar or a 1D array of same length as fields.NumPoints()

        Returns
        ----------
        d2VdField2 : list[Fields]
            Field Hessian of the potential. For each temperature, this is
            a matrix of the same size as Fields.
        """
        assert self.areDerivativesConfigured(), "EffectivePotential Error: configureDerivatives() must be "\
                                    "called before computing a derivative."
        axis = np.arange(self.fieldCount).tolist()
        return hessian(self.__wrapperPotential, self.__combineInputs(fields, temperature), epsilon=self.effectivePotentialError, 
                       scale=self.__combinedScales, xAxis=axis, yAxis=axis)
    
    def allSecondDerivatives(self, fields: Fields | FieldPoint, temperature: npt.ArrayLike):
        r""" Computes :math:`d^2V/(d\text{Field}^2)`, :math:`d^2V/(d\text{Field} dT)` 
        and :math:`d^2V/(dT^2)` at the ssame time. This function is more efficient
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
        assert self.areDerivativesConfigured(), "EffectivePotential Error: configureDerivatives() must be "\
                                    "called before computing a derivative."
        res = hessian(self.__wrapperPotential, self.__combineInputs(fields, temperature), epsilon=self.effectivePotentialError, scale=self.__combinedScales)
        
        hess = res[...,:-1,:-1]
        dgraddT = res[...,-1,:-1]
        d2VdT2 = res[...,-1,-1]
        
        return hess, dgraddT, d2VdT2
