import numpy as np
import numpy.typing as npt
from typing import Tuple
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers
import scipy.optimize
import scipy.interpolate

from .helpers import derivative

from .Fields import Fields


class EffectivePotential(ABC):
    """Base class for the effective potential Veff. WallGo uses this to identify phases and their temperature dependence, 
    and computing free energies (pressures) in the two phases.
    
    Hydrodynamical routines in WallGo need the full pressure in the plasma, which in principle is p = -Veff(phi) if phi is a local minimum.
    However for phase transitions it is common to neglect field-independent parts of Veff, for example one may choose normalization so that Veff(0) = 0.
    Meanwhile for hydrodynamics we require knowledge of all temperature-dependent parts.
    With in mind, WallGo requires that the effective potential is defined with full T-dependence included.

    The final technicality you should be aware of is the variable fieldLowerBound, which is used as a cutoff for avoiding spurious behavior at phi = 0.
    You may need to adjust this to suit your needs, especially if using a complicated 2-loop potential. 
    """

    """TODO we could optimize some routines that only depend on free-energy differences ( dV/dField, findTc ) by
    separating the field-dependent parts of Veff(phi, T) and the T-dependent constant terms. This was done in intermediate commits
    but scrapped because it was too error prone (Veff appears in too many places). But let's keep this possibility in mind. 
    If attempting this, keep full Veff as the default and use the field-only part internally when needed.
    """


    ## How many background fields. This is explicitly required so that we can have better control over array shapes 
    fieldCount: int

    ## Lower bound for field values, used in normalize(). Using a small but nonzero value to avoid spurious divergences from eg. logarithms
    fieldLowerBound: float = 1e-8

    ## In practice we'll get the model params from a GenericModel subclass 
    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        self.modelParameters = modelParameters
        self.fieldCount = fieldCount

        ## Used for derivatives. TODO read from config file probably
        self.dT = 0.1  ## HACK! These defaults are bad, as the quantities have dimensions
        self.dPhi = 0.1  ## field difference


    @abstractmethod
    def evaluate(self, fields: Fields, temperature: npt.ArrayLike, checkForImaginary: bool = False) -> npt.ArrayLike:
        """Implement the actual computation of Veff(phi) here. The return value should be (the UV-finite part of) Veff 
        at the input field configuration and temperature. 
        
        Normalization of the potential DOES matter: You have to ensure that full T-dependence is included.
        Pay special attention to field-independent "constant" terms such as (minus the) pressure from light fermions. 
        """
        raise NotImplementedError("You are required to give an expression for the effective potential.")
    

    #### Non-abstract stuff from here on

    def findLocalMinimum(self, initialGuess: Fields, temperature: npt.ArrayLike, tol: float = None) -> Tuple[Fields, npt.ArrayLike]:
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

        numPoints = max(T.shape[0], initialGuess.NumPoints())

        ## Reshape for broadcasting
        guesses = initialGuess.Resize(numPoints, initialGuess.NumFields())
        T = np.resize(T, (numPoints))

        resValue = np.empty_like(T)
        resLocation = np.empty_like(guesses)

        for i in range(0, numPoints):

            """Numerically minimize the potential wrt. fields. 
            We can pass a fields array to scipy routines normally, but scipy seems to forcibly convert back to standard ndarray
            causing issues in the Veff evaluate function if it uses extended functionality from the Fields class. 
            So we need a wrapper that casts back to Fields type. It also needs to fix the temperature, and we only minimize the real part
            """

            def evaluateWrapper(fieldArray: np.ndarray):
                fields = Fields.CastFromNumpy(fieldArray)
                return self.evaluate(fields, T[i]).real

            guess = guesses.GetFieldPoint(i)

            res = scipy.optimize.minimize(evaluateWrapper, guess, tol=tol)

            resLocation[i] = res.x
            resValue[i] = res.fun

            # Check for presenece of imaginary parts at minimum
            self.evaluate(Fields((res.x)), T[i], checkForImaginary=True)

        ## Need to cast the field location
        return Fields.CastFromNumpy(resLocation), resValue

    def derivT(self, fields: Fields, temperature: npt.ArrayLike):
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
            dx=self.dT,
            n=1,
            order=4,
        )
        return der

    def derivField(self, fields: Fields, temperature: npt.ArrayLike):
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
        ## LN: had trouble setting the offset because numpy tried to use it integer and rounded it to 0. So paranoid dtype everywhere here

        res = np.empty_like(fields, dtype=float)

        for idx in range(fields.NumFields()):
            fieldsOffset = np.zeros_like(fields, dtype=float)
            fieldsOffset.SetField(
                idx, np.full(fields.NumPoints(), self.dPhi, dtype=float)
            )

            # O(dPhi^4) accurate, central finite difference scheme
            dV = (
                -self.evaluate(fields + 2 * fieldsOffset, temperature)
                + 8 * self.evaluate(fields + fieldsOffset, temperature)
                - 8 * self.evaluate(fields - fieldsOffset, temperature)
                + self.evaluate(fields - 2 * fieldsOffset, temperature)
            )
            dVdphi = dV / (12 * self.dPhi)

            res.SetField(idx, dVdphi)

        return res

    def deriv2FieldT(self, fields: Fields, temperature: npt.ArrayLike):
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
        if len(fields.shape) == 1:
            # HACK! Not sure how best to deal with this edge case, which
            # arises due to how scipyint.RK45 is initialised
            fields = Fields((fields))

        # O(dPhi^4) accurate, central finite difference scheme
        res = (
            - self.derivField(fields, temperature + 2 * self.dT)
            + 8 * self.derivField(fields, temperature + self.dT)
            - 8 * self.derivField(fields, temperature - self.dT)
            + self.derivField(fields, temperature - 2 * self.dT)
        ) / (12 * self.dT)

        if len(res.shape) == 2 and res.shape[0] == 1:
            # HACK! SHOULD DO THIS MORE TIDILY
            return res[0]
        return res

    def deriv2Field2(self, fields: Fields, temperature: npt.ArrayLike):
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
        if len(fields.shape) == 1:
            # HACK! Not sure how best to deal with this edge case, which
            # arises due to how scipyint.RK45 is initialised
            fields = Fields((fields))
        shapeRes = (fields.NumPoints(), fields.NumFields(), fields.NumFields())
        res = np.empty(shapeRes, dtype=float)

        # OG: This is all a bit of a mess, and should probably be rewritten
        # more algorithmically.

        for idx in range(fields.NumFields()):
            fieldsOffsetX = np.zeros_like(fields, dtype=float)
            fieldsOffsetX.SetField(
                idx, np.full(fields.NumPoints(), self.dPhi, dtype=float)
            )
            for idy in range(idx, fields.NumFields()):
                fieldsOffsetY = np.zeros_like(fields, dtype=float)
                if idy == idx:
                    """# O(dPhi^2) accurate, central finite difference scheme
                    dV = (
                        self.evaluate(fields + fieldsOffsetX, temperature)
                        - 2 * self.evaluate(fields, temperature)
                        + self.evaluate(fields - fieldsOffsetX, temperature)
                    )
                    res[..., idx, idy] = dV / self.dPhi ** 2"""
                    # O(dPhi^4) accurate, central finite difference scheme
                    dV = (
                        -self.evaluate(fields + 2 * fieldsOffsetX, temperature)
                        + 16 * self.evaluate(fields + fieldsOffsetX, temperature)
                        - 30 * self.evaluate(fields, temperature)
                        + 16 * self.evaluate(fields - fieldsOffsetX, temperature)
                        - self.evaluate(fields - 2 * fieldsOffsetX, temperature)
                    )
                    res[..., idx, idy] = dV / (12 * self.dPhi ** 2)
                else:
                    fieldsOffsetY.SetField(
                        idy, np.full(fields.NumPoints(), self.dPhi, dtype=float)
                    )
                    """# O(dPhi^2) accurate, central finite difference scheme
                    dV = (
                        self.evaluate(fields + fieldsOffsetX + fieldsOffsetY, temperature)
                        - self.evaluate(fields + fieldsOffsetX - fieldsOffsetY, temperature)
                        - self.evaluate(fields - fieldsOffsetX + fieldsOffsetY, temperature)
                        + self.evaluate(fields - fieldsOffsetX - fieldsOffsetY, temperature)
                    )
                    res[..., idx, idy] = dV / (4 * self.dPhi ** 2)
                    res[..., idy, idx] = res[..., idx, idy]"""
                    # O(dPhi^4) accurate, central finite difference scheme
                    dV = (
                        self.evaluate(fields + 2 * fieldsOffsetX + 2 * fieldsOffsetY, temperature)
                        - self.evaluate(fields - 2 * fieldsOffsetX + 2 * fieldsOffsetY, temperature)
                        - self.evaluate(fields + 2 * fieldsOffsetX - 2 * fieldsOffsetY, temperature)
                        + self.evaluate(fields - 2 * fieldsOffsetX - 2 * fieldsOffsetY, temperature)
                    )
                    dV += 8 * (
                        -self.evaluate(fields + 2 * fieldsOffsetX + fieldsOffsetY, temperature)
                        + self.evaluate(fields - 2 * fieldsOffsetX + fieldsOffsetY, temperature)
                        + self.evaluate(fields + 2 * fieldsOffsetX - fieldsOffsetY, temperature)
                        - self.evaluate(fields - 2 * fieldsOffsetX - fieldsOffsetY, temperature)
                        - self.evaluate(fields + fieldsOffsetX + 2 * fieldsOffsetY, temperature)
                        + self.evaluate(fields - fieldsOffsetX + 2 * fieldsOffsetY, temperature)
                        + self.evaluate(fields + fieldsOffsetX - 2 * fieldsOffsetY, temperature)
                        - self.evaluate(fields - fieldsOffsetX - 2 * fieldsOffsetY, temperature)
                    )
                    dV += 64 * (
                        self.evaluate(fields + fieldsOffsetX + fieldsOffsetY, temperature)
                        - self.evaluate(fields - fieldsOffsetX + fieldsOffsetY, temperature)
                        - self.evaluate(fields + fieldsOffsetX - fieldsOffsetY, temperature)
                        + self.evaluate(fields - fieldsOffsetX - fieldsOffsetY, temperature)
                    )
                    res[..., idx, idy] = dV / (144 * self.dPhi ** 2)
                    res[..., idy, idx] = res[..., idx, idy]

        if len(res.shape) == 3 and res.shape[0] == 1:
            # HACK! SHOULD DO THIS MORE TIDILY
            return res[0]
        return res
