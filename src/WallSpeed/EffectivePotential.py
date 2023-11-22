import numpy as np
import numpy.typing as npt
from typing import Tuple
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers
import scipy.optimize
import scipy.interpolate
from .helpers import derivative # derivatives for callable functions

import WallSpeed.Integrals

class EffectivePotential(ABC):

    ## In practice we'll get the model params from a GenericModel subclass 
    def __init__(self, modelParameters: dict[str, float],
                dPhi=1e-3,
                dT=1e-3):
        self.modelParameters = modelParameters
        self.dPhi = dPhi
        self.dT = dT


    # do the actual calculation of Veff(phi) here
    @abstractmethod
    def evaluate(self, fields: np.ndarray[float], temperature: float) -> complex:
        raise NotImplementedError
    

    ## LN: Use of this and fermion_massSq seem to be very tied to the Coleman-Weinberg part so I would call these something else, and perhaps  
    ## define a separate helper class for the output (holds mass squares, dofs etc)
    @abstractmethod
    def boson_massSq(self, fields, temperature):
        """
        Calculate the boson particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `fields` and `temperature`
            should be such that ``fields.shape[:-1]`` and ``temperature.shape`` are
            broadcastable (that is, ``fields[0,...]*T`` is a valid operation).

        Returns
        -------
        massSq : array_like
            A list of the boson particle masses at each input point `X`. The
            shape should be such that
            ``massSq.shape == (X[...,0]*T).shape + (Nbosons,)``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c : float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        """ 
        pass

    
    # LN: I included temperature here since it's confusing that the boson version takes T but this one doesn't
    @abstractmethod
    def fermion_massSq(self, fields, temperature):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `field`. The
            shape should be such that  ``massSq.shape == (field[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degreesOfFreedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            len
        """
        pass


    #### Non-abstract stuff from here on

    def findLocalMinimum(self, initialGuess: list[float], temperature: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Finds a local minimum starting from a given initial configuration of background fields.
        Feel free to override this if your model requires more delicate minimization.

        Returns
        -------
        minimum, functionValue : tuple. 
        minimum: list[float] is the location x of the minimum in field space.
        functionValue: float is Veff(x) evaluated at the minimum .
        If the input temperature is a numpy array, the returned values will be arrays of same length. 
        """

        # I think we'll need to manually vectorize this wrt. T

        if (np.isscalar(temperature)):
            # Minimize real part only
            evaluateWrapper = lambda fields: self.evaluate(fields, temperature).real
            res = scipy.optimize.minimize(evaluateWrapper, initialGuess)

            # this spams a LOT:
            """
            if (not res.success):
                print("Veff minimization error:", res.message)
            """
            ## res.x is the minimum location, res.fun is the function value
            return res.x, res.fun

        else:
            ## Veff(T) values in the minimum go here 
            resValue = np.empty_like(temperature)

            ## And field values in the minimum, at each T, go into resLocation. 
            # But since each element would now be a list, need magic to get the shape right (we put the fields column-wise)

            if (np.isscalar(initialGuess) or len(initialGuess.shape) == 1):
                nColumns = initialGuess.shape[0]
            else:
                ## Now we for some reason got multi-dimensional array of initialGuesses. TODO will this even work?
                _, nColumns = initialGuess.shape

            resLocation = np.empty( (len(temperature), nColumns) )

            for i in np.ndindex(temperature.shape):
                evaluateWrapper = lambda fields: self.evaluate(fields, temperature[i]).real

                res = scipy.optimize.minimize(evaluateWrapper, initialGuess)

                resLocation[i] = res.x
                resValue[i] = res.fun
        
            return resLocation, resValue
        
    ### end findLocalMinimum()


    ## Find Tc for two minima, search only range [TMin, TMax].
    ## Feel free to override this if your potential needs a more sophisticated minimization algorithm.
    def findCriticalTemperature(self, minimum1: np.ndarray[float], minimum2: np.ndarray[float], TMin: float, TMax: float) -> float:

        if (TMax < TMin):
            raise ValueError("findCriticalTemperature needs TMin < TMax")
    

        ## @todo Should probably do something more sophisticated so that we can update initial guesses for the minima during T-loop

        def freeEnergyDifference(inputT):
            _, f1 = self.findLocalMinimum(minimum1, inputT)
            _, f2 = self.findLocalMinimum(minimum2, inputT)
            return f2.real - f1.real
        

        ## start from TMin and increase temperature in small steps until the free energy difference changes sign

        T = TMin
        dT = 0.5 # If this is too large the high-T phase may disappear before we see the free-energy sign change. TODO better solution
        signAtStart = np.sign(freeEnergyDifference(T))
        bConverged = False

        while (T < TMax):
            T += dT
            if (np.sign(freeEnergyDifference(T)) != signAtStart):
                bConverged = True
                break

        if (not bConverged):
            raise RuntimeError("Could not find critical temperature")


        # Improve Tc estimate by solving DeltaF = 0 in narrow range near the above T 

        # NB: bracket will break if the function has same sign on both ends. The rough loop above should prevent this.
        rootResults = scipy.optimize.root_scalar(freeEnergyDifference, bracket=(T-dT, T), rtol=1e-8, xtol=1e-8)


        return rootResults.root
        #return T - dT/2. # use this if the root_scalar thing doesn't work?



    ## @todo do we want this, or use Philipp's version Jcw below?
    @staticmethod
    def ColemanWeinberg(massSquared: float, RGScale: float, c: float) -> complex:
        return massSquared**2 / (64.*np.pi**2) * ( np.log(massSquared / RGScale**2 + 0j) - c)


    @staticmethod
    def Jcw(msq: float, degrees_of_freedom: int, c: float, RGScale: float):
        """
        Coleman-Weinberg potential

        Parameters
        ----------
        msq : array_like
            A list of the boson particle masses at each input point `X`.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c: float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        RGScale: float
            Renormalization scale to use. Should not be an array.

        Returns
        -------
        Jcw : float or array_like
            One-loop Coleman-Weinberg potential for given particle spectrum.
        """
        # do we want to take abs of the mass??
        return degrees_of_freedom*msq*msq * (np.log(np.abs(msq/RGScale**2) + 1e-100) - c)
    


    ## LN: Why is this separate from Jcw?
    def V1(self, bosons, fermions, RGScale: float):
        """
        One-loop corrections to the zero-temperature effective potential
        in dimensional regularization.

        Parameters
        ----------
        bosons : array of floats
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : array of floats
            fermionic particle spectrum (here: masses, number of dofs)
        RGscale: float
            RG scale of the effective potential

        Returns
        -------
        V1 : float 
        """

        ## LN: should the return value actually be complex in general?

        m2, nb, c = bosons
        V = np.sum(self.Jcw(m2,nb,c, RGScale), axis=-1)

        m2, nf = fermions
        c = 1.5
        V -= np.sum(self.Jcw(m2,nf,c, RGScale), axis=-1)

        return V/(64*np.pi*np.pi)


    def pressureLO(self, bosons, fermions, T: npt.ArrayLike):
        """
        Computes the leading order pressure for the light degrees of freedom
        depending on the effective degrees of freedom.
        
        Parameters
        ----------
        bosons : array of floats
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : array of floats
            fermionic particle spectrum (here: masses, number of dofs)
        T : ArrayLike 

        Returns
        -------
        pressureLO : LO contribution to the pressure of light degrees of freedom

        """

        # TODO is this function OK with array input?

        T4 = T*T*T*T

        _,nb,_ = bosons
        _,nf = fermions

        V = 0
        if self.num_boson_dof is not None:
            nb = self.num_boson_dof - np.sum(nb)
            V -= nb * np.pi*np.pi / 90.
        if self.num_fermion_dof is not None:
            nf = self.num_fermion_dof - np.sum(nf)
            V -= nf * 7*np.pi*np.pi / 720.

        return V*T4


    @staticmethod
    def V1T(bosons, fermions, temperature: npt.ArrayLike):
        """
        One-loop thermal correction to the effective potential without any temperature expansions.

        Parameters
        ----------
        bosons : ArrayLike 
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : ArrayLike 
            fermionic particle spectrum (here: masses, number of dofs)
        temperature: ArrayLike 

        Returns
        -------
        V1T : 4d 1loop thermal potential 
        """ 
        
        ## m2 is shape (len(T), 5), so to divide by T we need to transpose T, or add new axis in this case.
        # But make sure we don't modify the input temperature array here. 
        T = np.atleast_1d(temperature)
        if (len(T) > 1):
            T = T[:, np.newaxis]

        m2,nb,_ = bosons
        T2 = (T*T) + 1e-100

        ## NB: Jb, Jf take (mass/T)^2 as input, np.array is OK

        V = np.sum(nb* WallSpeed.Integrals.Jb(m2/T2), axis=-1)
        m2,nf = fermions
        V += np.sum(nf* WallSpeed.Integrals.Jf(m2/T2), axis=-1)
        return V*T**4 / (2*np.pi*np.pi)


    def derivT(self, fields: np.ndarray[float], T: npt.ArrayLike):
        """
        The temperature-derivative of the effective potential.

        Parameters
        ----------
        fields : array of floats
            the field values (here: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        dfdT : float
            The temperature derivative of the free energy density at this field
            value and temperature.
        """

        return derivative(
            lambda T: self.evaluate(fields, T),
            T,
            dx=self.dT,
            n=1,
            order=4,
        )

    def derivField(self, fields: np.ndarray[float], T: npt.ArrayLike):
        """
        The field-derivative of the effective potential.

        Parameters
        ----------
        fields : array of floats
            the background field values (e.g.: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        dfdX : array_like
            The field derivative of the free energy density at this field
            value and temperature.
        """

        fields = np.asanyarray(fields, dtype=float)
        return_val = np.empty_like(fields)
        for i in range(len(fields)):
            field = fields[i,...] 
            Xd_field = fields.copy()
            Xd_field[i,...] += self.dPhi * np.ones_like(field)
            dfd_field = (self.evaluate(Xd_field,T) - self.evaluate(fields,T)) / self.dPhi
            return_val[i,...] = np.diag(dfd_field)

        return return_val