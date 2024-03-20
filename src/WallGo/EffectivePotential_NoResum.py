import numpy as np
import numpy.typing as npt
from abc import ABC, abstractmethod

from .EffectivePotential import EffectivePotential
from .Integrals import Integrals

class EffectivePotential_NoResum(EffectivePotential, ABC):
    """Class EffectivePotential_NoResum -- Specialization of the abstract EffectivePotential class
    that implements common functions for computing the 1-loop potential at finite temperature, without 
    any assumptions regarding the temperature (no high- or low-T approximations). 
    In some literature this would be the ``4D effective potential''. 

    """

    integrals: Integrals


    def __init__(self, modelParameters: dict[str, float], fieldCount: int, integrals: Integrals = None):
        ##
        super().__init__(modelParameters, fieldCount)
        
        ## Use the passed Integrals object if provided, otherwise create a new one with default settings
        if integrals:
            self.integrals = integrals 
        else:
            self.integrals = Integrals()

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

    ## @todo do we want this, or use Philipp's version Jcw below?
    @staticmethod
    def ColemanWeinberg(massSquared: float, RGScale: float, c: float) -> complex:
        return massSquared**2 / (64.*np.pi**2) * (np.log(massSquared / RGScale**2 + 0j) - c)


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
    def V1(self, bosons, fermions, RGScale: float, checkForImaginary: bool = False):
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
        V = np.sum(self.Jcw(m2, nb, c, RGScale), axis=-1)

        m2, nf = fermions
        c = 1.5
        V -= np.sum(self.Jcw(m2, nf, c, RGScale), axis=-1)

        if checkForImaginary and np.any(m2 < 0):
            try:
                VI = V.imag/(64*np.pi*np.pi)[np.any(m2 < 0, axis=0)]
            except:
                VI = V.imag/(64*np.pi*np.pi)
            print(f"Im(V1)={VI}")

        return V/(64*np.pi*np.pi)


    def V1T(self, bosons, fermions, temperature: npt.ArrayLike, checkForImaginary: bool = False):
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
        T = np.asanyarray(temperature)
        
        T2 = T*T + 1e-100

        ## Need reshaping mess for numpy broadcasting to work
        if (T2.ndim > 0): 
            T2 = T2[:, np.newaxis]

        ## Jb, Jf take (mass/T)^2 as input, np.array is OK.
        ## Do note that for negative m^2 the integrals become wild and convergence is both slow and bad,
        ## so you may want to consider taking the absolute value of m^2. We will not enforce this however
            
        ## Careful with the sum, it needs to be column-wise. Otherwise things go horribly wrong with array T input. 
        ## TODO really not a fan of hardcoded axis index 
         
        m2, nb, _ = bosons
        V = np.sum(nb* self.integrals.Jb(m2 / T2), axis=-1)

        m2, nf = fermions
        V += np.sum(nf* self.integrals.Jf(m2 / T2), axis=-1)

        if checkForImaginary and np.any(m2 < 0):
            try:
                VI = V.imag*T**4 / (2*np.pi*np.pi)[np.any(m2 < 0, axis=-1)]
            except:
                VI = V.imag*T**4 / (2*np.pi*np.pi)
            print(f"Im(VT)={VI}")
        
        return V*T**4 / (2*np.pi*np.pi)