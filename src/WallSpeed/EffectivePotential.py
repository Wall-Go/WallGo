import numpy as np
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers
import scipy.optimize
import scipy.interpolate

import WallSpeed.Integrals

class EffectivePotential(ABC):

    ## In practice we'll get the model params from a GenericModel subclass 
    def __init__(self, modelParameters: dict[str, float]):
        self.modelParameters = modelParameters


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

    ## Finds a local minimum starting from a given initial configuration of background fields.
    ## Feel free to override this if your model requires more delicate minimization.
    def findLocalMinimum(self, initialGuess: np.ndarray[float], temperature: float) -> tuple:
        """
        Returns
        -------
        minimum, functionValue : tuple, location x of the minimum and value of Veff(x) 
        """

        ## Minimize real part only
        evaluateWrapper = lambda fields: self.evaluate(fields, temperature).real

        res = scipy.optimize.minimize(evaluateWrapper, initialGuess)

        # this spams a LOT:
        """
        if (not res.success):
            print("Veff minimization error:", res.message)
        """
            
        ## res.x is the minimum location, res.fun is the function value
        return res.x, res.fun
    
    

    ## Find Tc for two minima, search only range [TMin, TMax].
    ## Feel free to override this.
    def findCriticalTemperature(self, minimum1: np.ndarray[float], minimum2: np.ndarray[float], TMin: float, TMax: float):

        if (TMax < TMin):
            raise ValueError("findCriticalTemperature needs TMin < TMax")
    

        ## @todo Should probably do something more sophisticated so that we can update initial guesses for the minima during T-loop

        def freeEnergyDifference(inputT):
            _, f1 = self.findLocalMinimum(minimum1, inputT)
            _, f2 = self.findLocalMinimum(minimum2, inputT)
            return f2.real - f1.real
        

        ## start from TMin and increase temperature in small steps until the free energy difference changes sign

        T = TMin
        dT = 0.5
        signAtStart = np.sign(freeEnergyDifference(T))
        bConverged = False

        while (T < TMax):
            T += dT
            if (np.sign(freeEnergyDifference(T)) != signAtStart):
                bConverged = True
                break

        if (not bConverged):
            raise RuntimeError("Could not find critical temperature")


        # Improve Tc estimate by DeltaF = 0 in narrow range near the above T 

        # NB: bracket will break if the function has same sign on both ends. The rough loop above should prevent this.
        rootResults = scipy.optimize.root_scalar(freeEnergyDifference, bracket=(T-dT, T), rtol=1e-8, xtol=1e-8)

        # LN: In general root_scalar doesn't seem to work very well. 
        # I think there's an issue if the low-T phase becomes unstable very quickly at T > Tc, and our initial dT was too large
        
        return rootResults.root
        #return T - dT/2. # use this if the root_scalar thing doesn't work?



    ## @todo do we want this, or use Philipp's version Jcw below?
    @staticmethod
    def ColemanWeinberg(massSquared: float, RGScale: float, c: float) -> complex:
        return massSquared**2 / (64.*np.pi**2) * ( np.log(massSquared / RGScale**2 + 0j) - c)


    @staticmethod
    def Jcw(msq, degrees_of_freedom, c, RGScale: float):
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
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.

        Parameters
        ----------
        bosons : array of floats
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : array of floats
            fermionic particle spectrum (here: masses, number of dofs)

        Returns
        -------
        V1 : 1loop vacuum contribution to the pressure

        """
        m2, nb, c = bosons
        V = np.sum(self.Jcw(m2,nb,c, RGScale), axis=-1)

        m2, nf = fermions
        c = 1.5
        V -= np.sum(self.Jcw(m2,nf,c, RGScale), axis=-1)

        return V/(64*np.pi*np.pi)


    def PressureLO(self, bosons, fermions, T):
        """
        Computes the leading order pressure
        depending on the effective degrees of freedom.
        
        Parameters
        ----------
        bosons : array of floats
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : array of floats
            fermionic particle spectrum (here: masses, number of dofs)

        Returns
        -------
        PressureLO : LO contribution to the pressure

        """
        T4 = T*T*T*T

        _,nb,_ = bosons
        _,nf = fermions

        V = 0
        if self.num_boson_dof is not None:
            nb = self.num_boson_dof - np.sum(nb)
            V -= nb * np.pi**4 / 45.
        if self.num_fermion_dof is not None:
            nf = self.num_fermion_dof - np.sum(nf)
            V -= nf * 7*np.pi**4 / 360.

        return V*T4/(2*np.pi*np.pi)


    @staticmethod
    def V1T(bosons, fermions, T):
        """
        One-loop thermal correction to the effective potential without any temperature expansions.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.

        Note
        ----
        The `Jf` and `Jb` functions used here are
        aliases for :func:`finiteT.Jf_spline` and :func:`finiteT.Jb_spline`,
        each of which accept mass over temperature *squared* as inputs
        (this allows for negative mass-squared values, which I take to be the
        real part of the defining integrals.

        .. todo::
            Implement new versions of Jf and Jb that return zero when m=0, only
            adding in the field-independent piece later if
            ``include_radiation == True``. This should reduce floating point
            errors when taking derivatives at very high temperature, where
            the field-independent contribution is much larger than the
            field-dependent contribution.

        Parameters
        ----------

        Returns
        -------
        V1T : 4d 1loop thermal potential 
        """ 
        
        """ ??????
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        """

        m2,nb,_ = bosons
        V = np.sum(nb* WallSpeed.Integrals.Jb(m2/T**2 + 1e-100), axis=-1)
        m2,nf = fermions
        V += np.sum(nf* WallSpeed.Integrals.Jf(m2/T**2 + 1e-100), axis=-1)
        return V*T**4 / (2*np.pi*np.pi)

    

