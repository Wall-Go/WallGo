import numpy as np
import numpy.typing as npt
from typing import Tuple
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers
import scipy.optimize
import scipy.interpolate
from .helpers import derivative # derivatives for callable functions

class EffectivePotential(ABC):

     
    ## How many background fields. This is explicitly required so that we can have better control over array shapes 
    fieldCount: int

    ## In practice we'll get the model params from a GenericModel subclass 
    def __init__(self, modelParameters: dict[str, float], fieldCount: int,
                dPhi=1e-3,
                dT=1e-3
                ):
        self.modelParameters = modelParameters
        self.fieldCount = fieldCount
        self.dPhi = dPhi
        self.dT = dT


    
    @abstractmethod
    def evaluate(self, fields: np.ndarray[float], temperature: float) -> complex:
        # do the actual calculation of Veff(phi) here
        raise NotImplementedError
    

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

        temperature = np.asanyarray(temperature)
        guessArray = np.asanyarray(initialGuess)

        # I think we'll need to manually vectorize this wrt. T

        ## np.isscalar(x) does not catch the case where x is np.ndarray of dim 0
        if (np.isscalar(temperature) or np.ndim(temperature) == 0):             

            ## Make sure that we only one initial guess
            assert np.ndim(guessArray) == 0 if self.fieldCount == 1 else len(guessArray) == self.fieldCount

            # Minimize real part only
            evaluateWrapper = lambda fields: self.evaluate(fields, temperature).real
            res = scipy.optimize.minimize(evaluateWrapper, guessArray)

            # this spams a LOT:
            """
            if (not res.success):
                print("Veff minimization error:", res.message)
            """
            ## res.x is the minimum location, res.fun is the function value
            return res.x, res.fun

        else:
            ## Got many input temperatures. Veff(T) values in the minimum will go here 
            resValue = np.empty_like(temperature)

            ## And field values in the minimum, at each T, will go into resLocation.
            # We put them column wise, so that resLocation.shape = ( len(T), self.fieldCount ).

            resLocation = np.empty( (len(temperature), self.fieldCount) )
            
            ## Make sure that we didn't get more than one initial guess for each T
            ## TODO I hate this but dunno how to do it better:

            if (self.fieldCount == 1):
                
                if (np.ndim(guessArray) != 0):
                    assert len(temperature) == len(guessArray)

                # Else: just got one guess which is fine, we broadcast and use that for all T
                
            else:
                ## Now each initial guess is 1D in itself
                if (np.ndim(guessArray) == 1):
                    assert len(guessArray) == self.fieldCount
                
                else:
                    assert guessArray.shape == (len(temperature), self.fieldCount)

            ## Shapes probably ok...

            for i in np.ndindex(temperature.shape):
                evaluateWrapper = lambda fields: self.evaluate(fields, temperature[i]).real

                res = scipy.optimize.minimize(evaluateWrapper, guessArray)

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

        ## TODO use eq. (39) from https://arxiv.org/pdf/hep-ph/0510375.pdf.
        ## This is probably easier than having the user input degrees of freedom manually

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
        val = derivative(
            lambda T: self.evaluate(fields, T).real,
            T,
            dx=self.dT,
            n=1,
            order=4,
        )
        # print(val)
        return val

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
        
        # print(return_val)
        return return_val
