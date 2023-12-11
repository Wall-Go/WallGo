import numpy as np
import numpy.typing as npt
from typing import Tuple
from abc import ABC, abstractmethod ## Abstract Base Class
import cmath # complex numbers
import scipy.optimize
import scipy.interpolate

class EffectivePotential(ABC):

    ## In practice we'll get the model params from a GenericModel subclass 
    def __init__(self, modelParameters: dict[str, float]):
        self.modelParameters = modelParameters


    
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

        ## We kinda need to ensure that this is a numpy array
        guessArray = np.asanyarray(initialGuess)

        # I think we'll need to manually vectorize this wrt. T

        if (np.isscalar(temperature)):
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
            ## Veff(T) values in the minimum go here 
            resValue = np.empty_like(temperature)

            ## And field values in the minimum, at each T, go into resLocation. 
            # But since each element would now be a list, need magic to get the shape right (we put the fields column-wise)

            if (np.isscalar(guessArray) or len(guessArray.shape) == 1):
                nColumns = guessArray.shape[0]
            else:
                ## Now we for some reason got multi-dimensional array of initialGuesses. TODO will this even work?
                _, nColumns = guessArray.shape

            resLocation = np.empty( (len(temperature), nColumns) )

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
