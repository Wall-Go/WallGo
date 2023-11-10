import numpy as np
import numpy.typing as npt
from abc import abstractmethod
import scipy.integrate

from .InterpolatableFunction import InterpolatableFunction


r""" For 1-loop thermal potential WITHOUT high-T approximations, need to calculate
:math:`J_b(x) =  \int_0^\infty dy y^2 \ln( 1 - \exp(-\sqrt(y^2 + x) ))` (bosonic)
and 
:math:`J_f(x) =  -\int_0^\infty dy y^2 \ln( 1 + \exp(-\sqrt(y^2 + x) ))` (fermionic).
The thermal 1-loop correction from one particle species with N degrees of freedom is then
:math:`V_1(T) = T^4/(2\pi^2) N J(m^2 / T^2)`.
See eg. CosmoTransitions (arXiv:1109.4189, eq. (12)). 
Particularly for scalars the m^2 can be negative so we allow x < 0, 
but we calculate the real parts of integrals only (@todo imag parts?)
"""


# Bosonic Jb(x), in practice use with x = m^2 / T^2
class JbIntegral(InterpolatableFunction):

    ## LN: This doesn't vectorize nicely for numpy due to combination of piecewise scipy.integrate.quad and conditionals on x.
    # So for array input, let's just do a simple for loop 
    def _functionImplementation(self, xInput: npt.ArrayLike) -> npt.ArrayLike:
        """
        xInput: float or numpy array of floats.
        """

        def wrapper(x: float):
            # This is the integrand if always y^2 + x >= 0 
            integrand = lambda y:  y*y * np.log( 1. - np.exp(-np.sqrt( y*y + x )) )

            if (x >= 0):
                res = scipy.integrate.quad(integrand, 0.0, np.inf)[0]

            else:
                ## Now need to analytically continue, split the integral into two parts (complex for y < sqrt(-x))
                # Do some algebra to find the principal log (we do real parts only)
                integrand_principalLog = lambda y: y*y * np.log( 2. * np.abs(np.sin(0.5 * np.sqrt( -(y*y + x) ))) )
                
                res = ( scipy.integrate.quad(integrand_principalLog, 0.0, np.sqrt(np.abs(x)))[0]
                    + scipy.integrate.quad(integrand, np.sqrt(np.abs(x)), np.inf)[0]
                    )
            return res.real
        
        if (np.isscalar(xInput)):
            return wrapper(xInput)
        
        else:
            results = np.empty_like(xInput, dtype=float) # same shape as xInput
            for i in np.ndindex(xInput.shape):
                results[i] = wrapper(xInput[i])
            
            return results

                    
    

# Fermionic Jf(x), in practice use with x = m^2 / T^2. This is very similar to the bosonic counterpart Jb
class JfIntegral(InterpolatableFunction):
    
    def _functionImplementation(self, xInput: npt.ArrayLike) -> npt.ArrayLike:
        """
        xInput: float or numpy array of floats.
        """

        def wrapper(x: float):
            # This is the integrand if always y^2 + x >= 0 
            integrand = lambda y:  y*y * np.log( 1. + np.exp(-np.sqrt( y*y + x )) )

            if (x >= 0):
                res = scipy.integrate.quad(integrand, 0.0, np.inf)[0]

            else:
                # Like Jb but now we get a cos
                integrand_principalLog = lambda y: y*y * np.log( 2. * np.abs(np.cos(0.5 * np.sqrt( -(y*y + x) ))) )
                
                res = ( scipy.integrate.quad(integrand_principalLog, 0.0, np.sqrt(np.abs(x)))[0]
                    + scipy.integrate.quad(integrand, np.sqrt(np.abs(x)), np.inf)[0]
                    )
                
            # overall minus sign for Jf
            return -res.real
        
        if (np.isscalar(xInput)):
            return wrapper(xInput)
        
        else:
            results = np.empty_like(xInput, dtype=float) # same shape as xInput
            for i in np.ndindex(xInput.shape):
                results[i] = wrapper(xInput[i])
            
            return results


Jb = JbIntegral(bUseAdaptiveInterpolation=True)
Jf = JfIntegral(bUseAdaptiveInterpolation=True)


##### init, TODO somewhere else

from pathlib import Path
sourcePath = Path(__file__).resolve()
sourceDir = sourcePath.parent

JbFile = str(sourceDir) + "/finiteT_b.dat.txt"
#Jb.readInterpolationTable(JbFile)


JfFile = str(sourceDir) + "/finiteT_f.dat.txt"
#Jf.readInterpolationTable(JfFile)
