import numpy as np
from abc import abstractmethod
import scipy.integrate

import WallSpeed.InterpolatableFunction 


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

## These integrals should be "static"/"singletons" so let's implement that
class InterpolatedIntegral(WallSpeed.InterpolatableFunction):

    @abstractmethod 
    def _evaluate(self, x): 
        pass

    @classmethod
    def createStaticInstance(cls):
        return cls()



# Bosonic Jb(x), in practice use with x = m^2 / T^2
class JbIntegral(InterpolatedIntegral):

    ## TODO would be good to make this work also with np array input

    def _evaluate(self, x: float) -> float:

        # This is the integrand y^2 + x >= 0 always 
        integrand = lambda y:  y*y * np.log( 1. - np.exp(-np.sqrt( y*y + x )) )

        # Using scipy.integrate.quad which returns a tuple. It has things like 
        # abserr, infodict etc but I don't think we need those, take just the result (first element) 

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
    


Jb = JbIntegral().createStaticInstance()

"""
print(Jb(-2.5))

from pathlib import Path
sourcePath = Path(__file__).resolve()
sourceDir = sourcePath.parent

JbFile = str(sourceDir) + "/finiteT_b.dat.txt"

Jb.readInterpolationTable(JbFile)
print(Jb(-2.5))


Jb.makeInterpolationTable(-100., 1000, 10000)
Jb.writeInterpolationTable("testTable.dat")
"""