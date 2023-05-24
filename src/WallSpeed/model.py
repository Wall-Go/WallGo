import numpy as np
import math
import warnings
import DRwizard
from scipy import integrate,interpolate,optimize,special,linalg,stats
from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf


class Model:
    '''
    Class that generates the model given external model file
    '''
    def __init__(self,mu3D,mu4D,ms,lambdaS,lambdaHS):
#    def __init__(self,scalarMasses,scalarCoupling,mu3D,mu4D):
        '''
        Initialise class
        '''
        self.ms,self.lambdaS,self.lambdaHS = ms,lambdaS,lambdaHS
        self.v0 = 246
        self.mh = 125
        self.lambdaH = self.mh**2/(2*self.v0**2)
        self.muh2 = -self.lambdaH*self.v0**2
        self.mus2 = +self.ms**2-self.lambdaHS*self.v0**2/2
        '''
        Number of bosonic and fermionic dofs
        '''
        self.num_boson_dof = 29
        self.num_fermion_dof = 90
        '''
        Number of fermion generations and colors
        '''
        self.nf = 3
        self.Nc = 3
        '''
        3D and 4D RG scale of EFT as fraction of temperature
        '''
        self.mu3D = mu3D
        self.mu4D = mu4D
        '''
        Z,W,t mass, strong gauge coupling and fermion generations
        '''
        self.MW = 80.379
        self.MZ = 91.1876
        self.Mt = 172.76
        #self.couplings = {}

        self.g2 = 2*self.MW/self.v0
        self.g1 = math.sqrt((2*self.MZ/self.v0)**2-self.g2**2)
        self.yt = math.sqrt(2)*self.Mt/self.v0
        self.musT = (
                +1./6*lambdaHS
                +1./4*lambdaS)
        self.muhT = (
                (
                +1*self.g1**2
                +3*self.g2**2
                +4./3*self.Nc*self.yt**2
                +8*self.lambdaH)/16
                +self.lambdaHS/24
                )

    def Run4Dparams(self,T):
        '''
        Runs 4D couplings at matching scale mu4D
        Takes list of all parameters and beta functions from DRwizard and runs them
        '''
        pars, beta = np.loadtxt('BetaFunctions4D.dat',delimiter=',',unpack=True,dtype=str)
        #data, pars= np.array_split(np.loadtxt('BetaFunctions4D.dat', dtype=str), [-1], axis=1)
        print(pars)
        print(beta)
        #print(pars.flatten().tolist())
        muBar = self.mu4D
        #ListofAllparams = BetaFunctions(self.ListofAllparams ).solveBetaFuncs(muBar)[1]
        return


    def J3(self,m):
        '''
        Log integral for 1loop 3d Veff
        '''
        return -m**3/(12*np.pi)

    def Vtree(self,X,show_V=True):
        '''
        Tree level effective potential
        X
        '''
        X = np.asanyarray(X)
        h1 = X[...,0]
        s1 = X[...,1]
        with open('Veff3dLO.dat','r') as file:
            vtree = file.read()
        #vtree = vtree.replace("lambdaH","lambdaH.self")
        #vtree = eval(vtree)

        vtree = (
            +self.muh2*h1**2/2
            +self.mus2*s1**2/2
            +self.lambdaH*h1**4/4
            +self.lambdaS*s1**4/4
            +self.lambdaHS*(h1*s1)**2/4)
        if show_V:
            print(vtree)
        return vtree

 # MINIMIZATION AND TRANSITION ANALYSIS --------------------------------

    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima.

        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.

        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        return np.array([
            [np.sqrt(-min(-abs(self.muh2)/20,self.muh2+self.muhT*T**2)/self.lambdaH),0],
            [0,np.sqrt(-min(-abs(self.mus2)/20,self.mus2+self.musT*T**2)/self.lambdaS)]
            ])

    def findMinimum(self, X=None, T=0.0):
        '''
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        '''
        if X is None:
            X = self.approxZeroTMin()[0]
        fh = lambda h: self.Vtot([abs(h),0],T)
        fs = lambda s: self.Vtot([0,abs(s)],T)

        return optimize.fmin(self.Vtot, X, args=(T,), disp=0)

    def Veff4d(self):
        '''
        Thermal effective potential at 1 loop in 4d
        '''

    def Veff3d(self):
        '''
        Thermal effective potential at 1 loop at the supersoft scale
        '''

    def dVdT(self,X,T,include_radiation=True):
        '''
        1st T-derivative of the effective potential
        '''

    def d2VdT2(self,X,T,include_radiation=True):
        '''
        2nd T-derivative of the effective potential
        '''

    def d3VdT3(self,X,T,include_radiation=True):
        '''
        3rd T-derivative of the effective potential
        '''

    def Vefftot(self):
        '''
        Function that generates the effective potential given a pre-defined model
        '''

# 100,h 110, s 130
#def main():
pot = Model(1,1,160,1.0,1.6)
#pot.Run4Dparams(1)
pot.Vtree([[0,0],[110,130]])

print("hello")

# if __name__ == '__main__':
#     main()
