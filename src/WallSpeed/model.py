import numpy as np
import math
import warnings
#import DRwizard
from scipy import integrate,interpolate,optimize,special,linalg,stats
from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf


class Particle:
    def __init__(
        self,
        msqVaccum,
        msqThermal,
        statistics,
        isOutOfEquilibrium,
        collisionPrefactors,
    ):
        self.msqVacuum = msqVaccum
        self.msqThermal = msqThermal
        assert statistics in [
            -1,
            1,
        ], "Particle error: statistics not understood %s" % (
            statistics
        )
        self.statistics = statistics
        self.isOutOfEquilibrium = isOutOfEquilibrium
        self.collisionPrefactors = collisionPrefactors

class Model:
    '''
    Class that generates the model given external model file
    '''
    def __init__(self,mu3D,mu4D,ms,lambdaS,lambdaHS):
#    def __init__(self,scalarMasses,scalarCoupling,mu3D,mu4D):
        '''
        Initialize class
        '''
        self.ms = ms
        self.lambdaS = lambdaS
        self.lambdaHS = lambdaHS
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
        self.mu4Dsq = mu4D*mu4D
        self.mu3Dsq = mu3D*mu3D
        '''
        Z,W,t mass, strong gauge coupling and fermion generations
        '''
        self.MW = 80.379
        self.MZ = 91.1876
        self.Mt = 172.76
        #self.couplings = {}

        self.g0 = 2*self.MW/self.v0
        self.g1 = self.g0*math.sqrt((self.MZ/self.MW)**2-1)
        self.g2 = self.g0
        self.yt = math.sqrt(1/2)*self.g0*self.Mt/self.MW
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
        '''
        Define dictionary of used parameters
        '''
        self.pars = {
                'muh2': self.muh2,
                'mus2': self.mus2,
                'lambdaH': self.lambdaH,
                'lambdaS': self.lambdaS,
                'lambdaHS': self.lambdaHS
        }

   # def Run4Dparams(self,T):
   #     '''
   #     Runs 4D couplings at matching scale mu4D
   #     Takes list of all parameters and beta functions from DRwizard and runs them
   #     '''
   #     pars, beta = np.loadtxt('BetaFunctions4D.dat',delimiter=',',unpack=True,dtype=str)
   #     #data, pars= np.array_split(np.loadtxt('BetaFunctions4D.dat', dtype=str), [-1], axis=1)
   #     print(pars)
   #     print(beta)
   #     #print(pars.flatten().tolist())
   #     muBar = self.mu4D
   #     #ListofAllparams = BetaFunctions(self.ListofAllparams ).solveBetaFuncs(muBar)[1]
   #     return


    # EFFECTIVE POTENTIAL CALCULATIONS -----------------------

    def J3(self,m):
        '''
        Log integral for 1loop 3d Veff
        '''
        return -m**3/(12*np.pi)

    def readModel(self,filePath,args,fields):
        file = open(filePath,'r')
        fileContent = file.read()
        file.close()

        argsInternal = args.copy()
        for k, v in argsInternal.items():
            k = v

        return eval(fileContent,fields,argsInternal)

    def V0(self,X,show_V=False):
        '''
        Tree level effective potential
        X
        '''
        X = np.asanyarray(X)
        h1 = X[...,0]
        s1 = X[...,1]

        fields = {
                'h1':h1,
                's1':s1
            }

        V = self.readModel('Veff3dLO.dat',self.pars,fields)
        print(V)

        #V = (
        #    +self.muh2*h1**2/2
        #    +self.mus2*s1**2/2
        #    +self.lambdaH*h1**4/4
        #    +self.lambdaS*s1**4/4
        #    +self.lambdaHS*(h1*s1)**2/4)
        if show_V:
            print(V)
        return V

    def Jcw(self,msq,n,c):
        '''
        Coleman-Weinberg potential
        '''
        return n*msq*msq * (np.log(np.abs(msq/self.mu4Dsq) + 1e-100) - c)

    def boson_massSq(self, X, T):
        X = np.asanyarray(X)
        h1,s1 = X[...,0],X[...,1]

        Nbosons = 5
        dof = np.array([1,1,3,6,3])#h,s,chi,W,Z
        c = np.array([1.5,1.5,1.5,5/6,5/6])

        '''
        mass matrix
        TODO: numerical determination of scalar masses from V0
        '''
        mh2 = self.muh2+3*self.lambdaH*h1**2+self.lambdaHS*s1**2/2
        ms2 = self.mus2+3*self.lambdaS*s1**2+self.lambdaHS*h1**2/2
        mhs2 = self.lambdaHS*h1*s1
        sqrt = np.sqrt((mh2-ms2)**2+4*mhs2**2)
        m1 = (mh2+ms2)/2+sqrt/2
        m2 = (mh2+ms2)/2-sqrt/2
        mChi = self.muh2+self.lambdaH*h1**2+self.lambdaHS*s1**2/2
        mz = (self.g1**2+self.g2**2)*h1**2/4
        mw = self.g2**2*h1**2/4

        massSq = np.column_stack((m1, m2, mChi, mw, mz))
        return massSq,dof,c

    def fermion_massSq(self, X):
        X = np.asanyarray(X)
        h1,s1 = X[...,0],X[...,1]

        Nfermions = 1
        dof = np.array([12])
        mt = self.yt**2*h1**2/2
        # todo include spins for each particle

        massSq = np.column_stack((mt,))
        return massSq,dof


    def V1(self, bosons, fermions):
        '''
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.

        This is generally not called directly, but is instead used by
        :func:`Vtot`.
        '''
        m2, nb, c = bosons
        #V = np.sum(n*m2*m2 * (np.log(np.abs(m2/self.mu4Dsq) + 1e-100)
        #                      - c), axis=-1)
        V = np.sum(self.Jcw(m2,nb,c), axis=-1)

        m2, nf = fermions
        c = 1.5
        V -= np.sum(self.Jcw(m2,nf,c), axis=-1)

        return V/(64*np.pi*np.pi)

    def V1T(self, bosons, fermions, T, include_radiation=True):
        '''
        The one-loop finite-temperature potential.

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
        '''
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        m2, nb, c = bosons
        V = np.sum(nb*Jb(m2/T2), axis=-1)
        m2, nf = fermions
        V += np.sum(nf*Jf(m2/T2), axis=-1)
        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                V -= nb * np.pi**4 / 45.
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                V -= nf * 7*np.pi**4 / 360.
        return V*T4/(2*np.pi*np.pi)

    def Vtot(self, X, T, include_radiation=True):
        '''
        The total finite temperature effective potential.

        Parameters
        ----------
        X : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        '''
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        V = self.V0(X)
        V += self.V1(bosons, fermions)
        V += self.V1T(bosons, fermions, T, include_radiation)
        return np.real(V)

    def energyDensity(self,X,T,include_radiation=True):
        '''
        Print energy density
        '''
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.V1T_from_X(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.V1T_from_X(X,T-T_eps, include_radiation)
            dVdT += 8*self.V1T_from_X(X,T+T_eps, include_radiation)
            dVdT -= self.V1T_from_X(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        V = self.Vtot(X,T, include_radiation)
        return V - T*dVdT

    # MINIMIZATION AND TRANSITION ANALYSIS --------------------------------

    def approxZeroTMin(self,T=0):
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
        muh2_abs = abs(self.muh2) / 20
        mus2_abs = abs(self.mus2) / 20
        lambdaH_sqrt = np.sqrt(self.lambdaH)
        lambdaS_sqrt = np.sqrt(self.lambdaS)
        muhT_squared = self.muhT * T**2
        musT_squared = self.musT * T**2

        m1 = np.sqrt(-min(-muh2_abs, self.muh2 + muhT_squared) / lambdaH_sqrt)
        m2 = np.sqrt(-min(-mus2_abs, self.mus2 + musT_squared) / lambdaS_sqrt)

        # This should be overridden.
        #return np.array([[m1, 0], [0, m2]])
        return np.array([[m1, 0], [0, m2]])


    def findMinimum(self, X=None, T=0.0):
        '''
        Convenience function for finding the nearest minimum to `X` at
        temperature `T`.
        '''
        if X is None:
            X = self.approxZeroTMin()[0]
        #fh = lambda h: self.Vtot([abs(h),0],T)
        #fs = lambda s: self.Vtot([0,abs(s)],T)

        #result = optimize.fmin(self.Vtot, X, args=(T,), disp=0)
        result = optimize.minimize(self.Vtot, X, args=(T,),
                                   method='Nelder-Mead',
                                   tol=1e-12,options={'disp': False})

        return result.x

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

# 100,h 110, s 130
#def main():
pot = Model(1,125,160,1.0,1.2)
#pot.Run4Dparams(1)
print(pot.V0([[110,130]]))
#print(pot.Vtot([[110,140],[110,130]],[110]))
#print(pot.findMinimum(None,110))


# if __name__ == '__main__':
#     main()
