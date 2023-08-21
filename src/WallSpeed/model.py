"""
Classes for user input of models
"""
import numpy as np # arrays, maths and stuff
from scipy import integrate,interpolate,optimize,special,linalg,stats
from .helpers import derivative # derivatives for callable functions


class Particle:
    """Particle configuration

    A simple class holding attributes of an out-of-equilibrium particle as
    relevant for calculations of Boltzmann equations.
    """
    STATISTICS_OPTIONS = ["Fermion", "Boson"]

    def __init__(
        self,
        name,
        msqVacuum,
        msqThermal,
        statistics,
        inEquilibrium,
        ultrarelativistic,
        collisionPrefactors,
    ):
        r"""Initialisation

        Parameters
        ----------
        name : string
            A string naming the particle.
        msqVacuum : function
            Function :math:`m^2_0(\phi)`, should take a float and return one.
        msqThermal : function
            Function :math:`m^2_T(T)`, should take a float and return one.
        statistics : {\"Fermion\", \"Boson\"}
            Particle statistics.
        inEquilibrium : boole
            True if particle is treated as in local equilibrium.
        ultrarelativistic : boole
            True if particle is treated as ultrarelativistic.
        collisionPrefactors : list
            Coefficients of collision integrals, :math:`\sim g^4`, currently
            must be of length 3.

        Returns
        -------
        cls : Particle
            An object of the Particle class.
        """
        Particle.__validateInput(
            name,
            msqVacuum,
            msqThermal,
            statistics,
            inEquilibrium,
            ultrarelativistic,
            collisionPrefactors,
        )
        self.name = name
        self.msqVacuum = msqVacuum
        self.msqThermal = msqThermal
        self.statistics = statistics
        self.inEquilibrium = inEquilibrium
        self.ultrarelativistic = ultrarelativistic
        self.collisionPrefactors = collisionPrefactors

    @staticmethod
    def __validateInput(
        name,
        msqVacuum,
        msqThermal,
        statistics,
        inEquilibrium,
        ultrarelativistic,
        collisionPrefactors,
    ):
        """
        Checks input fits expectations
        """
        #fields = np.array([1, 1])
        #assert isinstance(msqVacuum(fields), float), \
        #    f"msqVacuum({fields}) must return float"
        T = 100
        assert isinstance(msqThermal(T), float), \
            f"msqThermal({T}) must return float"
        if statistics not in Particle.STATISTICS_OPTIONS:
            raise ValueError(
                f"{statistics=} not in {Particle.STATISTICS_OPTIONS}"
            )
        assert isinstance(inEquilibrium, bool), \
            "inEquilibrium must be a bool"
        assert isinstance(ultrarelativistic, bool), \
            "ultrarelativistic must be a bool"
        assert len(collisionPrefactors) == 3, \
            "len(collisionPrefactors) must be 3"

class Model:
    '''
    Class that generates the model given external model file
    can be overrriden by user
    '''
    def __init__(self,mu4D,mus,lams,lamm):
        r"""Initialisation
        """
        self.mus = mus
        self.lams = lams
        self.lamm = lamm

        self.v0 = 246.22
        self.muh = 125.
        self.lamh = self.muh**2/(2*self.v0**2)
        self.muhsq = -self.lamh*self.v0**2
        self.mussq = +self.mus**2-self.lamm*self.v0**2/2
        
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
        self.mu4D = mu4D
        self.mu4Dsq = mu4D*mu4D
        '''
        Z,W,t mass, strong gauge coupling and fermion generations
        '''
        self.MW = 80.379
        self.MZ = 91.1876
        self.Mt = 172.76

        self.g0 = 2*self.MW/self.v0
        self.g1 = self.g0*np.sqrt((self.MZ/self.MW)**2-1)
        self.g2 = self.g0
        self.yt = np.sqrt(1/2)*self.g0*self.Mt/self.MW


        self.musT = (
                +1./6*lamm
                +1./4*lams)
        self.muhT = (
                (
                +1*self.g1**2
                +3*self.g2**2
                +4./3*self.Nc*self.yt**2
                +8*self.lamh)/16
                +self.lamm/24
                )

        '''
        Define dictionary of used parameters
        '''
        self.pars = {
                'muhsq': self.muhsq,
                'mussq': self.mussq,
                'lamh': self.lamh,
                'lams': self.lams,
                'lamm': self.lamm
        }

        print(self.musT)

    def V0(self,X,show_V=False):
        '''
        Tree level effective potential
        X
        '''
        X = np.asanyarray(X)
        h1 = X[...,0]
        s1 = X[...,1]

        V = (
           +1/2*self.muhsq*h1**2
           +1/2*self.mussq*s1**2
           +1/4*self.lamh*h1**4
           +1/4*self.lams*s1**4
           +1/4*self.lamm*(h1*s1)**2)
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
        c = np.array([3/2,3/2,3/2,5/6,5/6])

        '''
        mass matrix
        TODO: numerical determination of scalar masses from V0
        '''
        mh2 = self.muhsq+3*self.lamh*h1**2+self.lamm*s1**2/2
        ms2 = self.mussq+3*self.lams*s1**2+self.lamm*h1**2/2
        mhs2 = self.lamm*h1*s1
        sqrt = np.sqrt((mh2-ms2)**2+4*mhs2**2)
        m1 = (mh2+ms2)/2+sqrt/2
        m2 = (mh2+ms2)/2-sqrt/2
        mChi = self.muhsq+self.lamh*h1**2+self.lamm*s1**2/2
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
        print("debug")
        print(X)
        return
        X = np.asanyarray(X, dtype=float)
        print(X)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        V = self.V0(X)
        V += self.V1(bosons, fermions)
        V += self.V1T(bosons, fermions, T, include_radiation)
        return np.real(V)


xsm = Model(1,217.,1.,1.2)



class FreeEnergy:
    def __init__(
        self,
        f,
        Tc,
        Tnucl,
        dfdT=None,
        dfdPhi=None,
        dPhi=1e-3,
        dT=1e-3,
        params=None,
    ):
        r"""Initialisation

        Initialisation for FreeEnergy class from potential.

        Parameters
        ----------
        f : function
            Free energy density function :math:`f(\phi, T)`.
        Tc : float
            Value of the critical temperature, to be defined by the user
        Tnucl : float
            Value of the nucleation temperature, to be defined by the user
        dfdT : function
            Derivative of free energy density function with respect to
            temperature.
        dfdPhi : function
            Derivative of free energy density function with respect to
            field values. Should return a vector in the space of scalar fields.
        dPhi : float, optional
            Small value with which to take numerical derivatives with respect
            to the field.
        dT : float, optional
            Small value with which to take numerical derivatives with respect
            to the temperature.
        params : dict, optional
            Additional fixed arguments to be passed to f as kwargs. Default is
            None.

        Returns
        -------
        cls : FreeEnergy
            An object of the FreeEnergy class.
        """
        if params is None:
            self.f = f
            self.dfdT = dfdT
            self.dfdPhi = dfdPhi
        else:
            self.f = lambda v, T: f(v, T, **params)
            if dfdT is None:
                self.dfdT = None
            else:
                self.dfdT = lambda v, T: dfdT(v, T, **params)
            if dfdPhi is None:
                self.dfdPhi = None
            else:
                self.dfdPhi = lambda v, T: dfdPhi(v, T, **params)
        self.Tc = Tc
        self.Tnucl = Tnucl
        self.dPhi = dPhi
        self.dT = dPhi
        self.params = params # Would not normally be stored. Here temporarily.

    def __call__(self, X, T):
        """
        The effective potential.

        Parameters
        ----------
        X : array of floats
            the field values (here: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        f : float
            The free energy density at this field value and temperature.

        """
        return self.f(X, T)

    def derivT(self, X, T):
        """
        The temperature-derivative of the effective potential.

        Parameters
        ----------
        X : array of floats
            the field values (here: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        dfdT : float
            The temperature derivative of the free energy density at this field
            value and temperature.
        """
        if self.dfdT is not None:
            return self.dfdT(X, T)
        else:
            return derivative(
                lambda T: self(X, T),
                T,
                dx=self.dT,
                n=1,
                order=4,
            )

    def derivField(self, X, T):

        """
        The field-derivative of the effective potential.

        Parameters
        ----------
        X : array of floats
            the field values (here: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        dfdX : array_like
            The field derivative of the free energy density at this field
            value and temperature.
        """
        if self.dfdPhi is not None:
            return self.dfdPhi(X, T)
        else:
            X = np.asanyarray(X)
            # this needs generalising to arbitrary fields
            h, s = X[..., 0], X[..., 1]
            Xdh = X.copy()
            Xdh[..., 0] += self.dPhi * np.ones_like(h)
            Xds = X.copy()
            Xds[..., 1] += self.dPhi * np.ones_like(h)

            dfdh = (self(Xdh, T) - self(X, T)) / self.dPhi
            dfds = (self(Xds, T) - self(X, T)) / self.dPhi

            return_val = np.empty_like(X)
            return_val[..., 0] = dfdh
            return_val[..., 1] = dfds

            return return_val


    def pressureLowT(self,T):
        """
        Returns the value of the pressure as a function of temperature in the low-T phase
        """
        return -self(self.findPhases(T)[0],T)

    def pressureHighT(self,T):
        """
        The pressure in the high-temperature (singlet) phase

        Parameters
        ----------
        T : float
            The temperature for which to find the pressure.

        """
        return -self(self.findPhases(T)[1],T)

   
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
        p = self.params

        muh2_abs = abs(p["muhsq"]) / 20
        mus2_abs = abs(p["mussq"]) / 20
        lambdaH_sqrt = np.sqrt(p["lamh"])
        lambdaS_sqrt = np.sqrt(p["lams"])
        muhT_squared = p["th"] * T**2
        musT_squared = p["ts"] * T**2

        mhsq = np.sqrt(-min(-muh2_abs, p["muhsq"] + muhT_squared) / lambdaH_sqrt)
        mssq = np.sqrt(-min(-mus2_abs, p["mussq"] + musT_squared) / lambdaS_sqrt)

        # mssq = (-p["ts"]*T**2+p["mussq"])/p["lams"]
        # mhsq = (-p["th"]*T**2+p["muhsq"])/p["lamh"]

        # This should be overridden.
        #return np.array([[m1, 0], [0, m2]])
        return [[mhsq, 0], [0, mssq]]


    def findPhases(self, T, X=None):
        """Finds all phases at a given temperature T

        Parameters
        ----------
        T : float
            The temperature for which to find the phases.

        Returns
        -------
        phases : array_like
            A list of phases

        """
        if X is None:
            X = self.approxZeroTMin(T)
            X = np.asanyarray(X)
        #print(X)
            
        p = self.params
        #print(self.f(X,T))

        fh = lambda h: self.f([abs(h),0],T)
        fs = lambda s: self.f([0,abs(s)],T)
        
        vT,wT = X[0,0],X[1,1]
        if fh(vT) < fh(0) and fh(3*vT) > fh(vT):
            vT = optimize.minimize_scalar(fh,(0,vT,3*vT)).x
        else:
            vT = optimize.minimize_scalar(fh,(vT,2*vT),(0,10*vT)).x
        if fs(wT) < fs(0) and fs(3*wT) > fs(wT):
            wT = optimize.minimize_scalar(fs,(0,wT,3*wT)).x
        else:
            wT = optimize.minimize_scalar(fs,(wT,2*wT),(0,10*wT)).x
        return np.array([[vT,0],[0,wT]])

        # result = optimize.minimize(self.Vtot, X, args=(T,),
        #                            method='Nelder-Mead',
        #                            tol=1e-12,options={'disp': False})

        # #hardcoded!
        # p = self.params
        # hsq = (-p["th"]*T**2+p["muhsq"])/p["lamh"]
        # ssq = (-p["ts"]*T**2+p["mussq"])/p["lams"]
        # return np.array([[np.sqrt(hsq),0],[0,np.sqrt(ssq)]])

    def findPhasesAnalytical(self, T):
        """Finds all phases at a given temperature T

        Parameters
        ----------
        T : float
            The temperature for which to find the phases.

        Returns
        -------
        phases : array_like
            A list of phases

        """
        # hardcoded!
        p = self.params
        hsq = (-p["th"]*T**2+p["muhsq"])/p["lamh"]
        ssq = (-p["ts"]*T**2+p["mussq"])/p["lams"]
        return np.array([[np.sqrt(hsq),0],[0,np.sqrt(ssq)]])
