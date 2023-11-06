"""
Classes for user input of models
"""
import numpy as np # arrays, maths and stuff
import math
from scipy import integrate,interpolate,optimize,special,linalg,stats
from .helpers import derivative # derivatives for callable functions
from cosmoTransitions.finiteT import Jb_spline as Jb
from cosmoTransitions.finiteT import Jf_spline as Jf


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
        inEquilibrium : bool
            True if particle is treated as in local equilibrium.
        ultrarelativistic : bool
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
    """
    Class that generates the model given external model file
    can be overrriden by user
    """
    def __init__(self,
        mu4D,mus,lams,lamm,
        use_EFT=False,
        ):
        r"""Initialisation

        Parameters
        ----------
        mu4D : float 
            4d renormalisaton scale.
        use_EFT : bool
            True if 3d EFT is used.

        Returns
        -------
        cls : Model 
            An object of the Model class.
        """
        self.use_EFT=use_EFT

        self.mus = mus
        self.lams = lams
        self.lamm = lamm

        self.v0 = 246.
        self.muh = 125.
        self.lamh = self.muh**2/(2*self.v0**2)
        self.muhsq = -self.lamh*self.v0**2
        self.mussq = +self.mus**2-self.lamm*self.v0**2/2

        self.musq = [self.muhsq, self.mussq]

        """
        Number of bosonic and fermionic dofs
        """
        self.num_boson_dof = 29
        self.num_fermion_dof = 90
        """
        Number of fermion generations and colors
        """
        self.nf = 3
        self.Nc = 3
        """
        4D RG scale of EFT as fraction of temperature
        """
        self.mu4D = mu4D
        self.mu4Dsq = mu4D*mu4D
        """
        Z,W,t mass, strong gauge coupling and fermion generations
        """
        self.MW = 80.379
        self.MZ = 91.1876
        self.Mt = 173.

        self.g0 = 2*self.MW/self.v0
        self.g1 = self.g0*math.sqrt((self.MZ/self.MW)**2-1)
        self.g2 = self.g0
        self.g3 = 1.2279920495357861
        self.yt = math.sqrt(1/2)*self.g0*self.Mt/self.MW

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

        self.musqT = [self.muhT, self.musT]


        """
        Define dictionary of used parameters
        """
        self.params = {
            'muhsq': self.muhsq,
            'mussq': self.mussq,
            'lamh': self.lamh,
            'lams': self.lams,
            'lamm': self.lamm,
            'muhT': self.muhT,
            'musT': self.musT,
            'g1': self.g1,
            'g2': self.g2,
            'g3': self.g3,
            'yt': self.yt
        }


    def V0(self, fields: float, T, show_V=False):
        """
        Tree-level effective potential

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `fields` and `T`
            should be such that ``fields.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``fields[...,0]*T`` is a valid operation).

        Returns
        -------
        V: tree-level effective potential 
        """
        fields = np.asanyarray(fields)

        if self.use_EFT:
            for i in range(len(self.musq)):
                self.musq[0] += T**2*self.musqT[0]
            lamh = self.lamh*T
            lams = self.lams*T
            lamm = self.lamm*T
        else:
            musq = self.musq
            lamh = self.lamh
            lams = self.lams
            lamm = self.lamm

        h1,s1 = fields[0,...],fields[1,...]
        V = (
            +1/2*musq[0]*h1**2
            +1/2*musq[1]*s1**2
            +1/4*lamh*h1**4
            +1/4*lams*s1**4
            +1/4*lamm*(h1*s1)**2)
        if show_V:
            print(V)
        return V

    def Jcw(self, msq, degrees_of_freedom, c):
        """
        Coleman-Weinberg potential

        Parameters
        ----------
        msq : array_like
            A list of the boson particle masses at each input point `fields`.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c: float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.

        Returns
        -------
        Jcw : float or array_like
            One-loop Coleman-Weinberg potential for given particle spectrum.
        """
        return degrees_of_freedom*msq*msq * (np.log(np.abs(msq/self.mu4Dsq) + 1e-100) - c)

    def boson_massSq(self, fields, T):
        """
        Calculate the boson particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `fields` and `T`
            should be such that ``fields.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``fields[0,...]*T`` is a valid operation).

        Returns
        -------
        massSq : array_like
            A list of the boson particle masses at each input point `fields`. The
            shape should be such that
            ``massSq.shape == (fields[...,0]*T).shape + (Nbosons,)``.
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
        fields = np.asanyarray(fields)
        h1,s1 = fields[0,...],fields[1,...]

        Nbosons = 5
        degrees_of_freedom = np.array([1,1,3,6,3])#h,s,chi,W,Z
        c = np.array([3/2,3/2,3/2,5/6,5/6])

        """
        mass matrix
        TODO: numerical determination of scalar masses from V0
        """
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
        return massSq, degrees_of_freedom, c

    def fermion_massSq(self, fields):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `fields`. The
            shape should be such that  ``massSq.shape == (fields[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            len
        """
        fields = np.asanyarray(fields)
        h1,s1 = fields[0,...],fields[1,...]

        Nfermions = 1
        degrees_of_freedom = np.array([12])
        mt = self.yt**2*h1**2/2
        # todo include spins for each particle

        massSq = np.column_stack((mt,))
        return massSq, degrees_of_freedom

    def V1(self, bosons, fermions):
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
        V = np.sum(self.Jcw(m2,nb,c), axis=-1)

        m2, nf = fermions
        c = 1.5
        V -= np.sum(self.Jcw(m2,nf,c), axis=-1)

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

        V = 0;
        if self.num_boson_dof is not None:
            nb = self.num_boson_dof - np.sum(nb)
            V -= nb * np.pi**4 / 45.
        if self.num_fermion_dof is not None:
            nf = self.num_fermion_dof - np.sum(nf)
            V -= nf * 7*np.pi**4 / 360.

        return V*T4/(2*np.pi*np.pi)

    def V1T(self, bosons, fermions, T):
        """
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

        Parameters
        ----------

        Returns
        -------
        V1T : 4d 1loop thermal potential 
        """ 
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        m2,nb,_ = bosons
        V = np.sum(nb*Jb(m2/T2), axis=-1)
        m2,nf = fermions
        V += np.sum(nf*Jf(m2/T2), axis=-1)
        return V*T4/(2*np.pi*np.pi)

    def Vtot(self, fields, T, include_radiation=True):
        """
        The total finite temperature effective potential.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `fields` and `T`
            should be such that ``fields.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``fields[0,...]*T`` is a valid operation).
        include_radiation : bool, optional
            If False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.

        Returns
        -------
        Vtot : total effective potential
        """
        T = np.asanyarray(T)
        fields = np.asanyarray(fields)
        if self.use_EFT:
            fields = fields/np.sqrt(T + 1e-100)
        bosons = self.boson_massSq(fields,T)
        fermions = self.fermion_massSq(fields)
        V = self.V0(fields, T)
        if self.use_EFT:
            V *= T
        else:
            V += self.V1(bosons, fermions)
            V += self.V1T(bosons, fermions, T)
        if include_radiation:
            V += self.PressureLO(bosons, fermions, T)
        return np.real(V)

class FreeEnergy:
    def __init__(
        self,
        f,
        Tc=None,
        Tnucl=None,
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
        """
        Number of dynamics scalars
        """
        self.nbrFields = 2
        """
        self.transField`i' take values between [0,self.nbrFields-1] indicating 
        which of the fields undergo the transition.
        self.transField0 is the field assoiciated to the low temperature phase 
        self.transField1 is the field assoiciated to the high temperature phase 
        """
        self.transField0 = 0
        self.transField1 = 1

        if params is None:
            self.f = f
            self.dfdT = dfdT
            self.dfdPhi = dfdPhi
        else:
            self.f = lambda fields, T: f(fields, T)
            if dfdT is None:
                self.dfdT = None
            else:
                self.dfdT = lambda v, T: dfdT(v, T, **params)
            if dfdPhi is None:
                self.dfdPhi = None
            else:
                self.dfdPhi = lambda v, T: dfdPhi(v, T, **params)

        self.dPhi = dPhi
        self.dT = dPhi
        self.params = params # Would not normally be stored. Here temporarily.

        self.Ti_int = None
        self.Tf_int = None

        if Tc is None:
            print("No critical temperature defined")
            self.findTc()
            print("Found Tc=", self.Tc)
            # raise ValueError("No critical temperature defined")
        else:
            self.Tc = Tc
        if Tnucl is None:
            raise ValueError("No nucleation temperature defined")
        else:
            self.Tnucl = Tnucl
        FreeEnergy.__validateInput(self.Tc, Tnucl)


    @staticmethod
    def __validateInput(
        Tc,
        Tn
    ):
        """
        Checks input fits expectations
        """

        assert Tc > Tn, \
            f"Tc must be larger than Tn"

    def __call__(self, fields, T):
        """
        The effective potential.

        Parameters
        ----------
        fields : array of floats
            the field values (here: Higgs, singlet)
        T : float
            the temperature

        Returns
        ----------
        f : float
            The free energy density at this field value and temperature.

        """
        return self.f(fields, T)

    def derivT(self, fields, T):
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
        if self.dfdT is not None:
            return self.dfdT(fields, T)
        else:
            return derivative(
                lambda T: self(fields, T),
                T,
                dx=self.dT,
                n=1,
                order=4,
            )

    def derivField(self, fields, T):
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
        dfdfields : array_like
            The field derivative of the free energy density at this field
            value and temperature.
        """
        if self.dfdPhi is not None:
            return self.dfdPhi(fields, T)
        else:
            fields = np.asanyarray(fields, dtype=float)
            return_val = np.empty_like(fields)
            for i in range(self.nbrFields):
                field = fields[i,...] 
                dfields_dfield = fields.copy()
                dfields_dfield[i,...] += self.dPhi * np.ones_like(field)
                dfd_field = (self(dfields_dfield,T) - self(fields,T)) / self.dPhi
                return_val[i,...] = dfd_field

            return return_val


    def pressureLowT(self,T):
        """
        Returns the value of the pressure as a function of temperature in the low-T phase
        indicated by self.transfield0 (e.g. Higgs)

        Parameters
        ----------
        T : float
            The temperature for which to find the pressure.

        Returns
        ----------
        pressureLowT : float
           pressure in the low-temperature phase 
        """
        return -self(self.findPhases(T),T)[0]

    def pressureHighT(self,T):
        """
        The pressure in the high-temperature phase
        indicated by self.transfield1 (e.g. Singlet)

        Parameters
        ----------
        T : float
            The temperature for which to find the pressure.

        Returns
        ----------
        pressureHighT : float
           pressure in the high-temperature phase 
        """
        return -self(self.findPhases(T),T)[1]

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
        lamh_sqrt = np.sqrt(p["lamh"])
        lams_sqrt = np.sqrt(p["lams"])
        muhT_squared = p["muhT"] * T**2
        musT_squared = p["musT"] * T**2

        mhsq = np.sqrt(-min(-muh2_abs, p["muhsq"] + muhT_squared) / lamh_sqrt)
        mssq = np.sqrt(-min(-mus2_abs, p["mussq"] + musT_squared) / lams_sqrt)

        return [[mhsq, 0], [0, mssq]]

    def interpolateMinima(self, Ti, Tf, dT):
        """Interpolates the minima of all phases for a given termperature range

        Parameters
        ----------
        Ti : float
            Lower limit of the temperature bracket

        Tf : float
            Upper limit of the temperature bracket

        dT : float
            Increment during the interpolation

        Returns
        -------
        self.fieldsInt : array_like, univariate splines
                scipy.univariate splines has the advantage of derivative() method
        """
        if Tf < Ti:
            raise ValueError("Interpolation range not well defined: Tf below Ti")

        min_nodes = 10
        n_nodes = max(min_nodes, int(np.ceil((Tf-Ti)/dT)))
        Trange = np.linspace(Ti,Tf,n_nodes)

        vmin = [[]]*2
        for T in Trange:
            mins = self.findPhases(T)
            vmin=np.append(vmin, np.diag(mins)[:, np.newaxis], axis=1)
        self.Ti_int = Ti
        self.Tf_int = Tf
        self.fieldsInt = [
            interpolate.UnivariateSpline(Trange,vmin[i,...],s=0)
            for i in range(2)] 

    def findPhases(self, T, fields=None):
        """
        Tracks the two phases indicated at init by
        self.transField0 and
        self.transField1
        at a given temperature T

        Parameters
        ----------
        T : float
            The temperature for which to find the phases.

        Returns
        -------
        findPhases : array_like
            A list of two phases
        """
        if T is None:
            raise TypeError('Temperature is None')

        if self.Ti_int is not None and self.Ti_int <= T <= self.Tf_int:
            vmin = [self.fieldsInt[i](T) for i in range(2)]

        else:
            if fields is None:
                fields = self.approxZeroTMin(T)
                fields = np.asanyarray(fields)

            vmin = []
            for i in (self.transField0,self.transField1):
                fPhase = lambda v: self.f([[abs(v)] if j == i else [0] for j in range(len(fields))],T)[0]
                vT=fields[i,i]

                cond1 = fPhase(vT) < fPhase(0)
                cond2 = fPhase(3*vT) > fPhase(vT)
                if cond1 and cond2:
                    vT = optimize.minimize_scalar(fPhase, bracket=(0, vT, 3*vT)).x
                else:
                    vT = optimize.minimize_scalar(fPhase, bracket=(vT, 2*vT), bounds=(0, 10 * vT)).x

                vmin.append(vT)

        return np.array(np.diag(vmin))

    def d2V(self, fields, T):
        """
        Calculates the Hessian (second derivative) matrix for the
        finite-temperature effective potential.

        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """
        # fields = np.asanyarray(fields, dtype=float)
        # T = np.asanyarray(T, dtype=float)
        self.Ndim = 2
        self.x_eps = 1e-3
        self.deriv_order = 4
        # f1 = lambda fields: np.asanyarray(self.f(fields,T))[...,np.newaxis]
        try:
            f1 = self._d2V
        except:
            # Create the gradient function
            self._d2V = helper_functions.hessianFunction(
                self.f, self.x_eps, self.Ndim, self.deriv_order)
            f1 = self._d2V
        # Need to add extra axes to T since extra axes get added to fields in
        # the helper function.

        T = np.asanyarray(T)[...,np.newaxis]
        
        return f1(fields, T)

    def findTc(self,Tmax=150):
        """
        Determines the critical temperature between the two scalar phases
        indicated upon init.
        The critical temperature is determined by tracking the pressure
        of the different phases at their minima and finding the intersection point.

        Parameters
        ----------
        Tmax : float
            Maximal temperature bracket for Tcrit search

        Returns
        -------
        Tc : array_like
            Critical temperature
        """

        deltaf = lambda v0,v1,T: ( 
            +self.f([[v0] if j == self.transField0 else [0] for j in range(self.nbrFields)],T)[0]
            -self.f([[v1] if j == self.transField1 else [0] for j in range(self.nbrFields)],T)[0]
        )

        def deltaPmin(T):
            mins = self.findPhases(T)
            return deltaf(mins[0,0],mins[1,1],T)
        
        self.Tc = optimize.root_scalar(deltaPmin,bracket=(90,150),rtol=1e-12,xtol=1e-12).root
            
        return self.Tc
