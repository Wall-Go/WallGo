"""
Classes for user input of models
"""
import numpy as np # arrays, maths and stuff
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


    def pressureHighT(self,T):
        """
        The pressure in the high-temperature (singlet) phase

        Parameters
        ----------
        T : float
            The temperature for which to find the pressure.

        """
        return -self(self.findPhases(T)[0],T)

    def pressureLowT(self,T):
        """
        Returns the value of the pressure as a function of temperature in the low-T phase
        """
        return -self(self.findPhases(T)[1],T)

    def findPhases(self, T):
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
        ssq = (-p["ts"]*T**2+p["mussq"])/p["lams"]
        hsq = (-p["th"]*T**2+p["muhsq"])/p["lamh"]
        return np.array([[0,np.sqrt(ssq)],[np.sqrt(hsq),0]])
