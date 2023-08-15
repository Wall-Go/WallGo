"""
Classes for user input of models
"""
from .helpers import derivative # derivatives for callable functions


class Particle:
    """Particle configuration

    A simple class holding attributes of an out-of-equilibrium particle as
    relevant for calculations of Boltzmann equations.
    """
    STATISTICS_OPTIONS = ["Fermion", "Boson"]

    def __init__(
        self,
        msqVacuum,
        msqThermal,
        statistics,
        collisionPrefactors,
    ):
        r"""Initialisation

        Parameters
        ----------
        msqVacuum : function
            Function :math:`m^2_0(\phi)`, should take a float and return one.
        msqThermal : function
            Function :math:`m^2_T(T)`, should take a float and return one.
        statistics : {\"Fermion\", \"Boson\"}
            Particle statistics.
        collisionPrefactors : list
            Coefficients of collision integrals, :math:`\sim g^4`, currently
            must be of length 3.

        Returns
        -------
        cls : Particle
            An object of the Particle class.
        """
        Particle.__validateInput(
            msqVacuum, msqThermal, statistics, collisionPrefactors,
        )
        self.msqVacuum = msqVacuum
        self.msqThermal = msqThermal
        self.statistics = statistics
        self.collisionPrefactors = collisionPrefactors

    @staticmethod
    def __validateInput(msqVacuum, msqThermal, statistics, collisionPrefactors):
        """
        Checks input fits expectations
        """
        fields = [1, 1]
        assert isinstance(msqVacuum(fields), float), \
            f"msqVacuum({fields}) must return float"
        T = 100
        assert isinstance(msqThermal(T), float), \
            f"msqThermal({T}) must return float"
        if statistics not in Particle.STATISTICS_OPTIONS:
            raise ValueError(
                f"{statistics=} not in {Particle.STATISTICS_OPTIONS}"
            )
        assert len(collisionPrefactors) == 3, \
            "len(collisionPrefactors) must be 3"


class FreeEnergy:
    def __init__(
        self,
        f,
        Tnucl,
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
        Tnucl : float
            Value of the nucleation temperature, to be defined by the user
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
        else:
            self.f = lambda v, T: f(v, T, **params)
        self.Tnucl = Tnucl
        self.dPhi = dPhi
        self.dT = dPhi

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
        return f(X, T)

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
        return (self(X, T + self.dT) - self(X, T)) / self.dT

    def derivField(self,X,T):

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
        dfdX : array_like
            The field derivative of the free energy density at this field
            value and temperature.
        """
        X = np.asanyarray(X)
        # this needs generalising to arbitrary fields
        h, s = X[..., 0], X[..., 1]
        Xdh = X.copy()
        Xdh[..., 0] += self.dPhi * np.ones_like(h)
        Xds = X.copy()
        Xds[..., 1] += self.dPhi * np.ones_like(h)

        dfdh = (self(Xdh, T) - self(X, T)) / self.dPhi
        dfds = (self(Xds, T) - self(X, T)) / self.dPhi

        return np.array([dfdh, dfds])

    def pressureHighT(self,T):
        """
        The pressure in the high-temperature (singlet) phase

        Parameters
        ----------
        T : float
            The temperature for which to find the pressure.

        For testing purposes the pressure was hard-coded, but it can be obtained from
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
        ssq = (-self.ts*T**2+self.mussq)/self.lams
        hsq = (-self.th*T**2+self.muhsq)/self.lamh
        return np.array([[0,np.sqrt(ssq)],[np.sqrt(hsq),0]])
