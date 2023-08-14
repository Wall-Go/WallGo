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
        msqVaccum,
        msqThermal,
        statistics,
        collisionPrefactors,
    ):
        r"""Initialisation

        Parameters
        ----------
        msqVaccum : function
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
            msqVaccum, msqThermal, statistics, collisionPrefactors,
        )
        self.msqVacuum = msqVaccum
        self.msqThermal = msqThermal
        self.statistics = statistics
        self.collisionPrefactors = collisionPrefactors

    @staticmethod
    def __validateInput(msqVaccum, msqThermal, statistics, collisionPrefactors):
        """
        Checks input fits expectations
        """
        fields = [1, 1]
        assert isinstance(msqVacuum(fields), float), \
            f"msqVacuum({fields}) must return float"
        T = 100
        assert isinstance(msqThermal(T), float), \
            f"msqThermal({T}) must return float"
        if statistics not in ParticleConfig.STATISTICS_OPTIONS:
            raise ValueError(
                f"{statistics=} not in {ParticleConfig.STATISTICS_OPTIONS}"
            )
        assert len(collisionPrefactors) == 3, \
            "len(collisionPrefactors) must be 3"


class FreeEnergy:
    def __init__(
        self,
        f,
        Tnucl,
        phi_eps=1e-3,
        T_eps=1e-3,
    ):
        r"""Initialisation

        Initialisation for FreeEnergy class from potential.

        Parameters
        ----------
        f : function
            Free energy density function :math:`f(\phi, T)`.
        phi_eps : float, optional
            Small value with which to take numerical derivatives with respect
            to the field.
        T_eps : float, optional
            Small value with which to take numerical derivatives with respect
            to the temperature.

        Returns
        -------
        cls : FreeEnergy
            An object of the FreeEnergy class.
        """
        self.f = f
        self.phi_eps = phi_eps
        self.T_eps = phi_eps
        self.Tnucl = 100

        self.v0 = 246.22
        self.muhsq = 7825.
        self.lamh = 0.129074
        self.mussq = 10774.6
        self.lams = 1.
        self.lamm = 1.2

        self.g = 0.652905
        self.gp = 0.349791
        self.yt = 0.992283
        self.th = 1/48.*(9*self.g**2+3*self.gp**2+2*(6*self.yt**2 + 12*self.lamh+ self.lamm))
        self.ts = 1/12.*(2*self.lamm + 3*self.lams)

    def FiniteTPotential(self, X, T):
        """
        The effective potential as a function of the field and temperature.

        For testing purposes it has a hard-coded potential, but this has to be replaced by f,
        which is user-defined.
        """
        X = np.asanyarray(X)
        h,s = X[...,0], X[...,1]

        Vtree = -1/2.*self.muhsq*h**2 + 1/4.*self.lamh*h**4 -1/2.*self.mussq*s**2 + 1/4.*self.lams*s**4 + 1/4.*self.lamm*s**2*h**2 + 1/4.*self.lamh*self.v0**4

        VT = 1/2.*(self.th*h**2 + self.ts*s**2)*T**2
        
        return Vtree + VT
    
    def pressureHighT(self,T):
        """
        Returns the value of the pressure as a function of temperature in the high-T phase
        """
        ssq = (-self.ts*T**2+self.mussq)/self.lams
        return -self.FiniteTPotential(np.array([0,np.sqrt(ssq)]),T)

    def pressureLowT(self,T):
        """
        Returns the value of the pressure as a function of temperature in the low-T phase        
        """
        hsq = (-self.th*T**2+self.muhsq)/self.lamh
        return -self.FiniteTPotential(np.array([np.sqrt(hsq),0]),T)

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
        return np.array([[0, 0], [1.0, 1.7]])
