

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
        msqDerivative,
        msqThermal,
        statistics,
        inEquilibrium,
        ultrarelativistic,
        totalDOFs
    ):
        r"""Initialisation

        Parameters
        ----------
        name : string
            A string naming the particle.
        msqVacuum : function
            Function :math:`m^2_0(\phi)`, should take a Fields object and 
            return a float or array.
        msqDerivative : function
            Function :math:`d(m_0^2)/d(\phi)`, should take a Fields object and 
            return an array.
        msqThermal : function
            Function :math:`m^2_T(T)`, should take a float and return one.
        statistics : {\"Fermion\", \"Boson\"}
            Particle statistics.
        inEquilibrium : bool
            True if particle is treated as in local equilibrium.
        ultrarelativistic : bool
            True if particle is treated as ultrarelativistic.
        totalDOFs : int
            Total number of degrees of freedom (should include the multiplicity 
            factor).
        

        Returns
        -------
        cls : Particle
            An object of the Particle class.
        """
        Particle.__validateInput(
            name,
            msqVacuum,
            msqDerivative,
            msqThermal,
            statistics,
            inEquilibrium,
            ultrarelativistic,
            totalDOFs
        )
        self.name = name
        self.msqVacuum = msqVacuum
        self.msqDerivative = msqDerivative
        self.msqThermal = msqThermal
        self.statistics = statistics
        self.inEquilibrium = inEquilibrium
        self.ultrarelativistic = ultrarelativistic
        self.totalDOFs = totalDOFs

    @staticmethod
    def __validateInput(
        name,
        msqVacuum,
        msqDerivative,
        msqThermal,
        statistics,
        inEquilibrium,
        ultrarelativistic,
        totalDOFs
    ):
        """
        Checks that the input fits expectations
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
        assert isinstance(totalDOFs, int) , \
            "totalDOFs must be an integer"
