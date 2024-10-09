import configparser
import io
from dataclasses import dataclass

@dataclass
class ConfigGrid:
    """ Holds the config of the Grid3Scales class. """

    spatialGridSize: int = 40
    """ Number of grid points in the spatial direction (M in 2204.13120). """

    momentumGridSize: int = 11
    """ Number of grid points in the momentum directions (N in 2204.13120). """

    ratioPointsWall: float = 0.5
    """
    Fraction of points inside the wall defined by the interval
    [-wallThickness+wallCenter, wallThickness+wallCenter]. Should be a number between 0
    and 1.
    """

    smoothing: float = 0.1
    """ Smoothing factor of the mapping function (the larger the smoother). """

@dataclass
class ConfigEOM:
    """ Holds the config of the EOM class. """

    errTol: float = 1e-3
    """ The absolute error tolerance for the wall velocity result. """

    pressRelErrTol: float = 0.1
    """ Relative error tolerance for the pressure. """

    maxIterations: int = 20
    """ Maximum number of iterations for the convergence of the pressure. """

    conserveEnergyMomentum: bool = True
    r"""
    Flag to enforce conservation of energy and momentum. Normally, this should be set to
    True, but it can help with numerical stability to set it to False. If True, there is
    an ambiguity in the separation between :math:`f_{eq}` and :math:`\delta f` when the
    out-of-equilibrium particles form a closed  system (or nearly closed). This can lead
    to a divergence of the iterative loop. In the end, it is better to set this to False
    if most of the degrees of freedom are treated as out-of-equilibrium particle. If
    most of the dofs are in the background fluid, setting it to True will give better
    results.
    """

    wallThicknessLowerBound: float = 0.1
    """ Lower bound on wall thickness (in units of 1/Tnucl). """

    wallThicknessUpperBound: float = 100.0
    """ Upper bound on wall thickness (in units of 1/Tnucl). """

    wallOffsetLowerBound: float = -10.0
    """ Lower bound on wall offset. """

    wallOffsetUpperBound: float = 10.0
    """ Upper bound on wall offset. """

    ## The following parameters are only used for detonation solutions ##
    vwMaxDeton: float = 0.99
    """ Maximal Velocity at which the solver will look to find a detonation solution """

    nbrPointsMinDeton: int = 5
    """ Minimal number of points probed to bracket the detonation roots. """

    nbrPointsMaxDeton: int = 20
    """ Maximal number of points probed to bracket the detonation roots. """

    overshootProbDeton: float = 0.05
    """
    Desired probability of overshooting a root. Must be between 0 and 1. A smaller value
    will lead to more pressure evaluations (and thus a longer time), but is less likely
    to miss a root.
    """

@dataclass
class ConfigEffectivePotential:
    """ Holds the config of the EffectivePotential class. """

    potentialError: float = 1e-8
    """
    Relative accuracy at which the potential can be evaluated. If the potential is
    polynomial, should be set at the machine precision ~1e-15.
    """

    phaseTracerTol: float = 1e-6
    """
    Desired accuracy of the phase tracer and the resulting FreeEnergy interpolation.
    """

@dataclass
class ConfigHydrodynamics:
    """ Holds the config of the Hydrodynamics class. """
    
    tmin: float = 0.01
    """ Minimum temperature that is probed in Hydrodynamics (in units of Tnucl). """
    
    tmax: float = 10.0
    """ Maximum temperature that is probed in Hydrodynamics (in units of Tnucl). """
    
    relativeTol: float = 1e-6
    """ Relative tolerance used in Hydrodynamics. """
    
    absolutionTol: float = 1e-10
    """ Absolute tolerance used in Hydrodynamics. """

@dataclass
class ConfigThermodynamics:
    """ Holds the config of the Hydrodynamics class. """
    
    tmin: float = 0.8
    """
    Minimum temperature used in the phase tracing (in units of the estimate for the
    minimum temperature obtained in the template model). 
    """
    
    tmax: float = 1.2
    """
    Maximum temperature used in the phase tracing (in units of the estimate for the
    maximum temperature obtained in the template model). 
    """

@dataclass
class ConfigBoltzmannSolver:
    """ Holds the config of the BoltzmannSolver class. """
    
    basisM: str = 'Cardinal'
    """ The position polynomial basis type, either 'Cardinal' or 'Chebyshev'. """
    
    basisN: str = 'Chebyshev'
    """ The momentum polynomial basis type, either 'Cardinal' or 'Chebyshev'. """
    
    collisionMultiplier: float = 1.0
    """
    Factor multiplying the collision term in the Boltzmann equation. Can be used for
    testing or for studying the solution's sensibility to the collision integrals. Don't
    forget to adjust meanFreePathScale accordingly if this is different from 1
    (meanFreePathScale should scale like 1/collisionMultiplier).
    WARNING: THIS CHANGES THE COLLISION TERMS WRT TO THEIR PHYSICAL VALUE.
    """

@dataclass
class Config:
    """
    Data class that holds all the model-independent configs.
    It contains objects of the data classes ConfigGrid, ConfigEOM,
    ConfigEffectivePotential, ConfigHydrodynamics, ConfigThermodynamics and
    ConfigBoltzmannSolver.
    It can also load the configs from an .ini file.
    """

    configGrid: ConfigGrid = ConfigGrid()
    """ Holds the config of the Grid3Scales class. """

    configEOM: ConfigEOM = ConfigEOM()
    """ Holds the config of the EOM class. """

    configEffectivePotential: ConfigEffectivePotential = ConfigEffectivePotential()
    """ Holds the config of the EffectivePotential class. """

    configHydrodynamics: ConfigHydrodynamics = ConfigHydrodynamics()
    """ Holds the config of the Hydrodynamics class. """

    configThermodynamics: ConfigThermodynamics = ConfigThermodynamics()
    """ Holds the config of the Thermodynamics class. """

    configBoltzmannSolver: ConfigBoltzmannSolver = ConfigBoltzmannSolver()
    """ Holds the config of the BoltzmannSolver class. """

    def loadConfigFromFile(self, filePath: str) -> None:
        """
        Load the configs from a file.

        Parameters
        ----------
        filePath : str
            Path of the file where the configs are.

        """

class ConfigParser:
    """class Config -- Manages configuration variables for WallGo. This is essentially a wrapper around ConfigParser.
    Accessing variables works as with ConfigParser: 
    config.get("Section", "someKey")
    """

    configParser: configparser.ConfigParser

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.optionxform = str ## preserve case 


    def readINI(self, filePath: str):
        self.config.read(filePath)


    def get(self, section: str, key: str) -> any:
        return self.config.get(section, key)
    
    def getint(self, section: str, key: str) -> int:
        return self.config.getint(section, key)
    
    def getfloat(self, section: str, key: str) -> float:
        return self.config.getfloat(section, key)
    
    def set(self, section: str, key: str, value: str):
        self.config.set(section, key, value)
    
    def getboolean(self, section: str, key: str) -> bool:
        return self.config.getboolean(section, key)
    
    def __str__(self) -> str:
        """Print the content in similar format as an .ini file would be.
        """

        ## Do magic by writing the contents into a file-like string buffer
        buffer = io.StringIO()
        self.config.write(buffer)
        
        return buffer.getvalue()