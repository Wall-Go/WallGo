""" Dataclasses to store the configs """

import os
import configparser
from dataclasses import dataclass, field

@dataclass
class ConfigGrid:
    """ Holds the config of the Grid3Scales class. """

    spatialGridSize: int = 40
    """ Number of grid points in the spatial direction (M in 2204.13120). """

    momentumGridSize: int = 11
    """
    Number of grid points in the momentum directions (N in 2204.13120).
    MUST BE ODD.
    """

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

    wallThicknessBounds: list[float] = field(default_factory=lambda: [0.1, 100.0])
    """ Lower and upper bounds on wall thickness (in units of 1/Tnucl). """

    wallOffsetBounds: list[float] = field(default_factory=lambda: [-10.0, 10.0])
    """ Lower and upper bounds on wall offset. """

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
class ConfigHydrodynamics:
    """ Holds the config of the Hydrodynamics class. """

    tmin: float = 0.01
    """ Minimum temperature that is probed in Hydrodynamics (in units of Tnucl). """

    tmax: float = 10.0
    """ Maximum temperature that is probed in Hydrodynamics (in units of Tnucl). """

    relativeTol: float = 1e-6
    """ Relative tolerance used in Hydrodynamics. """

    absoluteTol: float = 1e-10
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

    phaseTracerTol: float = 1e-6
    """
    Desired accuracy of the phase tracer and the resulting FreeEnergy interpolation.
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

    configGrid: ConfigGrid = field(default_factory=lambda: ConfigGrid())
    """ Holds the config of the Grid3Scales class. """

    configEOM: ConfigEOM = field(default_factory=lambda: ConfigEOM())
    """ Holds the config of the EOM class. """

    configHydrodynamics: ConfigHydrodynamics = field(
        default_factory=lambda: ConfigHydrodynamics())
    """ Holds the config of the Hydrodynamics class. """

    configThermodynamics: ConfigThermodynamics = field(
        default_factory=lambda: ConfigThermodynamics())
    """ Holds the config of the Thermodynamics class. """

    configBoltzmannSolver: ConfigBoltzmannSolver = field(
        default_factory=lambda: ConfigBoltzmannSolver())
    """ Holds the config of the BoltzmannSolver class. """

    def loadConfigFromFile(self, filePath: str) -> None:
        """
        Load the configs from a file.

        Parameters
        ----------
        filePath : str
            Path of the file where the configs are.

        """
        # Make sure that the file exists
        if not os.path.isfile(filePath):
            raise FileNotFoundError(filePath)

        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(filePath)

        # Read the Grid configs
        if 'Grid' in parser.sections():
            keys = parser['Grid'].keys()
            if 'spatialGridSize' in keys:
                self.configGrid.spatialGridSize = parser.getint("Grid",
                                                                "spatialGridSize")
            if 'momentumGridSize' in keys:
                self.configGrid.momentumGridSize = parser.getint("Grid",
                                                                "momentumGridSize")
            if 'ratioPointsWall' in keys:
                self.configGrid.ratioPointsWall = parser.getfloat("Grid",
                                                                "ratioPointsWall")
            if 'smoothing' in keys:
                self.configGrid.smoothing = parser.getfloat("Grid", "smoothing")

        # Read the EOM configs
        if 'EquationOfMotion' in parser.sections():
            keys = parser['EquationOfMotion'].keys()
            if 'errTol' in keys:
                self.configEOM.errTol = parser.getfloat("EquationOfMotion", "errTol")
            if 'pressRelErrTol' in keys:
                self.configEOM.pressRelErrTol = parser.getfloat("EquationOfMotion",
                                                                "pressRelErrTol")
            if 'maxIterations' in keys:
                self.configEOM.maxIterations = parser.getint("EquationOfMotion",
                                                             "maxIterations")
            if 'conserveEnergyMomentum' in keys:
                self.configEOM.conserveEnergyMomentum = parser.getboolean(
                    "EquationOfMotion",
                    "conserveEnergyMomentum")
            if 'wallThicknessLowerBound' in keys:
                self.configEOM.wallThicknessBounds[0] = parser.getfloat(
                    "EquationOfMotion",
                    "wallThicknessLowerBound")
            if 'wallThicknessUpperBound' in keys:
                self.configEOM.wallThicknessBounds[1] = parser.getfloat(
                    "EquationOfMotion",
                    "wallThicknessUpperBound")
            if 'wallOffsetLowerBound' in keys:
                self.configEOM.wallOffsetBounds[0] = parser.getfloat(
                    "EquationOfMotion",
                    "wallOffsetLowerBound")
            if 'wallOffsetUpperBound' in keys:
                self.configEOM.wallOffsetBounds[1] = parser.getfloat(
                    "EquationOfMotion",
                    "wallOffsetUpperBound")
            if 'vwMaxDeton' in keys:
                self.configEOM.vwMaxDeton = parser.getfloat("EquationOfMotion",
                                                            "vwMaxDeton")
            if 'nbrPointsMinDeton' in keys:
                self.configEOM.nbrPointsMinDeton = parser.getint("EquationOfMotion",
                                                                 "nbrPointsMinDeton")
            if 'nbrPointsMaxDeton' in keys:
                self.configEOM.nbrPointsMaxDeton = parser.getint("EquationOfMotion",
                                                                 "nbrPointsMaxDeton")
            if 'overshootProbDeton' in keys:
                self.configEOM.overshootProbDeton = parser.getfloat("EquationOfMotion",
                                                                "overshootProbDeton")

        # Read the Hydrodynamics configs
        if 'Hydrodynamics' in parser.sections():
            keys = parser['Hydrodynamics'].keys()
            if 'tmin' in keys:
                self.configHydrodynamics.tmin = parser.getfloat("Hydrodynamics", "tmin")
            if 'tmax' in keys:
                self.configHydrodynamics.tmax = parser.getfloat("Hydrodynamics", "tmax")
            if 'relativeTol' in keys:
                self.configHydrodynamics.relativeTol = parser.getfloat("Hydrodynamics",
                                                                       "relativeTol")
            if 'absoluteTol' in keys:
                self.configHydrodynamics.absoluteTol = parser.getfloat("Hydrodynamics",
                                                                       "absoluteTol")

        # Read the Thermodynamics configs
        if 'Thermodynamics' in parser.sections():
            keys = parser['Thermodynamics'].keys()
            if 'tmin' in keys:
                self.configThermodynamics.tmin = parser.getfloat("Thermodynamics",
                                                                  "tmin")
            if 'tmax' in keys:
                self.configThermodynamics.tmax = parser.getfloat("Thermodynamics",
                                                                 "tmax")
            if 'phaseTracerTol' in keys:
                self.configThermodynamics.phaseTracerTol = parser.getfloat(
                    "Thermodynamics",
                    "phaseTracerTol")

        # Read the BoltzmannSolver configs
        if 'BoltzmannSolver' in parser.sections():
            keys = parser['BoltzmannSolver'].keys()
            if 'collisionMultiplier' in keys:
                self.configBoltzmannSolver.collisionMultiplier = parser.getfloat(
                    "BoltzmannSolver",
                    "collisionMultiplier")
            if 'basisM' in keys:
                self.configBoltzmannSolver.basisM = parser.get("BoltzmannSolver",
                                                               "basisM")
            if 'basisN' in keys:
                self.configBoltzmannSolver.basisN = parser.get("BoltzmannSolver",
                                                               "basisN")
