"""
Initialisation for PotentialTools module, includes loading of Jb/Jf integral data
"""
import configparser
from .effectivePotentialNoResum import EffectivePotentialNoResum, EImaginaryOption
from .integrals import Integrals, JbIntegral, JfIntegral
from .utils import getSafePathToResource



_bInitialized = False  # pylint: disable=invalid-name
"""Configuration settings, using class from WallGo"""
config = configparser.ConfigParser()
config.optionxform = str

"""Default integral objects for WallGo. Calling WallGo.initialize() optimizes these by
replacing their direct computation with precomputed interpolation tables."""
defaultIntegrals = Integrals()
defaultIntegrals.Jb.disableAdaptiveInterpolation()
defaultIntegrals.Jf.disableAdaptiveInterpolation()


# Define a separate initializer function that does NOT get called automatically.
# This is good for preventing heavy startup operations from running if the user just
# wants a one part of WallGo and not the full framework, eg. `import WallGo.Integrals`.
# Downside is that programs need to manually call this, preferably as early as possible.
def _initializeInternal() -> None:
    """
    WallGo initializer. This should be called as early as possible in your program.
    """

    global _bInitialized  # pylint: disable=invalid-name
    global config  # pylint: disable=invalid-name

    if not _bInitialized:

        ## read default config
        config.read(getSafePathToResource("Config/PotentialToolsDefaults.ini"))

        # print(config)

        ## Initialize interpolations for our default integrals
        _initalizeIntegralInterpolations()

        _bInitialized = True

    else:
        raise RuntimeWarning("Warning: Repeated call to PotentialTools._initializeInternal()")


def _initalizeIntegralInterpolations() -> None:  # pylint: disable=invalid-name
    """
        Initialize the interpolation of the thermal integrals.
    """
    global config  # pylint: disable=invalid-name

    defaultIntegrals.Jb.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jb")),
    )
    defaultIntegrals.Jf.readInterpolationTable(
        getSafePathToResource(config.get("DataFiles", "InterpolationTable_Jf")),
    )


# Initialising integrals
_initializeInternal()
