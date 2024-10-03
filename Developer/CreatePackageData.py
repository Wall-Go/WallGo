
"""Script for producing pre-computed data that we package with WallGo.
Internal use only, do NOT package this file! 
"""

from WallGo import PotentialTools


def makeDefaultInterpolationTables():
    """Produces interpolation tables for integrals that WallGo uses by default. 
    """

    pointCount = 10000

    Jb = PotentialTools.JbIntegral(bUseAdaptiveInterpolation=False)
    Jf = PotentialTools.JfIntegral(bUseAdaptiveInterpolation=False)

    ## Range of (m/T)^2 that we interpolate over. After 1400 Jb/Jf are basically zero though.
    ## Do note that for negative input these need analytical continuation and are increasingly oscillatory,
    ## And the integrator tends to throw warnings. Whether it's physically correct to use these for negative m^2
    ## is something I'm not sure about. 
    #Jb.newInterpolationTable(-20., 1000., pointCount)
    #Jf.newInterpolationTable(-20., 1000., pointCount)

    Jb.newInterpolationTable(-20., 1000., pointCount)
    Jf.newInterpolationTable(-20., 1000., pointCount)

    Jb.writeInterpolationTable("InterpolationTable_Jb.txt")
    Jf.writeInterpolationTable("InterpolationTable_Jf.txt")


makeDefaultInterpolationTables()