class BoltzmannParticle:
    def __init__(
        self,
        msqVaccum,
        msqThermal,
        statistics,
        isOutOfEquilibrium,
        collisionPrefactors,
    ):
        self.msqVacuum = msqVaccum
        self.msqThermal = msqThermal
        assert statistics in [
            -1,
            1,
        ], "BoltzmannParticle error: statistics not understood %s" % (
            statistics
        )
        self.statistics = statistics
        self.isOutOfEquilibrium = isOutOfEquilibrium
        self.collisionPrefactors = collisionPrefactors


# Top
msqVacuum_Top = lambda x: 0.5 * x**2
msqThermal_Top = lambda T: 0.1 * T**2
statistics_Top = -1
isOutOfEquilibrium_Top = True
gsq = 0.5
collisionPrefactors_Top = [gsq**2, gsq**2, gsq**2]
top = BoltzmannParticle(
    msqVacuum_Top,
    msqThermal_Top,
    statistics_Top,
    isOutOfEquilibrium_Top,
    collisionPrefactors_Top,
)

# Higgs
msqVacuum_Higgs = lambda x: -10 + 0.5 * x**2
msqThermal_Higgs = lambda T: 0.2 * T**2
statistics_Higgs = 1
isOutOfEquilibrium_Higgs = False
collisionPrefactors_Higgs = [0, 0, 0]
Higgs = BoltzmannParticle(
    msqVacuum_Higgs,
    msqThermal_Higgs,
    statistics_Higgs,
    isOutOfEquilibrium_Higgs,
    collisionPrefactors_Higgs,
)
