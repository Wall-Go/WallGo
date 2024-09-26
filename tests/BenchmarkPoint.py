import WallGo.genericModel


## Collect input params + other benchmark-specific data for various things in one place.
class BenchmarkPoint:

    ## This is model-specific input like particle masses
    inputParams: dict[str, float]
    ## This is required input for WallGo to find the transition (Tn and approx phase locations)
    phaseInfo: dict[str, float]

    ## This is WallGo internal config info that we may want to fix on a per-benchmark basis. IE. temperature interpolation ranges
    config: dict[str, float]

    ## Expected results for the benchmark point
    expectedResults: dict[str, float]

    def __init__(
        self,
        inputParams: dict[str, float],
        phaseInfo: dict[str, float] = {},
        config: dict[str, float] = {},
        expectedResults: dict[str, float] = {},
    ):
        self.inputParams = inputParams
        self.phaseInfo = phaseInfo
        self.config = config
        self.expectedResults = expectedResults


class BenchmarkModel:
    """This just holds a model instance + BenchmarkPoint."""

    model: WallGo.GenericModel
    benchmarkPoint: BenchmarkPoint

    def __init__(self, model: WallGo.GenericModel, benchmarkPoint: BenchmarkPoint):
        self.model = model
        self.model.getEffectivePotential().configureDerivatives(WallGo.VeffDerivativeSettings(1.0, 1.0))
        self.model.getEffectivePotential().effectivePotentialError = 1e-15
        self.benchmarkPoint = benchmarkPoint
