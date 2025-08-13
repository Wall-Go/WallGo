import WallGo.genericModel


from typing import Any

## Collect input params + other benchmark-specific data for various things in one place.
class BenchmarkPoint:

    ## This is model-specific input like particle masses
    inputParams: dict[str, float]
    ## This is required input for WallGo to find the transition (Tn and approx phase locations)
    phaseInfo: dict[str, float]

    ## This is WallGo internal config info that we may want to fix on a per-benchmark basis. IE. temperature interpolation ranges
    # BREAKING CHANGE: The type annotation for `config` was changed from `dict[str, float]` to `dict[str, Any]`
    # to allow for more flexible configuration values (e.g., nested dictionaries for derivative settings).
    # This broadens the accepted types and may break code that expects all config values to be floats.
    config: dict[str, Any]

    ## Expected results for the benchmark point
    expectedResults: dict[str, float]

    def __init__(
        self,
        inputParams: dict[str, float],
        phaseInfo: dict[str, float] | None = None,
        config: dict[str, Any] | None = None,
        expectedResults: dict[str, float] | None = None,
    ):
        self.inputParams = inputParams
        self.phaseInfo = phaseInfo or {}
        self.config = config or {}
        self.expectedResults = expectedResults or {}


class BenchmarkModel:
    """This just holds a model instance + BenchmarkPoint."""

    model: WallGo.GenericModel
    benchmarkPoint: BenchmarkPoint

    def __init__(self, model: WallGo.GenericModel, benchmarkPoint: BenchmarkPoint):
        self.model = model
        
        # Apply derivative settings if specified in benchmark configuration
        derivative_settings = benchmarkPoint.config.get("derivativeSettings")
        if derivative_settings is not None:
            if isinstance(derivative_settings, dict):
                temp_scale = derivative_settings.get("temperatureVariationScale", 1.0)
                field_scale = derivative_settings.get("fieldValueVariationScale", 1.0)
                self.model.getEffectivePotential().configureDerivatives(
                    WallGo.VeffDerivativeSettings(temp_scale, field_scale)
                )
            else:
                # Use default settings if configuration is malformed
                self.model.getEffectivePotential().configureDerivatives(
                    WallGo.VeffDerivativeSettings(1.0, 1.0)
                )
        else:
            # Use default settings if not specified
            self.model.getEffectivePotential().configureDerivatives(
                WallGo.VeffDerivativeSettings(1.0, 1.0)
            )
        
        self.model.getEffectivePotential().effectivePotentialError = 1e-15
        self.benchmarkPoint = benchmarkPoint
