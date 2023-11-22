## Collect input params + other benchmark-specific data for various things in one place.
## Other data can be things like critical/nucleation temperatures
class BenchmarkPoint:
    
    def __init__(self, inputParams: dict[str, float], otherData: dict[str, float] = None):
        self.inputParams = inputParams
        self.otherData = otherData


