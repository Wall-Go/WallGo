from dataclasses import dataclass

from .Fields import Fields

## Put common data classes etc here

@dataclass
class PhaseInfo:
    # Field values at the two phases at T (we go from 1 to 2)
    phaseLocation1: Fields
    phaseLocation2: Fields
    temperature: float