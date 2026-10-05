"""GATTACA: measure how much fingerprint ridge detail a photo of *your own hand* exposes.

GATTACA is a privacy-awareness tool. It estimates exposure risk; it does not store
biometrics or defeat security systems.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "GATTACA is a privacy-awareness tool. It estimates exposure risk; "
    "it does not store biometrics or defeat security systems."
)

# Ridge period of an adult fingertip is ~0.40-0.50 mm. At the 500 ppi contact-scan
# standard (19.7 px/mm) that is ~8-10 px; every print is resampled to this period
# before extraction (automatic version of Lin & Kumar's scale correction).
CANONICAL_PERIOD_PX = 9.0
RIDGE_PERIOD_MM = 0.45
