"""Model rizika odepsání úvěrů s ručením SBA.

Použití:
    from sba_risk import predict_risk
    predict_risk({"amount": 100_000, "sba_share": 0.75, "naics": 44, "state": "CA"})
"""

from importlib.metadata import version

from sba_risk.api import FIELDS, RiskWarning, predict_many, predict_risk

__version__ = version("sba-risk")
__all__ = ["FIELDS", "RiskWarning", "predict_many", "predict_risk", "__version__"]
