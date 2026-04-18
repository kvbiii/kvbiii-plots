"""
Evaluation plotting modules.

This package contains specialized plotting classes for model evaluation:
- ClassificationPlots: For classification model evaluation
- RegressionPlots: For regression model evaluation
- SHAPPlots: For SHAP-based model explainability

All classes inherit from BasePlots which provides common functionality.
"""

from .classification_plots import ClassificationPlots
from .regression_plots import RegressionPlots
from .shap_plots import SHAPPlots
from .time_series_plots import TimeSeriesPlots

__all__ = [
    "ClassificationPlots",
    "RegressionPlots",
    "SHAPPlots",
    "TimeSeriesPlots",
]
