"""
Machine Learning plotting modules.

This package contains specialized plotting classes for ML algorithms and optimization:
- OptunaPlots: For Optuna hyperparameter optimization visualization

All classes inherit from BasePlots which provides common functionality.
"""

from .optuna_plots import OptunaPlots

__all__ = [
    "OptunaPlots",
]
