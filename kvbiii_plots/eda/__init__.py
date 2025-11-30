"""
EDA (Exploratory Data Analysis) plotting modules.

This package contains specialized plotting classes for different types of data analysis:
- ContinuousPlots: For continuous variable analysis
- CategoricalPlots: For categorical variable analysis
- TimeSeriesPlots: For time series data analysis
- MultivariatePlots: For multivariate analysis and correlations

All classes inherit from BasePlots which provides common functionality.
"""

from .continuous_plots import ContinuousPlots
from .categorical_plots import CategoricalPlots
from .time_series_plots import TimeSeriesPlots
from .multivariate_plots import MultivariatePlots

__all__ = [
    "ContinuousPlots",
    "CategoricalPlots",
    "TimeSeriesPlots",
    "MultivariatePlots",
]
