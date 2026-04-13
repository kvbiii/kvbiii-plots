"""
Plotting modules for data visualization.

This package provides a structured approach to data visualization with:
- BasePlots: Core functionality and utilities
- EDA subpackage: Specialized exploratory data analysis plots
- ML subpackage: Machine learning algorithms, clustering, and anomaly detection plots
- Evaluation subpackage: Model evaluation and performance visualization

The modular design allows for easy extension and maintenance of plotting functionality.
"""

from __future__ import annotations
import importlib
from importlib.metadata import PackageNotFoundError, version
from . import eda, evaluation, ml
from .base_plots import BasePlots

__version__ = "0+unknown"
try:
    __version__ = version("kvbiii_plots")
except PackageNotFoundError:
    pass

__all__ = [
    "__version__",
    "BasePlots",
    "eda",
    "ml",
    "evaluation",
]


def __getattr__(name: str):
    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
