"""
Plotting modules for data visualization.

This package provides a structured approach to data visualization with:
- BasePlots: Core functionality and utilities
- EDA subpackage: Specialized exploratory data analysis plots
- ML subpackage: Machine learning algorithms, clustering, and anomaly detection plots
- Evaluation subpackage: Model evaluation and performance visualization

The modular design allows for easy extension and maintenance of plotting functionality.
"""

from . import eda, evaluation, ml
from .base_plots import BasePlots

__all__ = ["BasePlots", "eda", "ml", "evaluation"]
