"""
Test module for kvbiii_plots.eda package.

This module tests the imports and structure of the eda subpackage,
ensuring all plotting classes are properly exposed through the package interface.
"""

from kvbiii_plots.eda import (
    ContinuousPlots,
    CategoricalPlots,
    TimeSeriesPlots,
    MultivariatePlots,
)
from kvbiii_plots.base_plots import BasePlots
import kvbiii_plots.eda as eda_module


def test_eda_imports():
    """Test that all expected classes can be imported from eda package."""
    # Verify classes exist
    assert ContinuousPlots is not None
    assert CategoricalPlots is not None
    assert TimeSeriesPlots is not None
    assert MultivariatePlots is not None


def test_eda_all_exports():
    """Test that __all__ contains expected exports."""
    expected_exports = [
        "ContinuousPlots",
        "CategoricalPlots",
        "TimeSeriesPlots",
        "MultivariatePlots",
    ]
    assert hasattr(eda_module, "__all__")
    assert set(eda_module.__all__) == set(expected_exports)


def test_eda_classes_inherit_from_baseplots():
    """Test that all eda classes properly inherit from BasePlots."""
    assert issubclass(ContinuousPlots, BasePlots)
    assert issubclass(CategoricalPlots, BasePlots)
    assert issubclass(TimeSeriesPlots, BasePlots)
    assert issubclass(MultivariatePlots, BasePlots)


def test_eda_package_docstring():
    """Test that eda package has proper documentation."""
    assert eda_module.__doc__ is not None
    assert (
        "eda" in eda_module.__doc__.lower()
        or "exploratory" in eda_module.__doc__.lower()
    )
    assert "plotting" in eda_module.__doc__.lower()


def test_eda_classes_have_docstrings():
    """Test that all eda classes have proper docstrings."""
    classes = [
        ContinuousPlots,
        CategoricalPlots,
        TimeSeriesPlots,
        MultivariatePlots,
    ]
    for cls in classes:
        assert cls.__doc__ is not None
        assert len(cls.__doc__.strip()) > 0


def test_eda_classes_instantiation():
    """Test that all eda classes can be instantiated without errors."""
    # Test instantiation
    continuous_plots = ContinuousPlots()
    categorical_plots = CategoricalPlots()
    time_series_plots = TimeSeriesPlots()
    multivariate_plots = MultivariatePlots()

    # Verify they have inherited attributes from BasePlots
    assert hasattr(continuous_plots, "default_template")
    assert hasattr(categorical_plots, "default_template")
    assert hasattr(time_series_plots, "default_template")
    assert hasattr(multivariate_plots, "default_template")


def test_eda_classes_have_expected_methods():
    """Test that eda classes have their expected public methods."""
    # Continuous plots expected methods
    continuous_methods = [
        "scatter_plot",
        "histogram_and_box_plot",
        "histogram_boxplot_linear_regression",
        "boxplot_histogram_boxplot_by_hue",
    ]

    continuous_plots = ContinuousPlots()
    for method in continuous_methods:
        assert hasattr(continuous_plots, method)
        assert callable(getattr(continuous_plots, method))

    # Categorical plots expected methods
    categorical_methods = [
        "barplot",
        "pie_barplot",
        "boxplot_by_categorical",
        "pie_boxplot_by_categorical",
        "pie_stacked_barplot_by_hue",
    ]

    categorical_plots = CategoricalPlots()
    for method in categorical_methods:
        assert hasattr(categorical_plots, method)
        assert callable(getattr(categorical_plots, method))

    # Time series plots expected methods
    time_series_methods = [
        "plot_time_series_mean",
        "plot_time_series_multiple_metrics",
        "plot_time_series_with_trend",
        "plot_seasonal_decomposition",
    ]

    time_series_plots = TimeSeriesPlots()
    for method in time_series_methods:
        assert hasattr(time_series_plots, method)
        assert callable(getattr(time_series_plots, method))

    # Multivariate plots expected methods
    multivariate_methods = [
        "correlation_plot",
        "scatter_matrix",
        "parallel_coordinates",
    ]

    multivariate_plots = MultivariatePlots()
    for method in multivariate_methods:
        assert hasattr(multivariate_plots, method)
        assert callable(getattr(multivariate_plots, method))
