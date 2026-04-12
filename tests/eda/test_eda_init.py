from kvbiii_plots.base_plots import BasePlots
from kvbiii_plots.eda import (
    CategoricalPlots,
    ContinuousPlots,
    MultivariatePlots,
    TimeSeriesPlots,
)
import kvbiii_plots.eda as eda_module


def test_eda_imports() -> None:
    """Test that all expected classes can be imported from eda package."""

    if not (ContinuousPlots is not None):
        raise AssertionError("Assertion failed.")
    if not (CategoricalPlots is not None):
        raise AssertionError("Assertion failed.")
    if not (TimeSeriesPlots is not None):
        raise AssertionError("Assertion failed.")
    if not (MultivariatePlots is not None):
        raise AssertionError("Assertion failed.")


def test_eda_all_exports() -> None:
    """Test that __all__ contains expected exports."""
    expected_exports = [
        "ContinuousPlots",
        "CategoricalPlots",
        "TimeSeriesPlots",
        "MultivariatePlots",
    ]
    if not (hasattr(eda_module, "__all__")):
        raise AssertionError("Assertion failed.")
    if not (set(eda_module.__all__) == set(expected_exports)):
        raise AssertionError("Assertion failed.")


def test_eda_classes_inherit_from_baseplots() -> None:
    """Test that all eda classes properly inherit from BasePlots."""
    if not (issubclass(ContinuousPlots, BasePlots)):
        raise AssertionError("Assertion failed.")
    if not (issubclass(CategoricalPlots, BasePlots)):
        raise AssertionError("Assertion failed.")
    if not (issubclass(TimeSeriesPlots, BasePlots)):
        raise AssertionError("Assertion failed.")
    if not (issubclass(MultivariatePlots, BasePlots)):
        raise AssertionError("Assertion failed.")


def test_eda_package_docstring() -> None:
    """Test that eda package has proper documentation."""
    if not (eda_module.__doc__ is not None):
        raise AssertionError("Assertion failed.")
    if not (
        "eda" in eda_module.__doc__.lower()
        or "exploratory" in eda_module.__doc__.lower()
    ):
        raise AssertionError("Assertion failed.")
    if not ("plotting" in eda_module.__doc__.lower()):
        raise AssertionError("Assertion failed.")


def test_eda_classes_have_docstrings() -> None:
    """Test that all eda classes have proper docstrings."""
    classes = [
        ContinuousPlots,
        CategoricalPlots,
        TimeSeriesPlots,
        MultivariatePlots,
    ]
    for cls in classes:
        if not (cls.__doc__ is not None):
            raise AssertionError("Assertion failed.")
        if not (len(cls.__doc__.strip()) > 0):
            raise AssertionError("Assertion failed.")


def test_eda_classes_instantiation() -> None:
    """Test that all eda classes can be instantiated without errors."""

    continuous_plots = ContinuousPlots()
    categorical_plots = CategoricalPlots()
    time_series_plots = TimeSeriesPlots()
    multivariate_plots = MultivariatePlots()

    if not (hasattr(continuous_plots, "default_template")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(categorical_plots, "default_template")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(time_series_plots, "default_template")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(multivariate_plots, "default_template")):
        raise AssertionError("Assertion failed.")


def test_eda_classes_have_expected_methods() -> None:
    """Test that eda classes have their expected public methods."""

    continuous_methods = [
        "scatter_plot",
        "histogram_and_box_plot",
        "histogram_boxplot_linear_regression",
        "boxplot_histogram_boxplot_by_hue",
    ]

    continuous_plots = ContinuousPlots()
    for method in continuous_methods:
        if not (hasattr(continuous_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(continuous_plots, method))):
            raise AssertionError("Assertion failed.")

    categorical_methods = [
        "barplot",
        "pie_barplot",
        "boxplot_by_categorical",
        "pie_boxplot_by_categorical",
        "pie_stacked_barplot_by_hue",
    ]

    categorical_plots = CategoricalPlots()
    for method in categorical_methods:
        if not (hasattr(categorical_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(categorical_plots, method))):
            raise AssertionError("Assertion failed.")

    time_series_methods = [
        "plot_time_series_mean",
        "plot_time_series_multiple_metrics",
        "plot_time_series_with_trend",
        "plot_seasonal_decomposition",
    ]

    time_series_plots = TimeSeriesPlots()
    for method in time_series_methods:
        if not (hasattr(time_series_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(time_series_plots, method))):
            raise AssertionError("Assertion failed.")

    multivariate_methods = [
        "correlation_plot",
        "scatter_matrix",
        "parallel_coordinates",
    ]

    multivariate_plots = MultivariatePlots()
    for method in multivariate_methods:
        if not (hasattr(multivariate_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(multivariate_plots, method))):
            raise AssertionError("Assertion failed.")
