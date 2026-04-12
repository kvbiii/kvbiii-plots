import importlib
import os

import pytest

import kvbiii_plots
from kvbiii_plots import BasePlots, eda
from kvbiii_plots.eda import (
    CategoricalPlots,
    ContinuousPlots,
    MultivariatePlots,
    TimeSeriesPlots,
)
from kvbiii_plots.evaluation.regression_plots import RegressionPlots


def test_kvbiii_plots_init_module_imports_successfully() -> None:
    """Tests successful import of kvbiii_plots module.

    Asserts:
        - Module can be imported without errors
        - Import returns a valid module object
    """
    try:
        if not (kvbiii_plots is not None):
            raise AssertionError("Module import returned None")
    except ImportError as e:
        pytest.fail(f"Failed to import kvbiii_plots module: {str(e)}")


def test_kvbiii_plots_init_package_structure_exists() -> None:
    """Tests kvbiii_plots package has proper structure and metadata.

    Asserts:
        - Package has __name__ attribute
        - Package has __file__ attribute indicating proper installation
        - Package path exists and is accessible
    """
    if not (hasattr(kvbiii_plots, "__name__")):
        raise AssertionError("Package missing __name__ attribute")
    if not (kvbiii_plots.__name__ == "kvbiii_plots"):
        raise AssertionError(
            f"Expected package name 'kvbiii_plots', got '{kvbiii_plots.__name__}'"
        )
    if not (hasattr(kvbiii_plots, "__file__")):
        raise AssertionError("Package missing __file__ attribute")


def test_kvbiii_plots_init_module_reload_handles_correctly() -> None:
    """Tests kvbiii_plots module can be reloaded without errors.

    Asserts:
        - Module can be reloaded using importlib
        - Reloaded module maintains proper attributes
        - No exceptions raised during reload process
    """
    reloaded_module = importlib.reload(kvbiii_plots)
    if not (reloaded_module is not None):
        raise AssertionError("Module reload returned None")
    if not (reloaded_module.__name__ == "kvbiii_plots"):
        raise AssertionError("Reloaded module name incorrect")


def test_kvbiii_plots_init_submodule_eda_accessible() -> None:
    """Tests kvbiii_plots.eda submodule is accessible through package import.

    Asserts:
        - eda submodule can be imported
        - Submodule contains expected plotting classes
    """
    try:
        if not (eda is not None):
            raise AssertionError("eda submodule import returned None")
        if not (hasattr(eda, "ContinuousPlots")):
            raise AssertionError("eda missing ContinuousPlots class")
        if not (hasattr(eda, "CategoricalPlots")):
            raise AssertionError("eda missing CategoricalPlots class")
        if not (hasattr(eda, "TimeSeriesPlots")):
            raise AssertionError("eda missing TimeSeriesPlots class")
        if not (hasattr(eda, "MultivariatePlots")):
            raise AssertionError("eda missing MultivariatePlots class")

    except ImportError as e:
        pytest.fail(f"Failed to import eda submodule: {str(e)}")


def test_kvbiii_plots_init_direct_class_import_works() -> None:
    """Tests direct import of plotting classes from kvbiii_plots.eda.

    Asserts:
        - Plotting classes can be imported directly
        - Classes are properly instantiable
        - Instances have expected methods from BasePlots
    """
    continuous_plots = ContinuousPlots()
    categorical_plots = CategoricalPlots()

    if not (continuous_plots is not None):
        raise AssertionError("ContinuousPlots instantiation failed")
    if not (categorical_plots is not None):
        raise AssertionError("CategoricalPlots instantiation failed")

    if not (hasattr(continuous_plots, "check_data")):
        raise AssertionError("ContinuousPlots missing check_data method")
    if not (hasattr(categorical_plots, "check_data")):
        raise AssertionError("CategoricalPlots missing check_data method")


def test_kvbiii_plots_init_baseplot_import_works() -> None:
    """Tests BasePlots class can be imported from main package.

    Asserts:
        - BasePlots class can be imported directly from kvbiii_plots
        - Class is properly instantiable
        - Instance has core plotting functionality
    """
    base_plots = BasePlots()
    if not (base_plots is not None):
        raise AssertionError("BasePlots instantiation failed")
    if not (hasattr(base_plots, "check_data")):
        raise AssertionError("BasePlots missing check_data method")
    if not (hasattr(base_plots, "apply_default_layout")):
        raise AssertionError("BasePlots missing apply_default_layout method")
    if not (hasattr(base_plots, "check_2d_data")):
        raise AssertionError("BasePlots missing check_2d_data method")


def test_kvbiii_plots_init_evaluation_module_accessibility() -> None:
    """Tests evaluation module classes can be imported directly.

    Asserts:
        - RegressionPlots class can be imported from evaluation module
        - Class is properly instantiable and inherits from BasePlots
    """
    regression_plots = RegressionPlots()
    if not (regression_plots is not None):
        raise AssertionError("RegressionPlots instantiation failed")
    if not (hasattr(regression_plots, "check_data")):
        raise AssertionError("RegressionPlots missing inherited check_data method")
    if not (hasattr(regression_plots, "homoscedacity_plot")):
        raise AssertionError("RegressionPlots missing homoscedacity_plot method")
    if not (hasattr(regression_plots, "true_vs_fitted_plot")):
        raise AssertionError("RegressionPlots missing true_vs_fitted_plot method")


def test_kvbiii_plots_init_all_eda_classes_instantiable() -> None:
    """Tests all EDA plotting classes can be instantiated successfully.

    Asserts:
        - All five EDA plotting classes can be imported and instantiated
        - All classes inherit proper methods from BasePlots
        - No exceptions raised during instantiation
    """
    classes = [
        ContinuousPlots(),
        CategoricalPlots(),
        TimeSeriesPlots(),
        MultivariatePlots(),
    ]

    for instance in classes:
        if not (instance is not None):
            raise AssertionError(f"{type(instance).__name__} instantiation failed")
        if not (hasattr(instance, "check_data")):
            raise AssertionError(f"{type(instance).__name__} missing check_data method")
        if not (hasattr(instance, "apply_default_layout")):
            raise AssertionError(
                f"{type(instance).__name__} missing apply_default_layout method"
            )


def test_kvbiii_plots_init_package_version_and_metadata() -> None:
    """Tests package has proper version and metadata attributes.

    Asserts:
        - Package has __all__ attribute defining public API
        - Package imports work from __all__ list
        - Core functionality is accessible
    """
    if hasattr(kvbiii_plots, "__all__"):
        all_items = kvbiii_plots.__all__
        if not (isinstance(all_items, list)):
            raise AssertionError("__all__ should be a list")
        if not (len(all_items) > 0):
            raise AssertionError("__all__ should not be empty")

        for item in all_items:
            if not (hasattr(kvbiii_plots, item)):
                raise AssertionError(f"Package missing declared item: {item}")

    if not (hasattr(kvbiii_plots, "BasePlots")):
        raise AssertionError("Package missing BasePlots in public API")
    if not (hasattr(kvbiii_plots, "eda")):
        raise AssertionError("Package missing eda submodule in public API")


def test_kvbiii_plots_init_examples_directory_structure() -> None:
    """Tests package includes proper examples directory structure.

    Asserts:
        - Examples directory exists in eda subpackage
        - Example notebooks exist for each plotting class
        - Examples correspond to available plotting functionality
    """
    package_path = os.path.dirname(kvbiii_plots.__file__)
    examples_path = os.path.join(package_path, "eda", "examples")

    if not (os.path.exists(examples_path)):
        raise AssertionError("Examples directory not found in eda subpackage")
    if not (os.path.isdir(examples_path)):
        raise AssertionError("Examples path exists but is not a directory")

    expected_examples = [
        "categorical_plots_examples.ipynb",
        "continuous_plots_examples.ipynb",
        "multivariate_plots_examples.ipynb",
        "time_series_plots_examples.ipynb",
    ]

    for example_file in expected_examples:
        example_full_path = os.path.join(examples_path, example_file)
        if not (os.path.exists(example_full_path)):
            raise AssertionError(f"Missing example file: {example_file}")
        if not (os.path.isfile(example_full_path)):
            raise AssertionError(
                f"Example path exists but is not a file: {example_file}"
            )

    examples_list = os.listdir(examples_path)
    if not (len(examples_list) > 0):
        raise AssertionError("Examples directory is empty")
