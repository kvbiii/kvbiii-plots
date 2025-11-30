import importlib
import os
import pytest

import kvbiii_plots
from kvbiii_plots import eda
from kvbiii_plots import BasePlots
from kvbiii_plots.eda import (
    ContinuousPlots,
    CategoricalPlots,
    TimeSeriesPlots,
    MultivariatePlots,
)
from kvbiii_plots.evaluation.regression_plots import RegressionPlots


def test_kvbiii_plots_init_module_imports_successfully() -> None:
    """Tests successful import of kvbiii_plots module.

    Asserts:
        - Module can be imported without errors
        - Import returns a valid module object
    """
    try:
        assert kvbiii_plots is not None, "Module import returned None"
    except ImportError as e:
        pytest.fail(f"Failed to import kvbiii_plots module: {str(e)}")


def test_kvbiii_plots_init_package_structure_exists() -> None:
    """Tests kvbiii_plots package has proper structure and metadata.

    Asserts:
        - Package has __name__ attribute
        - Package has __file__ attribute indicating proper installation
        - Package path exists and is accessible
    """
    assert hasattr(kvbiii_plots, "__name__"), "Package missing __name__ attribute"
    assert (
        kvbiii_plots.__name__ == "kvbiii_plots"
    ), f"Expected package name 'kvbiii_plots', got '{kvbiii_plots.__name__}'"
    assert hasattr(kvbiii_plots, "__file__"), "Package missing __file__ attribute"


def test_kvbiii_plots_init_module_reload_handles_correctly() -> None:
    """Tests kvbiii_plots module can be reloaded without errors.

    Asserts:
        - Module can be reloaded using importlib
        - Reloaded module maintains proper attributes
        - No exceptions raised during reload process
    """
    try:
        reloaded_module = importlib.reload(kvbiii_plots)
        assert reloaded_module is not None, "Module reload returned None"
        assert (
            reloaded_module.__name__ == "kvbiii_plots"
        ), "Reloaded module name incorrect"
    except Exception as e:
        pytest.fail(f"Module reload failed: {str(e)}")


def test_kvbiii_plots_init_submodule_eda_accessible() -> None:
    """Tests kvbiii_plots.eda submodule is accessible through package import.

    Asserts:
        - eda submodule can be imported
        - Submodule contains expected plotting classes
    """
    try:
        assert eda is not None, "eda submodule import returned None"
        assert hasattr(eda, "ContinuousPlots"), "eda missing ContinuousPlots class"
        assert hasattr(eda, "CategoricalPlots"), "eda missing CategoricalPlots class"
        assert hasattr(eda, "TimeSeriesPlots"), "eda missing TimeSeriesPlots class"
        assert hasattr(eda, "MultivariatePlots"), "eda missing MultivariatePlots class"
    # OtherPlots now lives under ml subpackage, not eda
    except ImportError as e:
        pytest.fail(f"Failed to import eda submodule: {str(e)}")


def test_kvbiii_plots_init_direct_class_import_works() -> None:
    """Tests direct import of plotting classes from kvbiii_plots.eda.

    Asserts:
        - Plotting classes can be imported directly
        - Classes are properly instantiable
        - Instances have expected methods from BasePlots
    """
    try:
        continuous_plots = ContinuousPlots()
        categorical_plots = CategoricalPlots()

        assert continuous_plots is not None, "ContinuousPlots instantiation failed"
        assert categorical_plots is not None, "CategoricalPlots instantiation failed"

        # Check inheritance from BasePlots
        assert hasattr(
            continuous_plots, "check_data"
        ), "ContinuousPlots missing check_data method"
        assert hasattr(
            categorical_plots, "check_data"
        ), "CategoricalPlots missing check_data method"

    except ImportError as e:
        pytest.fail(f"Failed to import plotting classes: {str(e)}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate plotting classes: {str(e)}")


def test_kvbiii_plots_init_baseplot_import_works() -> None:
    """Tests BasePlots class can be imported from main package.

    Asserts:
        - BasePlots class can be imported directly from kvbiii_plots
        - Class is properly instantiable
        - Instance has core plotting functionality
    """
    try:
        base_plots = BasePlots()
        assert base_plots is not None, "BasePlots instantiation failed"
        assert hasattr(base_plots, "check_data"), "BasePlots missing check_data method"
        assert hasattr(
            base_plots, "apply_default_layout"
        ), "BasePlots missing apply_default_layout method"
        assert hasattr(
            base_plots, "check_2d_data"
        ), "BasePlots missing check_2d_data method"

    except ImportError as e:
        pytest.fail(f"Failed to import BasePlots class: {str(e)}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate BasePlots class: {str(e)}")


def test_kvbiii_plots_init_evaluation_module_accessibility() -> None:
    """Tests evaluation module classes can be imported directly.

    Asserts:
        - RegressionPlots class can be imported from evaluation module
        - Class is properly instantiable and inherits from BasePlots
    """
    try:
        regression_plots = RegressionPlots()
        assert regression_plots is not None, "RegressionPlots instantiation failed"
        assert hasattr(
            regression_plots, "check_data"
        ), "RegressionPlots missing inherited check_data method"
        assert hasattr(
            regression_plots, "homoscedacity_plot"
        ), "RegressionPlots missing homoscedacity_plot method"
        assert hasattr(
            regression_plots, "true_vs_fitted_plot"
        ), "RegressionPlots missing true_vs_fitted_plot method"

    except ImportError as e:
        pytest.fail(f"Failed to import RegressionPlots class: {str(e)}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate RegressionPlots class: {str(e)}")


def test_kvbiii_plots_init_all_eda_classes_instantiable() -> None:
    """Tests all EDA plotting classes can be instantiated successfully.

    Asserts:
        - All five EDA plotting classes can be imported and instantiated
        - All classes inherit proper methods from BasePlots
        - No exceptions raised during instantiation
    """
    try:
        # Instantiate all classes
        classes = [
            ContinuousPlots(),
            CategoricalPlots(),
            TimeSeriesPlots(),
            MultivariatePlots(),
        ]

        # Verify all instances are valid and have inherited methods
        for instance in classes:
            assert (
                instance is not None
            ), f"{type(instance).__name__} instantiation failed"
            assert hasattr(
                instance, "check_data"
            ), f"{type(instance).__name__} missing check_data method"
            assert hasattr(
                instance, "apply_default_layout"
            ), f"{type(instance).__name__} missing apply_default_layout method"

    except ImportError as e:
        pytest.fail(f"Failed to import EDA plotting classes: {str(e)}")
    except Exception as e:
        pytest.fail(f"Failed to instantiate EDA plotting classes: {str(e)}")


def test_kvbiii_plots_init_package_version_and_metadata() -> None:
    """Tests package has proper version and metadata attributes.

    Asserts:
        - Package has __all__ attribute defining public API
        - Package imports work from __all__ list
        - Core functionality is accessible
    """
    try:
        # Check if package has __all__ defined
        if hasattr(kvbiii_plots, "__all__"):
            all_items = kvbiii_plots.__all__
            assert isinstance(all_items, list), "__all__ should be a list"
            assert len(all_items) > 0, "__all__ should not be empty"

            # Verify each item in __all__ is accessible
            for item in all_items:
                assert hasattr(
                    kvbiii_plots, item
                ), f"Package missing declared item: {item}"

        # Test that main classes are accessible
        assert hasattr(
            kvbiii_plots, "BasePlots"
        ), "Package missing BasePlots in public API"
        assert hasattr(
            kvbiii_plots, "eda"
        ), "Package missing eda submodule in public API"

    except ImportError as e:
        pytest.fail(f"Failed to test package metadata: {str(e)}")
    except Exception as e:
        pytest.fail(f"Error testing package structure: {str(e)}")


def test_kvbiii_plots_init_examples_directory_structure() -> None:
    """Tests package includes proper examples directory structure.

    Asserts:
        - Examples directory exists in eda subpackage
        - Example notebooks exist for each plotting class
        - Examples correspond to available plotting functionality
    """
    try:
        # Get the package path
        package_path = os.path.dirname(kvbiii_plots.__file__)
        examples_path = os.path.join(package_path, "eda", "examples")

        # Check if examples directory exists
        assert os.path.exists(
            examples_path
        ), "Examples directory not found in eda subpackage"
        assert os.path.isdir(
            examples_path
        ), "Examples path exists but is not a directory"

        # Check for expected example notebooks
        expected_examples = [
            "categorical_plots_examples.ipynb",
            "continuous_plots_examples.ipynb",
            "multivariate_plots_examples.ipynb",
            "time_series_plots_examples.ipynb",
        ]

        for example_file in expected_examples:
            example_full_path = os.path.join(examples_path, example_file)
            assert os.path.exists(
                example_full_path
            ), f"Missing example file: {example_file}"
            assert os.path.isfile(
                example_full_path
            ), f"Example path exists but is not a file: {example_file}"

        # Verify examples directory is not empty
        examples_list = os.listdir(examples_path)
        assert len(examples_list) > 0, "Examples directory is empty"

    except Exception as e:
        pytest.fail(f"Error testing examples directory structure: {str(e)}")
