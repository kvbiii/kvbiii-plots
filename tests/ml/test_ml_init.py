"""
Test module for kvbiii_plots.ml package.

This module tests the imports and structure of the ml subpackage,
ensuring all plotting classes are properly exposed through the package interface.
"""

from kvbiii_plots.ml import OptunaPlots
from kvbiii_plots.base_plots import BasePlots
import kvbiii_plots.ml as ml_module


def test_ml_imports():
    """Test that all expected classes can be imported from ml package."""
    # Verify class exists
    assert OptunaPlots is not None


def test_ml_all_exports():
    """Test that __all__ contains expected exports."""
    expected_exports = ["OptunaPlots"]
    assert hasattr(ml_module, "__all__")
    assert set(ml_module.__all__) == set(expected_exports)


def test_ml_classes_inherit_from_baseplots():
    """Test that all ml classes properly inherit from BasePlots."""
    assert issubclass(OptunaPlots, BasePlots)


def test_ml_package_docstring():
    """Test that ml package has proper documentation."""
    assert ml_module.__doc__ is not None
    assert (
        "machine learning" in ml_module.__doc__.lower()
        or "ml" in ml_module.__doc__.lower()
    )
    assert "plotting" in ml_module.__doc__.lower()


def test_ml_classes_have_docstrings():
    """Test that all ml classes have proper docstrings."""
    classes = [OptunaPlots]
    for cls in classes:
        assert cls.__doc__ is not None
        assert len(cls.__doc__.strip()) > 0


def test_ml_classes_instantiation():
    """Test that all ml classes can be instantiated without errors."""
    # Test instantiation
    optuna_plots = OptunaPlots()

    # Verify they have inherited attributes from BasePlots
    assert hasattr(optuna_plots, "default_template")


def test_ml_classes_have_expected_methods():
    """Test that ml classes have their expected public methods."""
    # Optuna plots expected methods
    optuna_methods = [
        "plot_optuna_optimization_history",
        "plot_optuna_param_importance",
    ]

    optuna_plots = OptunaPlots()
    for method in optuna_methods:
        assert hasattr(optuna_plots, method)
        assert callable(getattr(optuna_plots, method))
