from kvbiii_plots.base_plots import BasePlots
from kvbiii_plots.ml import OptunaPlots
import kvbiii_plots.ml as ml_module


def test_ml_imports() -> None:
    """Test that all expected classes can be imported from ml package."""

    assert OptunaPlots is not None


def test_ml_all_exports() -> None:
    """Test that __all__ contains expected exports."""
    expected_exports = ["OptunaPlots"]
    assert hasattr(ml_module, "__all__")
    assert set(ml_module.__all__) == set(expected_exports)


def test_ml_classes_inherit_from_baseplots() -> None:
    """Test that all ml classes properly inherit from BasePlots."""
    assert issubclass(OptunaPlots, BasePlots)


def test_ml_package_docstring() -> None:
    """Test that ml package has proper documentation."""
    assert ml_module.__doc__ is not None
    assert (
        "machine learning" in ml_module.__doc__.lower()
        or "ml" in ml_module.__doc__.lower()
    )
    assert "plotting" in ml_module.__doc__.lower()


def test_ml_classes_have_docstrings() -> None:
    """Test that all ml classes have proper docstrings."""
    classes = [OptunaPlots]
    for cls in classes:
        assert cls.__doc__ is not None
        assert len(cls.__doc__.strip()) > 0


def test_ml_classes_instantiation() -> None:
    """Test that all ml classes can be instantiated without errors."""

    optuna_plots = OptunaPlots()

    assert hasattr(optuna_plots, "default_template")


def test_ml_classes_have_expected_methods() -> None:
    """Test that ml classes have their expected public methods."""

    optuna_methods = [
        "plot_optuna_optimization_history",
        "plot_optuna_param_importance",
    ]

    optuna_plots = OptunaPlots()
    for method in optuna_methods:
        assert hasattr(optuna_plots, method)
        assert callable(getattr(optuna_plots, method))
