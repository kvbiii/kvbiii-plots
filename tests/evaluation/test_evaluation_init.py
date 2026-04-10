from kvbiii_plots.base_plots import BasePlots
from kvbiii_plots.evaluation import ClassificationPlots, RegressionPlots, SHAPPlots
import kvbiii_plots.evaluation as evaluation_module


def test_evaluation_imports() -> None:
    """Test that all expected classes can be imported from evaluation package."""

    assert ClassificationPlots is not None
    assert RegressionPlots is not None
    assert SHAPPlots is not None


def test_evaluation_all_exports() -> None:
    """Test that __all__ contains expected exports."""
    expected_exports = ["ClassificationPlots", "RegressionPlots", "SHAPPlots"]
    assert hasattr(evaluation_module, "__all__")
    assert set(evaluation_module.__all__) == set(expected_exports)


def test_evaluation_classes_inherit_from_baseplots() -> None:
    """Test that all evaluation classes properly inherit from BasePlots."""
    assert issubclass(ClassificationPlots, BasePlots)
    assert issubclass(RegressionPlots, BasePlots)
    assert issubclass(SHAPPlots, BasePlots)


def test_evaluation_package_docstring() -> None:
    """Test that evaluation package has proper documentation."""
    assert evaluation_module.__doc__ is not None
    assert "evaluation" in evaluation_module.__doc__.lower()
    assert "plotting" in evaluation_module.__doc__.lower()


def test_evaluation_classes_have_docstrings() -> None:
    """Test that all evaluation classes have proper docstrings."""
    classes = [ClassificationPlots, RegressionPlots, SHAPPlots]
    for cls in classes:
        assert cls.__doc__ is not None
        assert len(cls.__doc__.strip()) > 0


def test_evaluation_classes_instantiation() -> None:
    """Test that all evaluation classes can be instantiated without errors."""

    classification_plots = ClassificationPlots()
    regression_plots = RegressionPlots()
    shap_plots = SHAPPlots()

    assert hasattr(classification_plots, "default_template")
    assert hasattr(regression_plots, "default_template")
    assert hasattr(shap_plots, "default_template")


def test_evaluation_classes_have_expected_methods() -> None:
    """Test that evaluation classes have their expected public methods."""

    classification_methods = [
        "plot_confusion_matrix",
        "subplot_multilabel_conf_matrix",
        "plot_probabilities_per_class",
        "plot_roc_auc",
    ]

    classification_plots = ClassificationPlots()
    for method in classification_methods:
        assert hasattr(classification_plots, method)
        assert callable(getattr(classification_plots, method))

    regression_methods = [
        "homoscedacity_plot",
        "true_vs_fitted_plot",
        "residual_distribution_plot",
    ]

    regression_plots = RegressionPlots()
    for method in regression_methods:
        assert hasattr(regression_plots, method)
        assert callable(getattr(regression_plots, method))

    shap_methods = [
        "plot_shap_bar",
        "plot_custom_shap_beeswarm",
        "plot_shap_categorical_box",
        "plot_shap_numerical_scatter",
        "plot_shap_force",
    ]

    shap_plots = SHAPPlots()
    for method in shap_methods:
        assert hasattr(shap_plots, method)
        assert callable(getattr(shap_plots, method))
