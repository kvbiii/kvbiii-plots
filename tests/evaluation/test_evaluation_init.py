from kvbiii_plots.base_plots import BasePlots
from kvbiii_plots.evaluation import ClassificationPlots, RegressionPlots, SHAPPlots
import kvbiii_plots.evaluation as evaluation_module


def test_evaluation_imports() -> None:
    """Test that all expected classes can be imported from evaluation package."""

    if not (ClassificationPlots is not None):
        raise AssertionError("Assertion failed.")
    if not (RegressionPlots is not None):
        raise AssertionError("Assertion failed.")
    if not (SHAPPlots is not None):
        raise AssertionError("Assertion failed.")


def test_evaluation_all_exports() -> None:
    """Test that __all__ contains expected exports."""
    expected_exports = ["ClassificationPlots", "RegressionPlots", "SHAPPlots"]
    if not (hasattr(evaluation_module, "__all__")):
        raise AssertionError("Assertion failed.")
    if not (set(evaluation_module.__all__) == set(expected_exports)):
        raise AssertionError("Assertion failed.")


def test_evaluation_classes_inherit_from_baseplots() -> None:
    """Test that all evaluation classes properly inherit from BasePlots."""
    if not (issubclass(ClassificationPlots, BasePlots)):
        raise AssertionError("Assertion failed.")
    if not (issubclass(RegressionPlots, BasePlots)):
        raise AssertionError("Assertion failed.")
    if not (issubclass(SHAPPlots, BasePlots)):
        raise AssertionError("Assertion failed.")


def test_evaluation_package_docstring() -> None:
    """Test that evaluation package has proper documentation."""
    if not (evaluation_module.__doc__ is not None):
        raise AssertionError("Assertion failed.")
    if not ("evaluation" in evaluation_module.__doc__.lower()):
        raise AssertionError("Assertion failed.")
    if not ("plotting" in evaluation_module.__doc__.lower()):
        raise AssertionError("Assertion failed.")


def test_evaluation_classes_have_docstrings() -> None:
    """Test that all evaluation classes have proper docstrings."""
    classes = [ClassificationPlots, RegressionPlots, SHAPPlots]
    for cls in classes:
        if not (cls.__doc__ is not None):
            raise AssertionError("Assertion failed.")
        if not (len(cls.__doc__.strip()) > 0):
            raise AssertionError("Assertion failed.")


def test_evaluation_classes_instantiation() -> None:
    """Test that all evaluation classes can be instantiated without errors."""

    classification_plots = ClassificationPlots()
    regression_plots = RegressionPlots()
    shap_plots = SHAPPlots()

    if not (hasattr(classification_plots, "default_template")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(regression_plots, "default_template")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(shap_plots, "default_template")):
        raise AssertionError("Assertion failed.")


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
        if not (hasattr(classification_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(classification_plots, method))):
            raise AssertionError("Assertion failed.")

    regression_methods = [
        "homoscedacity_plot",
        "true_vs_fitted_plot",
        "residual_distribution_plot",
    ]

    regression_plots = RegressionPlots()
    for method in regression_methods:
        if not (hasattr(regression_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(regression_plots, method))):
            raise AssertionError("Assertion failed.")

    shap_methods = [
        "plot_shap_bar",
        "plot_custom_shap_beeswarm",
        "plot_shap_categorical_box",
        "plot_shap_numerical_scatter",
        "plot_shap_force",
    ]

    shap_plots = SHAPPlots()
    for method in shap_methods:
        if not (hasattr(shap_plots, method)):
            raise AssertionError("Assertion failed.")
        if not (callable(getattr(shap_plots, method))):
            raise AssertionError("Assertion failed.")
