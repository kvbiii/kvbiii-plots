import matplotlib.pyplot as plt
import numpy as np
import pytest
import shap
from sklearn.datasets import make_classification, make_regression

from kvbiii_plots.evaluation.shap_plots import SHAPPlots


class TestSHAPPlots:
    """Test suite for SHAPPlots class methods."""

    def _create_mock_shap_explanation(
        self,
        values: np.ndarray,
        data: np.ndarray,
        feature_names: list[str],
        base_values: float | np.ndarray | None = None,
    ) -> shap.Explanation:
        """Create a mock shap.Explanation object for testing."""
        explanation = shap.Explanation(
            values=values,
            base_values=base_values if base_values is not None else 0.0,
            data=data,
            feature_names=feature_names,
        )
        return explanation

    @pytest.fixture
    def sample_classification_shap_data(self) -> dict[str, object]:
        """Generate sample SHAP data for classification."""

        X, y = make_classification(
            n_samples=100, n_features=8, n_classes=2, random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        shap_values = np.random.uniform(-1, 1, size=(X.shape[0], X.shape[1]))
        base_value = 0.0

        explanation = self._create_mock_shap_explanation(
            values=shap_values,
            data=X,
            feature_names=feature_names,
            base_values=base_value,
        )

        return {
            "shap_explanation": explanation,
            "shap_values": shap_values,
            "X": X,
            "feature_names": feature_names,
            "base_value": base_value,
            "y": y,
        }

    @pytest.fixture
    def sample_regression_shap_data(self) -> dict[str, object]:
        """Generate sample SHAP data for regression."""

        X, y = make_regression(n_samples=100, n_features=6, noise=0.1, random_state=42)
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        shap_values = np.random.uniform(-2, 2, size=(X.shape[0], X.shape[1]))
        base_value = np.mean(y)

        explanation = self._create_mock_shap_explanation(
            values=shap_values,
            data=X,
            feature_names=feature_names,
            base_values=base_value,
        )

        return {
            "shap_explanation": explanation,
            "shap_values": shap_values,
            "X": X,
            "feature_names": feature_names,
            "base_value": base_value,
            "y": y,
        }

    @pytest.fixture
    def sample_categorical_data(self) -> dict[str, object]:
        """Generate sample data with categorical features."""
        np.random.seed(42)
        n_samples = 80

        numerical_features = np.random.randn(n_samples, 3)
        categorical_feature = np.random.choice(["A", "B", "C"], size=n_samples)

        shap_values = np.random.uniform(-1, 1, size=(n_samples, 4))

        feature_names = ["num_1", "num_2", "num_3", "category"]

        cat_explanation = self._create_mock_shap_explanation(
            values=shap_values[:, 3],
            data=categorical_feature,
            feature_names=["category"],
        )

        return {
            "shap_values": shap_values,
            "numerical_features": numerical_features,
            "categorical_feature": categorical_feature,
            "categorical_explanation": cat_explanation,
            "feature_names": feature_names,
        }

    def test_shap_plots_initialization(self) -> None:
        """Test SHAPPlots class initialization."""
        shap_plots = SHAPPlots()
        if not (shap_plots is not None):
            raise AssertionError("Assertion failed.")
        if not (hasattr(shap_plots, "plot_shap_bar")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(shap_plots, "plot_custom_shap_beeswarm")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(shap_plots, "plot_shap_categorical_box")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(shap_plots, "plot_shap_numerical_scatter")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(shap_plots, "plot_shap_force")):
            raise AssertionError("Assertion failed.")

    def test_plot_shap_bar_basic(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test basic SHAP bar plot functionality."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        shap_plots.plot_shap_bar(
            shap_values=data["shap_explanation"], plot_title="Basic SHAP Bar Plot"
        )

    def test_plot_shap_bar_custom_parameters(
        self,
        sample_regression_shap_data: object,
    ) -> None:
        """Test SHAP bar plot with custom parameters."""
        shap_plots = SHAPPlots()
        data = sample_regression_shap_data

        shap_plots.plot_shap_bar(
            shap_values=data["shap_explanation"],
            plot_title="Custom SHAP Bar Plot",
            width=1200,
            height=700,
            top_n=4,
            color_scale="viridis",
            show_values=True,
            font_size=16,
        )

    def test_plot_shap_bar_single_instance(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test SHAP bar plot for single instance."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        single_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][0:1],
            data=data["X"][0:1],
            feature_names=data["feature_names"],
        )

        shap_plots.plot_shap_bar(
            shap_values=single_explanation,
            plot_title="Single Instance SHAP Bar Plot",
            top_n=5,
        )

    def test_plot_custom_shap_beeswarm_basic(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test basic custom SHAP beeswarm plot."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        shap_plots.plot_custom_shap_beeswarm(
            shap_values=data["shap_explanation"], plot_title="Basic SHAP Beeswarm Plot"
        )

    def test_plot_custom_shap_beeswarm_custom_parameters(
        self,
        sample_regression_shap_data: object,
    ) -> None:
        """Test custom SHAP beeswarm plot with advanced parameters."""
        shap_plots = SHAPPlots()
        data = sample_regression_shap_data

        shap_plots.plot_custom_shap_beeswarm(
            shap_values=data["shap_explanation"],
            plot_title="Custom SHAP Beeswarm Plot",
            plot_size=(14, 8),
            top_n=4,
            colormap="plasma",
            show_colorbar=True,
            font_size=14,
        )

    def test_plot_shap_categorical_box(self, sample_categorical_data: object) -> None:
        """Test SHAP categorical box plot."""
        shap_plots = SHAPPlots()
        data = sample_categorical_data

        shap_plots.plot_shap_categorical_box(
            scatter=data["categorical_explanation"],
            feature="Category Feature",
            plot_title="SHAP Categorical Box Plot",
        )

    def test_plot_shap_categorical_box_custom_parameters(
        self,
        sample_categorical_data: object,
    ) -> None:
        """Test SHAP categorical box plot with custom parameters."""
        shap_plots = SHAPPlots()
        data = sample_categorical_data

        shap_plots.plot_shap_categorical_box(
            scatter=data["categorical_explanation"],
            feature="Custom Category Feature",
            plot_title="Custom SHAP Categorical Box Plot",
            width=1000,
            height=600,
            color_scale=["red", "green", "blue"],
            font_size=16,
        )

    def test_plot_shap_numerical_scatter(
        self,
        sample_regression_shap_data: object,
    ) -> None:
        """Test SHAP numerical scatter plot."""
        shap_plots = SHAPPlots()
        data = sample_regression_shap_data

        single_feature_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][:, 0],
            data=data["X"][:, 0],
            feature_names=["feature_0"],
        )

        shap_plots.plot_shap_numerical_scatter(
            feature_shap_values=single_feature_explanation,
            feature="Numerical Feature 1",
            plot_title="SHAP Numerical Scatter Plot",
        )

    def test_plot_shap_numerical_scatter_custom_parameters(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test SHAP numerical scatter plot with custom parameters."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        single_feature_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][:, 1],
            data=data["X"][:, 1],
            feature_names=["feature_1"],
        )

        shap_plots.plot_shap_numerical_scatter(
            feature_shap_values=single_feature_explanation,
            feature="Custom Numerical Feature",
            plot_title="Custom SHAP Numerical Scatter Plot",
            width=1100,
            height=700,
            marker_size=10,
            colorscale="viridis",
            add_trendline=True,
            trendline_color="red",
            font_size=15,
        )

    def test_plot_shap_force_single_instance(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test SHAP force plot for single instance."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        single_obs_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][0],
            data=data["X"][0],
            feature_names=data["feature_names"],
            base_values=data["base_value"],
        )

        shap_plots.plot_shap_force(
            observation_shap_values=single_obs_explanation,
            contribution_threshold=0.05,
            figsize=(20, 4),
        )

    def test_plot_shap_force_custom_parameters(
        self,
        sample_regression_shap_data: object,
    ) -> None:
        """Test SHAP force plot with custom parameters."""
        shap_plots = SHAPPlots()
        data = sample_regression_shap_data

        single_obs_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][5],
            data=data["X"][5],
            feature_names=data["feature_names"],
            base_values=data["base_value"],
        )

        shap_plots.plot_shap_force(
            observation_shap_values=single_obs_explanation,
            contribution_threshold=0.07,
            figsize=(25, 5),
            font_size=12,
            text_rotation=45,
        )

    def test_error_handling_mismatched_dimensions(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test error handling for mismatched dimensions."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        with pytest.raises((ValueError, IndexError, TypeError)):
            invalid_explanation = self._create_mock_shap_explanation(
                values=data["shap_values"],
                data=data["X"][:, :2],
                feature_names=["feature_1", "feature_2"],
            )
            shap_plots.plot_shap_bar(shap_values=invalid_explanation)

    def test_error_handling_invalid_shap_values(self) -> None:
        """Test error handling for invalid SHAP values."""
        shap_plots = SHAPPlots()

        with pytest.raises((ValueError, IndexError, TypeError)):
            empty_explanation = self._create_mock_shap_explanation(
                values=np.array([]), data=np.array([]), feature_names=[]
            )
            shap_plots.plot_shap_bar(shap_values=empty_explanation)

    def test_error_handling_force_plot_dimensions(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test error handling for force plot dimension mismatches."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        with pytest.raises(
            (ValueError, IndexError, TypeError, shap.utils._exceptions.DimensionError)
        ):
            invalid_explanation = self._create_mock_shap_explanation(
                values=data["shap_values"][0],
                data=data["X"][0][:3],
                feature_names=data["feature_names"],
            )
            shap_plots.plot_shap_force(observation_shap_values=invalid_explanation)

    def test_parameter_validation_ranges(
        self,
        sample_regression_shap_data: object,
    ) -> None:
        """Test parameter validation for various methods."""
        shap_plots = SHAPPlots()
        data = sample_regression_shap_data

        shap_plots.plot_shap_bar(
            shap_values=data["shap_explanation"], width=800, height=500
        )

        shap_plots.plot_custom_shap_beeswarm(
            shap_values=data["shap_explanation"], top_n=3
        )

    def test_color_customization(
        self,
        sample_classification_shap_data: object,
    ) -> None:
        """Test color customization across different plot types."""
        shap_plots = SHAPPlots()
        data = sample_classification_shap_data

        shap_plots.plot_custom_shap_beeswarm(
            shap_values=data["shap_explanation"], colormap="plasma"
        )

        single_obs_explanation = self._create_mock_shap_explanation(
            values=data["shap_values"][0],
            data=data["X"][0],
            feature_names=data["feature_names"],
            base_values=data["base_value"],
        )

        shap_plots.plot_shap_force(
            observation_shap_values=single_obs_explanation, font_color="blue"
        )

    def test_edge_cases_small_datasets(self) -> None:
        """Test handling of small datasets."""
        shap_plots = SHAPPlots()

        small_shap = np.array([[0.1, -0.2]])
        small_x = np.array([[1.0, 2.0]])
        feature_names = ["feat1", "feat2"]

        small_explanation = self._create_mock_shap_explanation(
            values=small_shap, data=small_x, feature_names=feature_names
        )

        shap_plots.plot_shap_bar(
            shap_values=small_explanation, plot_title="Small Dataset Test"
        )

        shap_plots.plot_custom_shap_beeswarm(
            shap_values=small_explanation, plot_title="Small Beeswarm Test"
        )


class TestSHAPPlotsPrivateMethods:
    """Test suite for SHAPPlots private methods."""

    def _create_mock_shap_explanation(
        self,
        values: np.ndarray,
        data: np.ndarray,
        feature_names: list[str],
        base_values: float | np.ndarray | None = None,
    ) -> shap.Explanation:
        """Create a mock shap.Explanation object for testing."""
        explanation = shap.Explanation(
            values=values,
            base_values=base_values if base_values is not None else 0.0,
            data=data,
            feature_names=feature_names,
        )
        return explanation

    def test_compute_shap_importance(self) -> None:
        """Test _compute_shap_importance private method."""
        shap_plots = SHAPPlots()

        shap_values = np.array([[0.1, -0.2, 0.3], [-0.1, 0.4, -0.2], [0.2, -0.1, 0.1]])
        feature_names = ["feature1", "feature2", "feature3"]

        explanation = self._create_mock_shap_explanation(
            values=shap_values, data=np.random.randn(3, 3), feature_names=feature_names
        )

        _, names, importance = shap_plots._compute_shap_importance(
            explanation, top_n=20, class_id=0
        )

        if not (len(names) == 3):
            raise AssertionError("Assertion failed.")
        if not (len(importance) == 3):
            raise AssertionError("Assertion failed.")
        if not (all(isinstance(imp, (float, np.floating)) for imp in importance)):
            raise AssertionError("Assertion failed.")
        if not (isinstance(names, np.ndarray)):
            raise AssertionError("Assertion failed.")

        _, names_limited, importance_limited = shap_plots._compute_shap_importance(
            explanation, top_n=2, class_id=0
        )

        if not (len(names_limited) == 2):
            raise AssertionError("Assertion failed.")
        if not (len(importance_limited) == 2):
            raise AssertionError("Assertion failed.")

        if not (importance_limited[0] >= importance_limited[1]):
            raise AssertionError("Assertion failed.")

    def test_get_dynamic_colors(self) -> None:
        """Test _get_dynamic_colors private method."""
        shap_plots = SHAPPlots()

        colors_small = shap_plots._get_dynamic_colors(5, use_qualitative=True)
        if not (len(colors_small) == 5):
            raise AssertionError("Assertion failed.")
        if not (isinstance(colors_small, list)):
            raise AssertionError("Assertion failed.")

        colors_large = shap_plots._get_dynamic_colors(15, use_qualitative=False)
        if not (len(colors_large) == 15):
            raise AssertionError("Assertion failed.")
        if not (isinstance(colors_large, list)):
            raise AssertionError("Assertion failed.")

        colors_auto = shap_plots._get_dynamic_colors(8, use_qualitative=None)
        if not (len(colors_auto) == 8):
            raise AssertionError("Assertion failed.")

    def test_clean_data_for_plotting(self) -> None:
        """Test _clean_data_for_plotting private method."""
        shap_plots = SHAPPlots()

        feature_data = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        shap_values = np.array([0.1, 0.2, np.nan, 0.4, 0.5])

        cleaned_data, cleaned_shap = shap_plots._clean_data_for_plotting(
            feature_data, shap_values
        )

        if not (len(cleaned_data) <= len(feature_data)):
            raise AssertionError("Assertion failed.")
        if not (len(cleaned_shap) == len(cleaned_data)):
            raise AssertionError("Assertion failed.")
        if not (not np.any(np.isnan(cleaned_data))):
            raise AssertionError("Assertion failed.")
        if not (not np.any(np.isnan(cleaned_shap))):
            raise AssertionError("Assertion failed.")

    def test_setup_matplotlib_styling(self) -> None:
        """Test _setup_matplotlib_styling private method."""
        shap_plots = SHAPPlots()

        fig, _ = plt.subplots(figsize=(6, 4))

        shap_plots._setup_matplotlib_styling(
            xlabel="X Label", ylabel="Y Label", font_size=12
        )

        current_ax = plt.gca()
        if not (current_ax.get_xlabel() == "X Label"):
            raise AssertionError("Assertion failed.")
        if not (current_ax.get_ylabel() == "Y Label"):
            raise AssertionError("Assertion failed.")

        plt.close(fig)

    def test_create_custom_colorbar(self) -> None:
        """Test _create_custom_colorbar private method."""
        shap_plots = SHAPPlots()

        fig, ax = plt.subplots(figsize=(6, 4))

        shap_plots._create_custom_colorbar(
            ax=ax, colormap="coolwarm", label="Test Label"
        )
        colorbar = ax.figure.colorbar(
            plt.cm.ScalarMappable(cmap="coolwarm"), ax=ax, label="Test Label"
        )
        if not (colorbar is not None):
            raise AssertionError("Assertion failed.")
        if not (colorbar.ax.get_ylabel() == "Test Label"):
            raise AssertionError("Assertion failed.")
        if not (colorbar.ax.get_yticklabels() is not None):
            raise AssertionError("Assertion failed.")
        if not (len(colorbar.ax.get_yticklabels()) > 0):
            raise AssertionError("Assertion failed.")
        if not (colorbar.ax.get_yticklabels()[0].get_text() == "0.0"):
            raise AssertionError("Assertion failed.")

        plt.close(fig)

    def test_private_methods_integration(self) -> None:
        """Test integration of private methods with realistic data."""
        shap_plots = SHAPPlots()

        n_samples, n_features = 50, 4
        shap_values = np.random.uniform(-1, 1, size=(n_samples, n_features))
        X = np.random.uniform(0, 10, size=(n_samples, n_features))
        feature_names = [f"Feature_{i}" for i in range(n_features)]

        explanation = self._create_mock_shap_explanation(
            values=shap_values, data=X, feature_names=feature_names
        )

        _, names, importance = shap_plots._compute_shap_importance(
            explanation, top_n=3, class_id=0
        )

        if not (len(names) == 3):
            raise AssertionError("Assertion failed.")
        if not (len(importance) == 3):
            raise AssertionError("Assertion failed.")
        if not (all(imp >= 0 for imp in importance)):
            raise AssertionError("Assertion failed.")

        colors = shap_plots._get_dynamic_colors(len(importance))
        if not (len(colors) == len(importance)):
            raise AssertionError("Assertion failed.")

        cleaned_data, cleaned_shap = shap_plots._clean_data_for_plotting(
            X[:, 0], shap_values[:, 0]
        )
        if not (len(cleaned_data) <= len(X[:, 0])):
            raise AssertionError("Assertion failed.")
        if not (len(cleaned_shap) == len(cleaned_data)):
            raise AssertionError("Assertion failed.")

    def test_edge_cases_private_methods(self) -> None:
        """Test edge cases for private methods."""
        shap_plots = SHAPPlots()

        single_shap = np.array([[0.5], [-0.3], [0.1]])
        single_x = np.array([[1.0], [2.0], [3.0]])
        single_features = ["only_feature"]

        single_explanation = self._create_mock_shap_explanation(
            values=single_shap, data=single_x, feature_names=single_features
        )

        _, names, importance = shap_plots._compute_shap_importance(
            single_explanation, top_n=1, class_id=0
        )

        if not (len(names) == 1):
            raise AssertionError("Assertion failed.")
        if not (len(importance) == 1):
            raise AssertionError("Assertion failed.")

        colors_zero = shap_plots._get_dynamic_colors(0)
        if not (len(colors_zero) == 0):
            raise AssertionError("Assertion failed.")

        colors_one = shap_plots._get_dynamic_colors(1)
        if not (len(colors_one) == 1):
            raise AssertionError("Assertion failed.")
