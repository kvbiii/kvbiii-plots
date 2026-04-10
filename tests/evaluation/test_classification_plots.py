import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from kvbiii_plots.evaluation.classification_plots import ClassificationPlots


class TestClassificationPlotsFixed:
    """Fixed test suite for ClassificationPlots class methods."""

    @pytest.fixture
    def binary_classification_data(self) -> dict[str, object]:
        """Generate sample binary classification data for testing."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)

        return {
            "y_true": y_test,
            "probabilities": probabilities,
            "id2label": {0: "Class_0", 1: "Class_1"},
            "model": model,
            "X_test": X_test,
        }

    @pytest.fixture
    def multiclass_classification_data(self) -> dict[str, object]:
        """Generate sample multiclass classification data for testing."""
        X, y = make_classification(
            n_samples=300,
            n_features=15,
            n_classes=3,
            n_redundant=0,
            n_informative=5,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)

        return {
            "y_true": y_test,
            "probabilities": probabilities,
            "id2label": {0: "Class_0", 1: "Class_1", 2: "Class_2"},
            "model": model,
            "X_test": X_test,
        }

    def test_plot_confusion_matrix_binary(
        self,
        binary_classification_data: object,
    ) -> None:
        """Test confusion matrix plot for binary classification."""
        plots = ClassificationPlots()
        data = binary_classification_data

        plots.plot_confusion_matrix(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Binary Classification Confusion Matrix",
        )

    def test_plot_confusion_matrix_multiclass(
        self,
        multiclass_classification_data: object,
    ) -> None:
        """Test confusion matrix plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_confusion_matrix(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification Confusion Matrix",
        )

    def test_plot_probabilities_per_class_binary(
        self,
        binary_classification_data: object,
    ) -> None:
        """Test probability distribution plot for binary classification."""
        plots = ClassificationPlots()
        data = binary_classification_data

        plots.plot_probabilities_per_class(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Binary Classification Probabilities",
        )

    def test_plot_probabilities_per_class_multiclass(
        self,
        multiclass_classification_data: object,
    ) -> None:
        """Test probability distribution plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_probabilities_per_class(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification Probabilities",
        )

    def test_plot_probabilities_histogram_overlay(
        self,
        multiclass_classification_data: object,
    ) -> None:
        """Test overlaid probability histogram across classes."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_probabilities_histogram(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            alpha=True,
            opacity=0.5,
            bins=30,
            plot_title="Overlay Probability Histogram",
        )

    def test_plot_probabilities_histogram_separate(
        self,
        multiclass_classification_data: object,
    ) -> None:
        """Test per-class probability histograms generated sequentially."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_probabilities_histogram(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            alpha=False,
            opacity=0.6,
            bins=25,
            plot_title="Sequential Probability Histograms",
        )

    def test_plot_probabilities_histogram_validation(
        self,
        binary_classification_data: object,
    ) -> None:
        """Test validation for histogram plotting arguments."""
        plots = ClassificationPlots()
        data = binary_classification_data

        with pytest.raises(TypeError, match="alpha must be bool"):
            plots.plot_probabilities_histogram(
                y_true=data["y_true"],
                probabilities=data["probabilities"],
                id2label=data["id2label"],
                alpha="yes",
            )

        with pytest.raises(ValueError, match="opacity must be in the range"):
            plots.plot_probabilities_histogram(
                y_true=data["y_true"],
                probabilities=data["probabilities"],
                id2label=data["id2label"],
                opacity=0,
            )

        with pytest.raises(ValueError, match="bins must be greater than 0"):
            plots.plot_probabilities_histogram(
                y_true=data["y_true"],
                probabilities=data["probabilities"],
                id2label=data["id2label"],
                bins=0,
            )

    def test_plot_roc_auc_binary(self, binary_classification_data: object) -> None:
        """Test ROC AUC plot for binary classification."""
        plots = ClassificationPlots()
        data = binary_classification_data

        plots.plot_roc_auc(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Binary Classification ROC AUC",
        )

    def test_plot_roc_auc_multiclass(
        self,
        multiclass_classification_data: object,
    ) -> None:
        """Test ROC AUC plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_roc_auc(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification ROC AUC",
        )

    def test_parameter_validation(self, binary_classification_data: object) -> None:
        """Test parameter validation across methods."""
        plots = ClassificationPlots()
        data = binary_classification_data

        plots.plot_confusion_matrix(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            normalize=True,
            colorscale="viridis",
            font_size=18,
        )

        plots.plot_probabilities_per_class(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            width=1200,
            height=600,
            threshold_color="blue",
        )

        plots.plot_roc_auc(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            width=1000,
            height=1000,
            line_width=3,
        )
