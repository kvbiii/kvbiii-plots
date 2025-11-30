import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from kvbiii_plots.evaluation.classification_plots import ClassificationPlots


class TestClassificationPlotsFixed:
    """Fixed test suite for ClassificationPlots class methods."""

    @pytest.fixture
    def binary_classification_data(self):
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
    def multiclass_classification_data(self):
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

    def test_plot_confusion_matrix_binary(self, binary_classification_data):
        """Test confusion matrix plot for binary classification."""
        plots = ClassificationPlots()
        data = binary_classification_data

        # Should create the plot without error
        plots.plot_confusion_matrix(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Binary Classification Confusion Matrix",
        )

    def test_plot_confusion_matrix_multiclass(self, multiclass_classification_data):
        """Test confusion matrix plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        # Test basic multiclass confusion matrix
        plots.plot_confusion_matrix(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification Confusion Matrix",
        )

    def test_plot_probabilities_per_class_binary(self, binary_classification_data):
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
        self, multiclass_classification_data
    ):
        """Test probability distribution plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_probabilities_per_class(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification Probabilities",
        )

    def test_plot_roc_auc_binary(self, binary_classification_data):
        """Test ROC AUC plot for binary classification."""
        plots = ClassificationPlots()
        data = binary_classification_data

        plots.plot_roc_auc(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Binary Classification ROC AUC",
        )

    def test_plot_roc_auc_multiclass(self, multiclass_classification_data):
        """Test ROC AUC plot for multiclass classification."""
        plots = ClassificationPlots()
        data = multiclass_classification_data

        plots.plot_roc_auc(
            y_true=data["y_true"],
            probabilities=data["probabilities"],
            id2label=data["id2label"],
            plot_title="Multiclass Classification ROC AUC",
        )

    def test_parameter_validation(self, binary_classification_data):
        """Test parameter validation across methods."""
        plots = ClassificationPlots()
        data = binary_classification_data

        # Test various parameter combinations
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
