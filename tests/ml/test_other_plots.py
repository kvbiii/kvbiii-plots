import numpy as np
import pytest
from sklearn.model_selection import KFold

from kvbiii_plots.ml.other_plots import OtherPlots


class TestMLOtherPlots:
    """Smoke tests for public methods in ml.other_plots.OtherPlots.

    We validate that methods accept inputs and run without raising, using small synthetic data.
    """

    @pytest.fixture
    def simple_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a simple synthetic dataset for testing.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing feature matrix X and target vector y.
        """
        x_data = np.random.rand(30, 3)
        y_data = (np.random.rand(30) > 0.5).astype(int)
        return x_data, y_data

    def test_cross_validation_split_plot_runs(
        self, simple_xy: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Test that cross-validation split plot runs without errors.

        Args:
            simple_xy (tuple[np.ndarray, np.ndarray]): A tuple containing
                feature matrix X and target vector y.
        """
        x_data, y_data = simple_xy
        cv = KFold(n_splits=3, shuffle=True, random_state=42)

        op = OtherPlots()
        op.cross_validation_split_plot(
            x_data,
            y_data,
            cv_splitter=cv,
            plot_title="CV Split Test",
            width=600,
            height=400,
        )

        assert True

    def test_anova_comparison_plot_runs(self) -> None:
        """Test that ANOVA comparison plot runs without errors."""
        rng = np.random.default_rng(42)
        x_groups = np.array(["A", "B", "A", "C", "B"] * 5)
        y_values = rng.normal(loc=0, scale=1, size=x_groups.shape[0])

        op = OtherPlots()
        op.anova_comparison_plot(
            x_groups,
            y_values,
            feature_name="grp",
            plot_title="ANOVA Box",
            width=600,
            height=400,
        )

        assert True

    def test_feature_ranking_scatter_plot_runs(self) -> None:
        """Test that feature ranking scatter plot runs without errors."""

        features = [f"f{i}" for i in range(1, 8)]
        spearman = {f: i for i, f in enumerate(features, start=1)}
        hoeffding = {f: i for i, f in enumerate(reversed(features), start=1)}

        op = OtherPlots()
        op.feature_ranking_scatter_plot(
            spearman_ranks=spearman,
            hoeffding_ranks=hoeffding,
            quantile=0.6,
            plot_title="Ranks",
            width=500,
            height=400,
        )

        assert True
