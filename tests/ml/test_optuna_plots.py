import pytest
import optuna
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from kvbiii_plots.ml.optuna_plots import OptunaPlots


class TestOptunaPlots:

    @pytest.fixture
    def sample_classification_study(self):
        """Create a sample Optuna study for classification."""

        def objective(trial):
            # Simple classification objective
            x_data, y_data = make_classification(
                n_samples=100, n_features=5, random_state=42
            )

            n_estimators = trial.suggest_int("n_estimators", 10, 50)
            max_depth = trial.suggest_int("max_depth", 3, 8)

            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42
            )

            scores = cross_val_score(model, x_data, y_data, cv=3, scoring="accuracy")
            return scores.mean()

        # Suppress Optuna logging for cleaner test output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        return study

    @pytest.fixture
    def sample_regression_study(self):
        """Create a sample Optuna study for regression."""

        def objective(trial):
            # Simple regression objective
            x_data, y_data = make_regression(
                n_samples=100, n_features=4, noise=0.1, random_state=42
            )

            n_estimators = trial.suggest_int("n_estimators", 10, 50)
            max_depth = trial.suggest_int("max_depth", 3, 8)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
            )

            scores = cross_val_score(
                model, x_data, y_data, cv=3, scoring="neg_mean_squared_error"
            )
            return scores.mean()

        # Suppress Optuna logging for cleaner test output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)

        return study

    @pytest.fixture
    def sample_minimization_study(self):
        """Create a sample Optuna study for minimization."""

        def objective(trial):
            # Simple mathematical function to minimize
            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_float("y", -5, 5)
            return x**2 + y**2

        # Suppress Optuna logging for cleaner test output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)

        return study

    @pytest.fixture
    def empty_study(self):
        """Create an empty Optuna study for error testing."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        return study

    def test_optuna_plots_initialization(self):
        """Test OptunaPlots class initialization."""
        optuna_plots = OptunaPlots()
        assert optuna_plots is not None
        assert hasattr(optuna_plots, "plot_optuna_optimization_history")
        assert hasattr(optuna_plots, "plot_optuna_param_importance")

        # Check for private helper methods
        assert hasattr(optuna_plots, "_extract_trial_data")
        assert hasattr(optuna_plots, "_compute_best_scores")

    def test_extract_trial_data(self, sample_classification_study):
        """Test _extract_trial_data private method."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        trial_numbers, values = optuna_plots._extract_trial_data(study)

        assert len(trial_numbers) == len(study.trials)
        assert len(values) == len(study.trials)
        assert trial_numbers[0] == 1
        assert trial_numbers[-1] == len(study.trials)

    def test_compute_best_scores_maximize(self, sample_classification_study):
        """Test _compute_best_scores for maximization problems."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        _, values = optuna_plots._extract_trial_data(study)
        best_scores = optuna_plots._compute_best_scores(values, "maximize")

        assert len(best_scores) == len(values)
        # Check that best scores are non-decreasing for maximization
        for i in range(1, len(best_scores)):
            assert best_scores[i] >= best_scores[i - 1]

    def test_compute_best_scores_minimize(self, sample_minimization_study):
        """Test _compute_best_scores for minimization problems."""
        optuna_plots = OptunaPlots()
        study = sample_minimization_study

        _, values = optuna_plots._extract_trial_data(study)
        best_scores = optuna_plots._compute_best_scores(values, "minimize")

        assert len(best_scores) == len(values)
        # Check that best scores are non-increasing for minimization
        for i in range(1, len(best_scores)):
            assert best_scores[i] <= best_scores[i - 1]

    def test_plot_optuna_optimization_history_basic(self, sample_classification_study):
        """Test basic optimization history plot."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test basic plot
        optuna_plots.plot_optuna_optimization_history(
            study=study, plot_title="Basic Optimization History", direction="maximize"
        )

    def test_plot_optuna_optimization_history_custom_parameters(
        self, sample_regression_study
    ):
        """Test optimization history plot with custom parameters."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        # Test with custom parameters
        optuna_plots.plot_optuna_optimization_history(
            study=study,
            plot_title="Custom Optimization History",
            width=1400,
            height=900,
            xaxis_title="Trial Index",
            yaxis_title="Score Value",
            direction="maximize",
            marker_size=10,
            marker_color="red",
            marker_opacity=0.8,
            line_color="blue",
            line_width=4,
            line_dash="dash",
            show_trial_legend=True,
            show_best_legend=True,
            font_size=18,
        )

    def test_plot_optuna_optimization_history_minimize(self, sample_minimization_study):
        """Test optimization history plot for minimization."""
        optuna_plots = OptunaPlots()
        study = sample_minimization_study

        # Test minimization direction
        optuna_plots.plot_optuna_optimization_history(
            study=study,
            plot_title="Minimization Optimization History",
            direction="minimize",
            marker_color="green",
            line_color="orange",
        )

    def test_plot_optuna_param_importance_basic(self, sample_classification_study):
        """Test basic parameter importance plot."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test basic parameter importance
        optuna_plots.plot_optuna_param_importance(
            study=study, plot_title="Basic Parameter Importance"
        )

    def test_plot_optuna_param_importance_custom_parameters(
        self, sample_regression_study
    ):
        """Test parameter importance plot with custom parameters."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        # Test with custom parameters
        optuna_plots.plot_optuna_param_importance(
            study=study,
            plot_title="Custom Parameter Importance",
            width=1300,
            height=750,
            xaxis_title="Hyperparameters",
            yaxis_title="Relative Importance",
            color_scale="plasma",
            use_qualitative_colors=True,
            show_values=True,
            value_precision=4,
            bar_line_color="red",
            bar_line_width=2,
            font_size=16,
            sort_descending=True,
        )

    def test_plot_optuna_param_importance_continuous_colors(
        self, sample_regression_study
    ):
        """Test parameter importance with continuous color scale."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        # Test with continuous colors
        optuna_plots.plot_optuna_param_importance(
            study=study,
            plot_title="Continuous Colors Parameter Importance",
            color_scale="viridis",
            use_qualitative_colors=False,
            show_values=True,
            value_precision=2,
        )

    def test_plot_optuna_param_importance_no_values(self, sample_classification_study):
        """Test parameter importance without showing values."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test without showing values
        optuna_plots.plot_optuna_param_importance(
            study=study,
            plot_title="No Values Parameter Importance",
            show_values=False,
            bar_line_width=0,
        )

    def test_error_handling_empty_study(self, empty_study):
        """Test error handling with empty study."""
        optuna_plots = OptunaPlots()
        study = empty_study

        # Test optimization history with empty study - should handle gracefully
        optuna_plots.plot_optuna_optimization_history(study=study)

        # Test parameter importance with empty study
        with pytest.raises(ValueError) as excinfo:
            optuna_plots.plot_optuna_param_importance(study=study)
        assert "Cannot evaluate parameter importances without completed trials" in str(
            excinfo.value
        )

    def test_parameter_validation_dimensions(self, sample_classification_study):
        """Test parameter validation for plot dimensions."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test with various dimension parameters
        optuna_plots.plot_optuna_optimization_history(
            study=study, width=800, height=500, font_size=10
        )

        optuna_plots.plot_optuna_param_importance(
            study=study, width=600, height=400, font_size=12
        )

    def test_parameter_validation_colors(self, sample_regression_study):
        """Test parameter validation for color options."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        # Test with various color parameters
        optuna_plots.plot_optuna_optimization_history(
            study=study, marker_color="purple", line_color="cyan", marker_opacity=0.5
        )

        optuna_plots.plot_optuna_param_importance(
            study=study, color_scale="rainbow", bar_line_color="black"
        )

    def test_edge_cases_single_trial(self):
        """Test handling of studies with single trial."""

        def single_objective(trial):
            x = trial.suggest_float("x", 0, 1)
            return x**2

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(single_objective, n_trials=1)

        optuna_plots = OptunaPlots()

        # Test with single trial - optimization history should work
        optuna_plots.plot_optuna_optimization_history(study=study, direction="minimize")

        # Parameter importance should fail with single trial - test exception handling
        with pytest.raises(ValueError) as excinfo:
            optuna_plots.plot_optuna_param_importance(study=study)

        # Verify correct error message
        assert "Cannot evaluate parameter importances with only a single trial" in str(
            excinfo.value
        )

    def test_direction_parameter_validation(self, sample_classification_study):
        """Test direction parameter validation."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test both direction options
        optuna_plots.plot_optuna_optimization_history(study=study, direction="maximize")

        optuna_plots.plot_optuna_optimization_history(study=study, direction="minimize")

    def test_line_dash_options(self, sample_regression_study):
        """Test different line dash options."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        # Test different line dash styles
        dash_styles = ["solid", "dash", "dot", "dashdot"]

        for style in dash_styles:
            optuna_plots.plot_optuna_optimization_history(
                study=study,
                plot_title=f"Line Dash: {style}",
                line_dash=style,
                direction="maximize",
            )

    def test_legend_options(self, sample_classification_study):
        """Test different legend display options."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        # Test different legend combinations
        legend_combinations = [
            (True, True),
            (True, False),
            (False, True),
            (False, False),
        ]

        for trial_legend, best_legend in legend_combinations:
            optuna_plots.plot_optuna_optimization_history(
                study=study,
                show_trial_legend=trial_legend,
                show_best_legend=best_legend,
                direction="maximize",
            )
