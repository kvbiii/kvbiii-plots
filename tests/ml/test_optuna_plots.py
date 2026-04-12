import optuna
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from kvbiii_plots.ml.optuna_plots import OptunaPlots


class TestOptunaPlots:
    """Tests for Optuna plotting utilities."""

    @pytest.fixture
    def sample_classification_study(self) -> optuna.Study:
        """Create a sample Optuna study for classification."""

        def objective(trial: optuna.Trial) -> float:

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

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)

        return study

    @pytest.fixture
    def sample_regression_study(self) -> optuna.Study:
        """Create a sample Optuna study for regression."""

        def objective(trial: optuna.Trial) -> float:

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

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)

        return study

    @pytest.fixture
    def sample_minimization_study(self) -> optuna.Study:
        """Create a sample Optuna study for minimization."""

        def objective(trial: optuna.Trial) -> float:

            x = trial.suggest_float("x", -5, 5)
            y = trial.suggest_float("y", -5, 5)
            return x**2 + y**2

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=15)

        return study

    @pytest.fixture
    def empty_study(self) -> optuna.Study:
        """Create an empty Optuna study for error testing."""
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        return study

    def test_optuna_plots_initialization(self) -> None:
        """Test OptunaPlots class initialization."""
        optuna_plots = OptunaPlots()
        if not (optuna_plots is not None):
            raise AssertionError("Assertion failed.")
        if not (hasattr(optuna_plots, "plot_optuna_optimization_history")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(optuna_plots, "plot_optuna_param_importance")):
            raise AssertionError("Assertion failed.")

        if not (hasattr(optuna_plots, "_extract_trial_data")):
            raise AssertionError("Assertion failed.")
        if not (hasattr(optuna_plots, "_compute_best_scores")):
            raise AssertionError("Assertion failed.")

    def test_extract_trial_data(self, sample_classification_study: object) -> None:
        """Test _extract_trial_data private method."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        trial_numbers, values = optuna_plots._extract_trial_data(study)

        if not (len(trial_numbers) == len(study.trials)):
            raise AssertionError("Assertion failed.")
        if not (len(values) == len(study.trials)):
            raise AssertionError("Assertion failed.")
        if not (trial_numbers[0] == 1):
            raise AssertionError("Assertion failed.")
        if not (trial_numbers[-1] == len(study.trials)):
            raise AssertionError("Assertion failed.")

    def test_compute_best_scores_maximize(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test _compute_best_scores for maximization problems."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        _, values = optuna_plots._extract_trial_data(study)
        best_scores = optuna_plots._compute_best_scores(values, "maximize")

        if not (len(best_scores) == len(values)):
            raise AssertionError("Assertion failed.")

        for i in range(1, len(best_scores)):
            if not (best_scores[i] >= best_scores[i - 1]):
                raise AssertionError("Assertion failed.")

    def test_compute_best_scores_minimize(
        self,
        sample_minimization_study: object,
    ) -> None:
        """Test _compute_best_scores for minimization problems."""
        optuna_plots = OptunaPlots()
        study = sample_minimization_study

        _, values = optuna_plots._extract_trial_data(study)
        best_scores = optuna_plots._compute_best_scores(values, "minimize")

        if not (len(best_scores) == len(values)):
            raise AssertionError("Assertion failed.")

        for i in range(1, len(best_scores)):
            if not (best_scores[i] <= best_scores[i - 1]):
                raise AssertionError("Assertion failed.")

    def test_plot_optuna_optimization_history_basic(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test basic optimization history plot."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        optuna_plots.plot_optuna_optimization_history(
            study=study, plot_title="Basic Optimization History", direction="maximize"
        )

    def test_plot_optuna_optimization_history_custom_parameters(
        self,
        sample_regression_study: object,
    ) -> None:
        """Test optimization history plot with custom parameters."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

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

    def test_plot_optuna_optimization_history_minimize(
        self,
        sample_minimization_study: object,
    ) -> None:
        """Test optimization history plot for minimization."""
        optuna_plots = OptunaPlots()
        study = sample_minimization_study

        optuna_plots.plot_optuna_optimization_history(
            study=study,
            plot_title="Minimization Optimization History",
            direction="minimize",
            marker_color="green",
            line_color="orange",
        )

    def test_plot_optuna_param_importance_basic(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test basic parameter importance plot."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        optuna_plots.plot_optuna_param_importance(
            study=study, plot_title="Basic Parameter Importance"
        )

    def test_plot_optuna_param_importance_custom_parameters(
        self,
        sample_regression_study: object,
    ) -> None:
        """Test parameter importance plot with custom parameters."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

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
        self,
        sample_regression_study: object,
    ) -> None:
        """Test parameter importance with continuous color scale."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        optuna_plots.plot_optuna_param_importance(
            study=study,
            plot_title="Continuous Colors Parameter Importance",
            color_scale="viridis",
            use_qualitative_colors=False,
            show_values=True,
            value_precision=2,
        )

    def test_plot_optuna_param_importance_no_values(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test parameter importance without showing values."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        optuna_plots.plot_optuna_param_importance(
            study=study,
            plot_title="No Values Parameter Importance",
            show_values=False,
            bar_line_width=0,
        )

    def test_error_handling_empty_study(self, empty_study: object) -> None:
        """Test error handling with empty study."""
        optuna_plots = OptunaPlots()
        study = empty_study

        optuna_plots.plot_optuna_optimization_history(study=study)

        with pytest.raises(ValueError) as excinfo:
            optuna_plots.plot_optuna_param_importance(study=study)
        if not (
            "Cannot evaluate parameter importances without completed trials"
            in str(excinfo.value)
        ):
            raise AssertionError("Assertion failed.")

    def test_parameter_validation_dimensions(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test parameter validation for plot dimensions."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        optuna_plots.plot_optuna_optimization_history(
            study=study, width=800, height=500, font_size=10
        )

        optuna_plots.plot_optuna_param_importance(
            study=study, width=600, height=400, font_size=12
        )

    def test_parameter_validation_colors(
        self,
        sample_regression_study: object,
    ) -> None:
        """Test parameter validation for color options."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        optuna_plots.plot_optuna_optimization_history(
            study=study, marker_color="purple", line_color="cyan", marker_opacity=0.5
        )

        optuna_plots.plot_optuna_param_importance(
            study=study, color_scale="rainbow", bar_line_color="black"
        )

    def test_edge_cases_single_trial(self) -> None:
        """Test handling of studies with single trial."""

        def single_objective(trial: optuna.Trial) -> float:
            x = trial.suggest_float("x", 0, 1)
            return x**2

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(single_objective, n_trials=1)

        optuna_plots = OptunaPlots()

        optuna_plots.plot_optuna_optimization_history(study=study, direction="minimize")

        with pytest.raises(ValueError) as excinfo:
            optuna_plots.plot_optuna_param_importance(study=study)

        if not (
            "Cannot evaluate parameter importances with only a single trial"
            in str(excinfo.value)
        ):
            raise AssertionError("Assertion failed.")

    def test_direction_parameter_validation(
        self,
        sample_classification_study: object,
    ) -> None:
        """Test direction parameter validation."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

        optuna_plots.plot_optuna_optimization_history(study=study, direction="maximize")

        optuna_plots.plot_optuna_optimization_history(study=study, direction="minimize")

    def test_line_dash_options(self, sample_regression_study: object) -> None:
        """Test different line dash options."""
        optuna_plots = OptunaPlots()
        study = sample_regression_study

        dash_styles = ["solid", "dash", "dot", "dashdot"]

        for style in dash_styles:
            optuna_plots.plot_optuna_optimization_history(
                study=study,
                plot_title=f"Line Dash: {style}",
                line_dash=style,
                direction="maximize",
            )

    def test_legend_options(self, sample_classification_study: object) -> None:
        """Test different legend display options."""
        optuna_plots = OptunaPlots()
        study = sample_classification_study

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
