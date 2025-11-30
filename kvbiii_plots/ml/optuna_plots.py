import numpy as np
import optuna
import plotly.graph_objects as go
import plotly.express as px
from ..base_plots import BasePlots


class OptunaPlots(BasePlots):
    """Class for creating plots specifically for Optuna hyperparameter optimization analysis.

    This class inherits from BasePlots and provides specialized methods
    for visualizing Optuna studies including optimization history, parameter importance,
    parallel coordinate plots, and parameter relationships.
    """

    def _extract_trial_data(
        self, study: optuna.study.Study
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts trial numbers and values from Optuna study.

        Args:
            study (optuna.study.Study): The Optuna study object.

        Returns:
            tuple[np.ndarray, np.ndarray]: Trial numbers and corresponding values.
        """
        scores = study.trials_dataframe()
        if len(scores) == 0:
            return np.array([]), np.array([])
        trial_numbers = np.arange(1, len(scores) + 1)
        values = scores["value"].values
        return trial_numbers, values

    def _compute_best_scores(self, values: np.ndarray, direction: str) -> np.ndarray:
        """Computes running best scores based on optimization direction.

        Args:
            values (np.ndarray): Array of trial values.
            direction (str): Optimization direction ("minimize" or "maximize").

        Returns:
            np.ndarray: Array of running best scores.
        """
        best_scores = []
        if direction == "maximize":
            for i in range(len(values)):
                best_scores.append(values[: i + 1].max())
        else:
            for i in range(len(values)):
                best_scores.append(values[: i + 1].min())
        return np.array(best_scores)

    def plot_optuna_optimization_history(
        self,
        study: optuna.study.Study,
        plot_title: str = "Optuna Optimization History",
        direction: str = "minimize",
        width: int = 1200,
        height: int = 800,
        xaxis_title: str = "Trial Number",
        yaxis_title: str = "Objective Value",
        marker_size: int = 25,
        marker_color: str = "blue",
        marker_opacity: float = 0.7,
        line_color: str = "red",
        line_width: int = 8,
        line_dash: str = "solid",
        show_trial_legend: bool = False,
        show_best_legend: bool = False,
        font_size: int = 20,
    ) -> None:
        """Plots the Optuna optimization history for a given study.

        Args:
            study (optuna.study.Study): The Optuna study object.
            plot_title (str, optional): Title for the plot. Defaults to "Optuna Optimization History".
            direction (str, optional): Optimization direction ("minimize" or "maximize"). Defaults to "minimize".
            width (int, optional): Width of the plot in pixels. Defaults to 1200.
            height (int, optional): Height of the plot in pixels. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Trial Number".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Objective Value".
            marker_size (int, optional): Size of trial markers. Defaults to 8.
            marker_color (str, optional): Color of trial markers. Defaults to "blue".
            marker_opacity (float, optional): Opacity of trial markers. Defaults to 0.7.
            line_color (str, optional): Color of best score line. Defaults to "red".
            line_width (int, optional): Width of best score line. Defaults to 3.
            line_dash (str, optional): Dash style for best score line. Defaults to "solid".
            show_trial_legend (bool, optional): Whether to show trial legend. Defaults to False.
            show_best_legend (bool, optional): Whether to show best score legend. Defaults to False.
            font_size (int, optional): Font size for text elements. Defaults to 20.
        """
        trial_numbers, values = self._extract_trial_data(study)

        # Handle empty study case
        if len(trial_numbers) == 0:
            fig = go.Figure()
            fig.update_layout(
                title=(
                    plot_title
                    if plot_title
                    else "Optuna Optimization History (No Trials)"
                ),
                xaxis_title="Trial Number",
                yaxis_title="Objective Value",
                width=width,
                height=height,
                font=dict(size=font_size),
            )
            return

        best_scores = self._compute_best_scores(values, direction)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=values,
                mode="markers",
                name="Trial Score",
                marker=dict(
                    size=marker_size, color=marker_color, opacity=marker_opacity
                ),
                showlegend=show_trial_legend,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=trial_numbers,
                y=best_scores,
                mode="lines",
                name="Best Score",
                line=dict(color=line_color, width=line_width, dash=line_dash),
                showlegend=show_best_legend,
            )
        )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_layout(
            font=dict(family="Times New Roman", size=font_size, color="Black"),
            showlegend=show_trial_legend or show_best_legend,
        )
        fig.show("png", width=width, height=height)

    def plot_optuna_param_importance(
        self,
        study: optuna.study.Study,
        plot_title: str = "Optuna Parameter Importance",
        width: int = 1200,
        height: int = 800,
        xaxis_title: str = "Hyperparameter",
        yaxis_title: str = "Importance",
        color_scale: str = "rainbow",
        use_qualitative_colors: bool | None = None,
        show_values: bool = True,
        value_precision: int = 3,
        bar_line_color: str = "black",
        bar_line_width: int = 1,
        font_size: int = 20,
        sort_descending: bool = True,
    ) -> None:
        """Plots Optuna hyperparameter importance as a bar plot.

        Args:
            study (optuna.study.Study): The Optuna study object.
            plot_title (str, optional): Title for the plot. Defaults to "Optuna Parameter Importance".
            width (int, optional): Width of the plot in pixels. Defaults to 1200.
            height (int, optional): Height of the plot in pixels. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Hyperparameter".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Importance".
            color_scale (str, optional): Plotly color scale name. Defaults to "rainbow".
            use_qualitative_colors (bool | None, optional): Whether to use qualitative colors. Defaults to None.
            show_values (bool, optional): Whether to display values on bars. Defaults to True.
            value_precision (int, optional): Decimal precision for displayed values. Defaults to 3.
            bar_line_color (str, optional): Color of bar outlines. Defaults to "black".
            bar_line_width (int, optional): Width of bar outlines. Defaults to 1.
            font_size (int, optional): Font size for text elements. Defaults to 20.
            sort_descending (bool, optional): Whether to sort parameters by importance descending. Defaults to True.
        """
        try:
            importances = optuna.importance.get_param_importances(study)
        except ValueError as e:
            if "Cannot evaluate parameter importances with only a single trial" in str(
                e
            ):
                raise ValueError(
                    "Cannot evaluate parameter importances with only a single trial."
                ) from e
            else:
                raise

        if not importances:
            raise ValueError(
                "Cannot evaluate parameter importances without completed trials."
            )

        param_names = list(importances.keys())
        importance_values = list(importances.values())

        if sort_descending:
            sorted_indices = np.argsort(importance_values)[::-1]
            param_names = [param_names[i] for i in sorted_indices]
            importance_values = [importance_values[i] for i in sorted_indices]

        n_colors = len(param_names)
        if use_qualitative_colors is None:
            use_qualitative_colors = n_colors <= 10

        if use_qualitative_colors:
            colors = px.colors.qualitative.Pastel[:n_colors]
        else:
            colors = px.colors.sample_colorscale(
                color_scale,
                [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)],
            )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=param_names,
                y=importance_values,
                marker=dict(
                    color=colors, line=dict(color=bar_line_color, width=bar_line_width)
                ),
                text=(
                    [f"{v:.{value_precision}f}" for v in importance_values]
                    if show_values
                    else None
                ),
                textposition="auto" if show_values else None,
                showlegend=False,
            )
        )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )

        fig.update_layout(
            font=dict(family="Times New Roman", size=font_size, color="Black"),
        )

        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import cross_val_score

    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: The mean cross-validated score of the model.
        """
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

        n_estimators = trial.suggest_int("n_estimators", 10, 100)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )

        scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_squared_error")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    optuna_plots = OptunaPlots()

    optuna_plots.plot_optuna_optimization_history(
        study=study,
        plot_title="Random Forest Optimization",
        direction="maximize",
        width=1000,
        height=600,
    )

    optuna_plots.plot_optuna_param_importance(
        study=study,
        plot_title="Hyperparameter Importance",
        width=1000,
        height=600,
        show_values=True,
        value_precision=3,
    )
