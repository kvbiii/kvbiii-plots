import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from ..base_plots import BasePlots


class RegressionPlots(BasePlots):
    """Class for regression-specific plotting functionality.

    This class inherits from BasePlots and provides methods for visualizing
    regression model performance, including homoscedacity and true vs fitted plots.
    All methods follow the enhanced parameterization pattern from CategoricalPlots.
    """

    def __init__(self) -> None:
        """Initialize the RegressionPlots class with regression-specific configurations."""
        super().__init__()
        self.default_regression_colors = {
            "scatter": "rgba(0, 123, 255, 0.7)",
            "line": "rgba(220, 53, 69, 0.85)",
            "trendline": "rgba(40, 167, 69, 0.85)",
            "residual_line": "rgba(220, 53, 69, 0.85)",
        }

    def _validate_regression_inputs(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validates and converts regression inputs to NumPy arrays.

        Args:
            y_true (np.ndarray | pd.Series | list): True values of the target variable.
            y_pred (np.ndarray | pd.Series | list): Predicted values of the target variable.

        Returns:
            tuple[np.ndarray, np.ndarray]: Validated arrays for y_true and y_pred.

        Raises:
            ValueError: If arrays have different lengths or contain invalid data.
        """
        y_true = self.check_data(y_true)
        y_pred = self.check_data(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")

        return y_true, y_pred

    def _calculate_metric_value(
        self, y_true: np.ndarray, y_pred: np.ndarray, metric_name: str
    ) -> float:
        """Calculate the specified metric value.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            metric_name (str): Name of the metric to calculate.

        Returns:
            float: Calculated metric value.

        Raises:
            ValueError: If metric is not supported.
        """
        if metric_name.upper() not in self.metrics_dict:
            raise ValueError(
                f"Unsupported metric: {metric_name}. Available metrics: "
                f"{list(self.metrics_dict.keys())}"
            )

        return self.metrics_dict[metric_name.upper()](y_true, y_pred)

    def _create_scatter_trace(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        color: str | None = None,
        size: int = 8,
        opacity: float = 0.6,
        name: str = "",
        show_legend: bool = False,
    ) -> go.Scatter:
        """Create a reusable scatter trace for regression plots.

        Args:
            x_data (np.ndarray): X-axis data.
            y_data (np.ndarray): Y-axis data.
            color (str | None, optional): Marker color. Defaults to None.
            size (int, optional): Marker size. Defaults to 8.
            opacity (float, optional): Marker opacity. Defaults to 0.6.
            name (str, optional): Trace name. Defaults to "".
            show_legend (bool, optional): Whether to show in legend. Defaults to False.

        Returns:
            go.Scatter: Configured scatter trace.
        """
        if color is None:
            color = self.default_regression_colors["scatter"]

        return go.Scatter(
            x=x_data,
            y=y_data,
            mode="markers",
            marker=dict(
                color=color,
                size=size,
                opacity=opacity,
                line=dict(color="black", width=1),
            ),
            name=name,
            showlegend=show_legend,
        )

    def _add_reference_line(
        self,
        fig: go.Figure,
        line_type: str = "horizontal",
        value: float = 0,
        color: str | None = None,
        width: int = 2,
        opacity: float = 0.8,
        dash: str = "dash",
    ) -> None:
        """Add a reference line to the plot.

        Args:
            fig (go.Figure): The plotly figure to add the line to.
            line_type (str, optional): Type of line ("horizontal", "vertical", "diagonal").
            Defaults to "horizontal".
            value (float, optional): Value for horizontal/vertical lines. Defaults to 0.
            color (str | None, optional): Line color. Defaults to None.
            width (int, optional): Line width. Defaults to 2.
            opacity (float, optional): Line opacity. Defaults to 0.8.
            dash (str, optional): Line dash style. Defaults to "dash".
        """
        if color is None:
            color = self.default_regression_colors["line"]

        if line_type == "horizontal":
            fig.add_hline(
                y=value,
                line_color=color,
                line_width=width,
                opacity=opacity,
                line_dash=dash,
            )
        elif line_type == "vertical":
            fig.add_vline(
                x=value,
                line_color=color,
                line_width=width,
                opacity=opacity,
                line_dash=dash,
            )
        elif line_type == "diagonal":
            min_val = min(fig.data[0].x.min(), fig.data[0].y.min())
            max_val = max(fig.data[0].x.max(), fig.data[0].y.max())
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color=color, width=width, dash=dash),
                    opacity=opacity,
                    showlegend=False,
                    name="Perfect Prediction",
                )
            )

    def homoscedacity_plot(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        plot_title: str = "",
        width: int = 1000,
        height: int = 1000,
        xaxis_title: str = "Fitted Values",
        yaxis_title: str = "Residuals",
        metric_name: str = "RMSE",
        metric_value: float | None = None,
        scatter_color: str | None = None,
        scatter_size: int = 8,
        scatter_opacity: float = 0.6,
        reference_line_color: str | None = None,
        reference_line_width: int = 4,
        show_trendline: bool = False,
        trendline_color: str | None = None,
    ) -> None:
        """Creates a homoscedacity plot to visualize residuals vs fitted values.

        Args:
            y_true (np.ndarray | pd.Series | list): True values of the target variable.
            y_pred (np.ndarray | pd.Series | list): Predicted values of the target variable.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1000.
            height (int, optional): Height of the plot. Defaults to 1000.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Fitted Values".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Residuals".
            metric_name (str, optional): Name of the evaluation metric. Defaults to "RMSE".
            metric_value (float | None, optional): Value of the evaluation metric. Defaults to None.
            scatter_color (str | None, optional): Color for scatter points. Defaults to None.
            scatter_size (int, optional): Size of scatter points. Defaults to 8.
            scatter_opacity (float, optional): Opacity of scatter points. Defaults to 0.6.
            reference_line_color (str | None, optional): Color for reference line. Defaults to None.
            reference_line_width (int, optional): Width of reference line. Defaults to 4.
            show_trendline (bool, optional): Whether to show trendline. Defaults to False.
            trendline_color (str | None, optional): Color for trendline. Defaults to None.
        """
        y_true, y_pred = self._validate_regression_inputs(y_true, y_pred)
        residuals = y_true - y_pred

        if metric_value is None:
            metric_value = self._calculate_metric_value(y_true, y_pred, metric_name)

        if not plot_title:
            plot_title = (
                f"Homoscedacity Plot ({metric_name.upper()}: {metric_value:.4f})"
            )
        else:
            plot_title = f"{plot_title} ({metric_name.upper()}: {metric_value:.4f})"

        fig = go.Figure()

        scatter_trace = self._create_scatter_trace(
            y_pred, residuals, scatter_color, scatter_size, scatter_opacity
        )
        fig.add_trace(scatter_trace)

        self._add_reference_line(
            fig, "horizontal", 0, reference_line_color, reference_line_width
        )

        if show_trendline:
            if trendline_color is None:
                trendline_color = self.default_regression_colors["trendline"]

            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=p(y_pred),
                    mode="lines",
                    line=dict(color=trendline_color, width=3),
                    name="Trend Line",
                    showlegend=False,
                )
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)

    def true_vs_fitted_plot(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        plot_title: str = "",
        width: int = 1000,
        height: int = 1000,
        xaxis_title: str = "Fitted Values",
        yaxis_title: str = "True Values",
        metric_name: str = "R2",
        metric_value: float | None = None,
        scatter_color: str | None = None,
        scatter_size: int = 8,
        scatter_opacity: float = 0.6,
        show_diagonal: bool = True,
        diagonal_color: str | None = None,
        diagonal_width: int = 3,
        show_trendline: bool = False,
        trendline_color: str | None = None,
        trendline_width: int = 4,
        axis_buffer_percent: float = 2.0,
    ) -> None:
        """Creates a true vs fitted values plot to assess prediction accuracy.

        Args:
            y_true (np.ndarray | pd.Series | list): True values of the target variable.
            y_pred (np.ndarray | pd.Series | list): Predicted values of the target variable.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1000.
            height (int, optional): Height of the plot. Defaults to 1000.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Fitted Values".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "True Values".
            metric_name (str, optional): Name of the evaluation metric. Defaults to "R2".
            metric_value (float | None, optional): Value of the evaluation metric. Defaults to None.
            scatter_color (str | None, optional): Color for scatter points. Defaults to None.
            scatter_size (int, optional): Size of scatter points. Defaults to 8.
            scatter_opacity (float, optional): Opacity of scatter points. Defaults to 0.6.
            show_diagonal (bool, optional): Whether to show perfect
            prediction line. Defaults to True.
            diagonal_color (str | None, optional): Color for diagonal line. Defaults to None.
            diagonal_width (int, optional): Width of diagonal line. Defaults to 3.
            show_trendline (bool, optional): Whether to show trendline. Defaults to False.
            trendline_color (str | None, optional): Color for trendline. Defaults to None.
            trendline_width (int, optional): Width of trendline. Defaults to 4.
            axis_buffer_percent (float, optional): Buffer percentage for
            axis ranges. Defaults to 2.0.
        """
        y_true, y_pred = self._validate_regression_inputs(y_true, y_pred)

        if metric_value is None:
            metric_value = self._calculate_metric_value(y_true, y_pred, metric_name)

        if not plot_title:
            plot_title = (
                f"True vs Fitted Plot ({metric_name.upper()}: {metric_value:.4f})"
            )

        fig = go.Figure()

        scatter_trace = self._create_scatter_trace(
            y_pred, y_true, scatter_color, scatter_size, scatter_opacity
        )
        fig.add_trace(scatter_trace)

        if show_diagonal:
            if diagonal_color is None:
                diagonal_color = self.default_regression_colors["line"]

            min_val = min(np.min(y_pred), np.min(y_true))
            max_val = max(np.max(y_pred), np.max(y_true))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color=diagonal_color, width=diagonal_width, dash="dash"),
                    name="Identity Line",
                    showlegend=True,
                )
            )

        if show_trendline:
            if trendline_color is None:
                trendline_color = self.default_regression_colors["trendline"]

            z = np.polyfit(y_pred, y_true, 1)
            p = np.poly1d(z)
            fig.add_trace(
                go.Scatter(
                    x=y_pred,
                    y=p(y_pred),
                    mode="lines",
                    line=dict(color=trendline_color, width=trendline_width),
                    name="Trendline",
                    showlegend=True,
                )
            )

        buffer_x = (np.max(y_pred) - np.min(y_pred)) * axis_buffer_percent / 100
        buffer_y = (np.max(y_true) - np.min(y_true)) * axis_buffer_percent / 100

        fig.update_xaxes(range=[np.min(y_pred) - buffer_x, np.max(y_pred) + buffer_x])
        fig.update_yaxes(range=[np.min(y_true) - buffer_y, np.max(y_true) + buffer_y])

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_layout(
            showlegend=True,
            legend=dict(
                x=1,
                y=0.05,
                xanchor="right",
                yanchor="bottom",
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="black",
                borderwidth=1,
            ),
        )
        fig.show("png", width=width, height=height)

    def residual_distribution_plot(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        plot_title: str = "",
        width: int = 1200,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "",
        plot_type: str = "histogram",
        metric_name: str = "RMSE",
        metric_value: float | None = None,
        bins: int = 50,
        color: str | None = None,
        opacity: float = 0.7,
        show_normal_curve: bool = True,
        normal_curve_color: str = "red",
    ) -> None:
        """Creates a residual distribution plot (histogram or box plot).

        Args:
            y_true (np.ndarray | pd.Series | list): True values of the target variable.
            y_pred (np.ndarray | pd.Series | list): Predicted values of the target variable.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1200.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            plot_type (str, optional): Type of plot ("histogram" or "box"). Defaults to "histogram".
            metric_name (str, optional): Name of the evaluation metric. Defaults to "RMSE".
            metric_value (float | None, optional): Value of the evaluation metric. Defaults to None.
            bins (int, optional): Number of bins for histogram. Defaults to 50.
            color (str | None, optional): Color for the plot. Defaults to None.
            opacity (float, optional): Opacity of the plot. Defaults to 0.7.
            show_normal_curve (bool, optional): Whether to overlay normal curve. Defaults to True.
            normal_curve_color (str, optional): Color for normal curve. Defaults to "red".
        """
        y_true, y_pred = self._validate_regression_inputs(y_true, y_pred)
        residuals = y_true - y_pred

        if metric_value is None:
            metric_value = self._calculate_metric_value(y_true, y_pred, metric_name)

        if not xaxis_title:
            xaxis_title = "Residuals" if plot_type == "histogram" else ""
        if not yaxis_title:
            yaxis_title = "Frequency" if plot_type == "histogram" else "Residuals"

        if not plot_title:
            plot_title = f"Residual {plot_type.title()} ({metric_name.upper()}: {metric_value:.4f})"

        if color is None:
            color = self.default_regression_colors["scatter"]

        fig = go.Figure()

        if plot_type.lower() == "histogram":
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    nbinsx=bins,
                    marker=dict(
                        color=color, opacity=opacity, line=dict(color="black", width=1)
                    ),
                    name="Residuals",
                    showlegend=False,
                )
            )

            if show_normal_curve:
                x_range = np.linspace(residuals.min(), residuals.max(), 100)
                normal_curve = (
                    len(residuals)
                    * (residuals.max() - residuals.min())
                    / bins
                    * (1 / np.sqrt(2 * np.pi * np.var(residuals)))
                    * np.exp(
                        -0.5 * ((x_range - np.mean(residuals)) / np.std(residuals)) ** 2
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=normal_curve,
                        mode="lines",
                        line=dict(color=normal_curve_color, width=3),
                        name="Normal Distribution",
                        showlegend=False,
                    )
                )

        elif plot_type.lower() == "box":
            fig.add_trace(
                go.Box(
                    y=residuals,
                    marker=dict(color=color, opacity=opacity),
                    name="Residuals",
                    showlegend=False,
                )
            )

        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}. Use 'histogram' or 'box'."
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    reg_plots = RegressionPlots()

    reg_plots.homoscedacity_plot(
        y_test, y_pred, metric_name="R2", width=800, height=600, show_trendline=True
    )

    reg_plots.true_vs_fitted_plot(
        y_test, y_pred, metric_name="RMSE", width=800, height=600, show_diagonal=True
    )

    reg_plots.residual_distribution_plot(
        y_test, y_pred, plot_type="histogram", width=800, height=600, bins=25
    )
