import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..base_plots import BasePlots


class TimeSeriesPlots(BasePlots):
    """Class for creating time series visualizations.

    This class inherits from BasePlots and provides specialized methods
    for visualizing time series data including trend analysis, seasonal
    decomposition, and aggregated time series plots.
    """

    def _prepare_time_series_data(
        self, data: pd.DataFrame, feature: str, target: str | list[str]
    ) -> pd.DataFrame:
        """Prepares time series data by filtering NaN values and converting datetime.

        Args:
            data (pd.DataFrame): Input DataFrame containing time feature and target variable(s).
            feature (str): Name of the datetime column.
            target (str | list[str]): Name(s) of the target variable(s).

        Returns:
            pd.DataFrame: Prepared DataFrame with datetime index.
        """
        if isinstance(target, str):
            target_cols = [target]
        else:
            target_cols = target
        data_copy = data.dropna(subset=[feature] + target_cols).copy()
        data_copy = data_copy.set_index(feature)
        data_copy.index = pd.to_datetime(
            data_copy.index, errors="coerce", dayfirst=True, utc=True
        )
        data_copy = data_copy.sort_index()
        return (
            data_copy[target_cols]
            if len(target_cols) > 1
            else data_copy[[target_cols[0]]]
        )

    def plot_time_series_mean(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str,
        agg_freq: str = "ME",
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Date",
        yaxis_title: str = "",
        line_color: str = "",
        line_width: int = 2,
        show_markers: bool = False,
        marker_size: int = 6,
    ) -> None:
        """Creates a time series plot showing mean values over time.

        Args:
            data (pd.DataFrame): Input DataFrame containing time feature and target variable.
            feature (str): Name of the datetime column for x-axis.
            target (str): Name of the target variable for y-axis.
            agg_freq (str, optional): Aggregation frequency (pandas frequency string). Defaults to 'ME'.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            line_color (str, optional): Color of the line. Defaults to "".
            line_width (int, optional): Width of the line. Defaults to 2.
            show_markers (bool, optional): Whether to show markers on the line. Defaults to False.
            marker_size (int, optional): Size of the markers. Defaults to 6.
        """
        data_copy = self._prepare_time_series_data(data, feature, target)
        resampled_data = data_copy.resample(agg_freq)[target].mean()
        if not line_color:
            line_color = self.default_colors["primary"]
        if not yaxis_title:
            yaxis_title = f"Mean of {target}"
        fig = go.Figure()
        mode = "lines+markers" if show_markers else "lines"

        fig.add_trace(
            go.Scatter(
                x=resampled_data.index,
                y=resampled_data.values,
                mode=mode,
                line=dict(color=line_color, width=line_width),
                marker=(
                    dict(size=marker_size, color=line_color) if show_markers else None
                ),
                name="Mean",
                showlegend=False,
            )
        )
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)

    def plot_time_series_multiple_metrics(
        self,
        data: pd.DataFrame,
        feature: str,
        targets: list[str],
        agg_freq: str = "ME",
        agg_func: str = "mean",
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Date",
        yaxis_title: str = "",
        show_markers: bool = True,
        marker_size: int = 6,
        line_width: int = 2,
        show_legend: bool = True,
        opacity: float = 1.0,
    ) -> None:
        """Creates a time series plot with multiple target variables.

        Args:
            data (pd.DataFrame): Input DataFrame containing time feature and target variables.
            feature (str): Name of the datetime column for x-axis.
            targets (list[str]): List of target variable names for y-axis.
            agg_freq (str, optional): Aggregation frequency (pandas frequency string). Defaults to 'ME'.
            agg_func (str, optional): Aggregation function ('mean', 'sum', 'max', 'min', 'std', 'median'). Defaults to 'mean'.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            show_markers (bool, optional): Whether to show markers on lines. Defaults to True.
            marker_size (int, optional): Size of the markers. Defaults to 6.
            line_width (int, optional): Width of the lines. Defaults to 2.
            show_legend (bool, optional): Whether to show the legend. Defaults to True.
            opacity (float, optional): Opacity of the lines (0-1). Defaults to 1.0.
        """
        data_copy = self._prepare_time_series_data(data, feature, targets)

        colors = self._get_colors(len(targets))

        if not yaxis_title:
            yaxis_title = f"{agg_func.title()} Values"

        fig = go.Figure()
        mode = "lines+markers" if show_markers else "lines"

        for idx, target in enumerate(targets):
            resampled_data = data_copy.resample(agg_freq)[target].agg(
                lambda x: self._apply_aggregation(x, agg_func)
            )

            fig.add_trace(
                go.Scatter(
                    x=resampled_data.index,
                    y=resampled_data.values,
                    mode=mode,
                    line=dict(color=colors[idx], width=line_width),
                    marker=(
                        dict(size=marker_size, color=colors[idx])
                        if show_markers
                        else None
                    ),
                    name=f"{agg_func.title()} of {target}",
                    showlegend=show_legend,
                    opacity=opacity,
                )
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)

    def plot_time_series_with_trend(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str,
        window_size: int = 30,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Date",
        yaxis_title: str = "",
        original_color: str = "",
        trend_color: str = "",
        original_opacity: float = 0.7,
        trend_line_width: int = 3,
        original_line_width: int = 1,
        show_legend: bool = True,
        trend_mode: str = "lines",
    ) -> None:
        """Creates a time series plot with trend line using moving average.

        Args:
            data (pd.DataFrame): Input DataFrame containing time feature and target variable.
            feature (str): Name of the datetime column for x-axis.
            target (str): Name of the target variable for y-axis.
            window_size (int, optional): Window size for moving average. Defaults to 30.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            original_color (str, optional): Color for the original data line. Defaults to "".
            trend_color (str, optional): Color for the trend line. Defaults to "".
            original_opacity (float, optional): Opacity of the original data line. Defaults to 0.7.
            trend_line_width (int, optional): Width of the trend line. Defaults to 3.
            original_line_width (int, optional): Width of the original data line. Defaults to 1.
            show_legend (bool, optional): Whether to show the legend. Defaults to True.
            trend_mode (str, optional): Display mode for trend line ('lines', 'lines+markers'). Defaults to "lines".
        """
        data_copy = self._prepare_time_series_data(data, feature, target)

        data_copy["trend"] = (
            data_copy[target].rolling(window=window_size, center=True).mean()
        )

        if not original_color:
            original_color = self.default_colors["secondary"]
        if not trend_color:
            trend_color = self.default_colors["primary"]
        if not yaxis_title:
            yaxis_title = target

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy[target],
                mode="lines",
                line=dict(color=original_color, width=original_line_width),
                name=f"Original {target}",
                opacity=original_opacity,
                showlegend=show_legend,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy["trend"],
                mode=trend_mode,
                line=dict(color=trend_color, width=trend_line_width),
                name=f"Trend (MA{window_size})",
                showlegend=show_legend,
            )
        )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_layout(showlegend=show_legend)
        fig.show("png", width=width, height=height)

    def plot_seasonal_decomposition(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str,
        freq: int = 365,
        plot_title: str = "",
        width: int = 1600,
        height: int = 1000,
        xaxis_title: str = "Date",
        yaxis_title: str = "",
        original_color: str = "",
        trend_color: str = "",
        seasonal_color: str = "",
        residual_color: str = "red",
        line_width: int = 2,
        show_subplot_titles: bool = True,
        vertical_spacing: float = 0.08,
    ) -> None:
        """Creates a seasonal decomposition plot showing trend, seasonal, and residual components.

        Args:
            data (pd.DataFrame): Input DataFrame containing time feature and target variable.
            feature (str): Name of the datetime column for x-axis.
            target (str): Name of the target variable for decomposition.
            freq (int, optional): Frequency for seasonal decomposition. Defaults to 365.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 1000.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            original_color (str, optional): Color for the original data line. Defaults to "".
            trend_color (str, optional): Color for the trend line. Defaults to "".
            seasonal_color (str, optional): Color for the seasonal line. Defaults to "".
            residual_color (str, optional): Color for the residual line. Defaults to "red".
            line_width (int, optional): Width of all lines. Defaults to 2.
            show_subplot_titles (bool, optional): Whether to show subplot titles. Defaults to True.
            vertical_spacing (float, optional): Vertical spacing between subplots. Defaults to 0.08.
        """
        data_copy = self._prepare_time_series_data(data, feature, target)

        if not original_color:
            original_color = self.default_colors["primary"]
        if not trend_color:
            trend_color = self.default_colors["accent"]
        if not seasonal_color:
            seasonal_color = self.default_colors["secondary"]
        if not yaxis_title:
            yaxis_title = target

        values = data_copy[target].values

        trend = pd.Series(values).rolling(window=freq, center=True).mean()

        detrended = values - trend
        seasonal_avg = []
        for i in range(freq):
            seasonal_vals = detrended[i::freq]
            seasonal_avg.append(np.nanmean(seasonal_vals))

        seasonal = np.tile(seasonal_avg, len(values) // freq + 1)[: len(values)]

        residual = values - trend - seasonal

        subplot_titles = (
            ("Original", "Trend", "Seasonal", "Residual")
            if show_subplot_titles
            else None
        )
        fig = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=subplot_titles,
            vertical_spacing=vertical_spacing,
        )

        if subplot_titles:
            for i, _ in enumerate(subplot_titles):
                fig.layout.annotations[i].font.size = 20

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=values,
                mode="lines",
                name="Original",
                line=dict(color=original_color, width=line_width),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=trend,
                mode="lines",
                name="Trend",
                line=dict(color=trend_color, width=line_width),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=seasonal,
                mode="lines",
                name="Seasonal",
                line=dict(color=seasonal_color, width=line_width),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=residual,
                mode="lines",
                name="Residual",
                line=dict(color=residual_color, width=line_width),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(title="", row=1, col=1)
        fig.update_xaxes(title=xaxis_title, row=4, col=1)
        fig.update_layout(showlegend=False)
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta

    # Create sample time series data
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(365)]

    # Generate synthetic time series with trend and seasonality
    trend = np.linspace(100, 200, 365)
    seasonal = 20 * np.sin(
        2 * np.pi * np.arange(365) / 365.25 * 4
    )  # Quarterly seasonality
    noise = np.random.normal(0, 10, 365)
    values1 = trend + seasonal + noise
    values2 = trend * 0.8 + seasonal * 1.5 + np.random.normal(0, 8, 365)

    sample_data = pd.DataFrame(
        {
            "date": dates,
            "metric1": values1,
            "metric2": values2,
            "metric3": np.random.normal(50, 15, 365),
        }
    )

    # Initialize TimeSeriesPlots
    ts_plots = TimeSeriesPlots()

    # Example 1: Time series mean plot
    ts_plots.plot_time_series_mean(
        data=sample_data,
        feature="date",
        target="metric1",
        agg_freq="ME",
        plot_title="Monthly Mean Time Series",
        width=1200,
        height=600,
        xaxis_title="Date",
        yaxis_title="Average Value",
        line_color="blue",
        line_width=3,
        show_markers=True,
        marker_size=8,
    )

    # Example 2: Multiple metrics plot
    ts_plots.plot_time_series_multiple_metrics(
        data=sample_data,
        feature="date",
        targets=["metric1", "metric2"],
        agg_freq="ME",
        agg_func="mean",
        plot_title="Multiple Metrics Comparison",
        width=1400,
        height=700,
        show_markers=True,
        marker_size=6,
        line_width=2,
        show_legend=True,
        opacity=0.8,
    )

    # Example 3: Time series with trend
    ts_plots.plot_time_series_with_trend(
        data=sample_data,
        feature="date",
        target="metric1",
        window_size=30,
        plot_title="Time Series with 30-Day Moving Average",
        width=1500,
        height=800,
        original_opacity=0.6,
        trend_line_width=4,
        show_legend=True,
    )

    # Example 4: Seasonal decomposition
    ts_plots.plot_seasonal_decomposition(
        data=sample_data,
        feature="date",
        target="metric1",
        freq=90,
        plot_title="Seasonal Decomposition Analysis",
        width=1600,
        height=1200,
        line_width=2,
        show_subplot_titles=True,
    )
