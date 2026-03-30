
from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base_plots import BasePlots


class TimeSeriesPlots(BasePlots):
    """Class for creating time series visualizations.

    This class inherits from BasePlots and provides specialized methods
    for visualizing time series data including trend analysis, seasonal
    decomposition, distributions over time, and correlation diagnostics.
    """

    def _prepare_time_series_data(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str | list[str],
    ) -> pd.DataFrame:
        """Prepare time series data by filtering NaN values and converting datetime.

        Args:
            data (pd.DataFrame): Input DataFrame containing date and value feature(s).
            date_feature (str): Name of the datetime column.
            feature (str | list[str]): Name(s) of the value feature(s).

        Returns:
            pd.DataFrame: Prepared DataFrame with datetime index.

        Raises:
            KeyError: If any required column is missing.
        """
        if isinstance(feature, str):
            feature_cols = [feature]
        else:
            feature_cols = list(feature)

        required_cols = [date_feature] + feature_cols
        missing_cols = [
            column for column in required_cols if column not in data.columns
        ]
        if missing_cols:
            missing_cols_display = ", ".join(missing_cols)
            raise KeyError(f"Missing required columns: {missing_cols_display}")

        data_copy = data.dropna(subset=required_cols).copy()
        data_copy = data_copy.set_index(date_feature)
        data_copy.index = self._parse_datetime_with_dayfirst_fallback(data_copy.index)
        data_copy = data_copy[~data_copy.index.isna()]
        data_copy = data_copy.sort_index()
        return data_copy[feature_cols]

    def _parse_datetime_with_dayfirst_fallback(
        self,
        values: pd.Index | pd.Series,
    ) -> pd.DatetimeIndex | pd.Series:
        """Parse datetimes without warnings and keep support for day-first strings.

        Args:
            values (pd.Index | pd.Series): Input datetime-like values.

        Returns:
            pd.DatetimeIndex | pd.Series: Parsed UTC datetimes.
        """
        parsed = pd.to_datetime(values, errors="coerce", utc=True)
        missing_mask = parsed.isna()

        if bool(np.any(missing_mask)):
            fallback_values = values[missing_mask]
            parsed_fallback = pd.to_datetime(
                fallback_values,
                errors="coerce",
                dayfirst=True,
                utc=True,
            )
            parsed[missing_mask] = parsed_fallback

        return parsed

    def _format_period_label(self, value: pd.Timestamp) -> str:
        """Format a timestamp label for grouped periods.

        Args:
            value (pd.Timestamp): Timestamp value.

        Returns:
            str: Human-readable period label.
        """
        return value.strftime("%Y-%m-%d")

    def _normalize_period_frequency(self, agg_freq: str) -> str:
        """Normalize pandas offset aliases for Period conversion.

        Args:
            agg_freq (str): Frequency used for resampling.

        Returns:
            str: Frequency compatible with Period conversion.
        """
        replacements = {
            "ME": "M",
            "QE": "Q",
            "YE": "Y",
            "BME": "BM",
            "BQE": "BQ",
            "BYE": "BY",
        }
        return replacements.get(agg_freq, agg_freq)

    def _build_period_labels(self, date_series: pd.Series, agg_freq: str) -> pd.Series:
        """Convert datetime values into period labels for grouped plotting.

        Args:
            date_series (pd.Series): Datetime values.
            agg_freq (str): Frequency alias.

        Returns:
            pd.Series: Period labels.
        """
        normalized_freq = self._normalize_period_frequency(agg_freq)
        timestamp_series = self._parse_datetime_with_dayfirst_fallback(
            date_series
        ).dt.tz_convert(None)
        return timestamp_series.dt.to_period(normalized_freq).astype(str)

    def _get_top_n_categories(
        self,
        data: pd.DataFrame,
        category_feature: str,
        top_n_categories: int,
    ) -> list[str]:
        """Return top categories by frequency.

        Args:
            data (pd.DataFrame): Input data.
            category_feature (str): Category column.
            top_n_categories (int): Number of top categories.

        Returns:
            list[str]: Ordered category labels.
        """
        value_counts = data[category_feature].value_counts(dropna=True)
        return [str(category) for category in value_counts.head(top_n_categories).index]

    def _compute_autocorrelation(self, values: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation values for lags from 0 to max_lag.

        Args:
            values (np.ndarray): Input 1D values.
            max_lag (int): Maximum lag.

        Returns:
            np.ndarray: Autocorrelation values.
        """
        centered = values.astype(float) - float(np.mean(values))
        denominator = float(np.dot(centered, centered))

        if denominator == 0:
            return np.zeros(max_lag + 1)

        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                correlations.append(1.0)
            elif lag >= len(centered):
                correlations.append(np.nan)
            else:
                numerator = float(np.dot(centered[:-lag], centered[lag:]))
                correlations.append(numerator / denominator)
        return np.array(correlations)

    def _compute_partial_autocorrelation(
        self, values: np.ndarray, max_lag: int
    ) -> np.ndarray:
        """Compute partial autocorrelation values using linear regression.

        Args:
            values (np.ndarray): Input 1D values.
            max_lag (int): Maximum lag.

        Returns:
            np.ndarray: Partial autocorrelation values.
        """
        centered = values.astype(float) - float(np.mean(values))
        n_values = len(centered)
        partial_correlations = [1.0]

        for lag in range(1, max_lag + 1):
            if lag >= n_values:
                partial_correlations.append(np.nan)
                continue

            target_values = centered[lag:]
            lagged_matrix = np.column_stack(
                [
                    centered[lag - shift - 1 : n_values - shift - 1]
                    for shift in range(lag)
                ]
            )
            try:
                coefficients, _, _, _ = np.linalg.lstsq(
                    lagged_matrix, target_values, rcond=None
                )
                partial_correlations.append(float(coefficients[-1]))
            except np.linalg.LinAlgError:
                partial_correlations.append(np.nan)

        return np.array(partial_correlations)

    def plot_time_series_mean(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
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
        """Create a time series plot showing mean values over time.

        Args:
            data (pd.DataFrame): Input DataFrame containing date and value feature.
            date_feature (str): Name of the datetime column for x-axis.
            feature (str): Name of the value feature for y-axis.
            agg_freq (str, optional): Aggregation frequency. Defaults to "ME".
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            line_color (str, optional): Color of the line. Defaults to "".
            line_width (int, optional): Width of the line. Defaults to 2.
            show_markers (bool, optional): Whether to show markers. Defaults to False.
            marker_size (int, optional): Size of markers. Defaults to 6.
        """
        data_copy = self._prepare_time_series_data(data, date_feature, feature)
        resampled_data = data_copy.resample(agg_freq)[feature].mean()

        resolved_line_color = line_color or self.default_colors["primary"]
        resolved_yaxis_title = yaxis_title or f"Mean of {feature}"
        mode = "lines+markers" if show_markers else "lines"

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=resampled_data.index,
                y=resampled_data.values,
                mode=mode,
                line=dict(color=resolved_line_color, width=line_width),
                marker=(
                    dict(size=marker_size, color=resolved_line_color)
                    if show_markers
                    else None
                ),
                name="Mean",
                showlegend=False,
            )
        )

        self.apply_default_layout(
            fig,
            plot_title,
            width,
            height,
            xaxis_title,
            resolved_yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def plot_time_series_multiple_metrics(
        self,
        data: pd.DataFrame,
        date_feature: str,
        features: list[str],
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
        """Create a time series plot with multiple value features.

        Args:
            data (pd.DataFrame): Input DataFrame containing date and value features.
            date_feature (str): Name of the datetime column for x-axis.
            features (list[str]): List of value features for y-axis.
            agg_freq (str, optional): Aggregation frequency. Defaults to "ME".
            agg_func (str, optional): Aggregation function. Defaults to "mean".
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            show_markers (bool, optional): Whether to show markers. Defaults to True.
            marker_size (int, optional): Size of markers. Defaults to 6.
            line_width (int, optional): Width of lines. Defaults to 2.
            show_legend (bool, optional): Whether to show legend. Defaults to True.
            opacity (float, optional): Opacity of lines. Defaults to 1.0.
        """
        data_copy = self._prepare_time_series_data(data, date_feature, features)
        colors = self._get_colors(len(features))
        resolved_yaxis_title = yaxis_title or f"{agg_func.title()} Values"
        mode = "lines+markers" if show_markers else "lines"

        fig = go.Figure()
        for index, feature_name in enumerate(features):
            resampled_data = data_copy.resample(agg_freq)[feature_name].agg(
                lambda series: self._apply_aggregation(series, agg_func)
            )
            fig.add_trace(
                go.Scatter(
                    x=resampled_data.index,
                    y=resampled_data.values,
                    mode=mode,
                    line=dict(color=colors[index], width=line_width),
                    marker=(
                        dict(size=marker_size, color=colors[index])
                        if show_markers
                        else None
                    ),
                    name=f"{agg_func.title()} of {feature_name}",
                    showlegend=show_legend,
                    opacity=opacity,
                )
            )

        self.apply_default_layout(
            fig,
            plot_title,
            width,
            height,
            xaxis_title,
            resolved_yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def plot_time_series_with_trend(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
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
        """Create a time series plot with moving-average trend line.

        Args:
            data (pd.DataFrame): Input DataFrame containing date and value feature.
            date_feature (str): Name of the datetime column for x-axis.
            feature (str): Name of the value feature for y-axis.
            window_size (int, optional): Window size for moving average. Defaults to 30.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            original_color (str, optional): Color of original series. Defaults to "".
            trend_color (str, optional): Color of trend line. Defaults to "".
            original_opacity (float, optional): Opacity of original series. Defaults to 0.7.
            trend_line_width (int, optional): Width of trend line. Defaults to 3.
            original_line_width (int, optional): Width of original line. Defaults to 1.
            show_legend (bool, optional): Whether to show legend. Defaults to True.
            trend_mode (str, optional): Trend mode. Defaults to "lines".
        """
        data_copy = self._prepare_time_series_data(data, date_feature, feature)
        data_copy["trend"] = (
            data_copy[feature].rolling(window=window_size, center=True).mean()
        )

        resolved_original_color = original_color or self.default_colors["secondary"]
        resolved_trend_color = trend_color or self.default_colors["primary"]
        resolved_yaxis_title = yaxis_title or feature

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy[feature],
                mode="lines",
                line=dict(color=resolved_original_color, width=original_line_width),
                name=f"Original {feature}",
                opacity=original_opacity,
                showlegend=show_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=data_copy.index,
                y=data_copy["trend"],
                mode=trend_mode,
                line=dict(color=resolved_trend_color, width=trend_line_width),
                name=f"Trend (MA{window_size})",
                showlegend=show_legend,
            )
        )

        self.apply_default_layout(
            fig,
            plot_title,
            width,
            height,
            xaxis_title,
            resolved_yaxis_title,
        )
        fig.update_layout(showlegend=show_legend)
        fig.show("png", width=width, height=height)

    def plot_seasonal_decomposition(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        agg_freq: str | None = "D",
        agg_func: str = "mean",
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
        """Create a seasonal decomposition plot with trend and residual components.

        Args:
            data (pd.DataFrame): Input DataFrame containing date and value feature.
            date_feature (str): Name of the datetime column for x-axis.
            feature (str): Name of the value feature for decomposition.
            agg_freq (str | None, optional): Optional frequency used to resample data before
                decomposition. Use None to skip pre-aggregation. Defaults to "D".
            agg_func (str, optional): Aggregation function used with agg_freq.
                Defaults to "mean".
            freq (int, optional): Frequency for decomposition. Defaults to 365.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 1000.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Date".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".
            original_color (str, optional): Color of original component. Defaults to "".
            trend_color (str, optional): Color of trend component. Defaults to "".
            seasonal_color (str, optional): Color of seasonal component. Defaults to "".
            residual_color (str, optional): Color of residual component. Defaults to "red".
            line_width (int, optional): Width of component lines. Defaults to 2.
            show_subplot_titles (bool, optional): Whether to show subplot titles. Defaults to True.
            vertical_spacing (float, optional): Spacing between subplots. Defaults to 0.08.
        """
        data_copy = self._prepare_time_series_data(data, date_feature, feature)

        if agg_freq is not None:
            decomposed_series = (
                data_copy[feature]
                .resample(agg_freq)
                .agg(lambda series: self._apply_aggregation(series, agg_func))
            )
            decomposed_series = decomposed_series.dropna()
            decomposition_index = decomposed_series.index
            values = decomposed_series.values
        else:
            decomposition_index = data_copy.index
            values = data_copy[feature].values

        resolved_original_color = original_color or self.default_colors["primary"]
        resolved_trend_color = trend_color or self.default_colors["accent"]
        resolved_seasonal_color = seasonal_color or self.default_colors["secondary"]
        resolved_yaxis_title = yaxis_title or feature

        trend_values = pd.Series(values).rolling(window=freq, center=True).mean().values

        detrended = values - trend_values
        seasonal_mean = [np.nanmean(detrended[offset::freq]) for offset in range(freq)]
        seasonal_values = np.tile(seasonal_mean, len(values) // freq + 1)[: len(values)]
        residual_values = values - trend_values - seasonal_values

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

        fig.add_trace(
            go.Scatter(
                x=decomposition_index,
                y=values,
                mode="lines",
                name="Original",
                line=dict(color=resolved_original_color, width=line_width),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=decomposition_index,
                y=trend_values,
                mode="lines",
                name="Trend",
                line=dict(color=resolved_trend_color, width=line_width),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=decomposition_index,
                y=seasonal_values,
                mode="lines",
                name="Seasonal",
                line=dict(color=resolved_seasonal_color, width=line_width),
                showlegend=False,
            ),
            row=3,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=decomposition_index,
                y=residual_values,
                mode="lines",
                name="Residual",
                line=dict(color=residual_color, width=line_width),
                showlegend=False,
            ),
            row=4,
            col=1,
        )

        self.apply_default_layout(
            fig,
            plot_title,
            width,
            height,
            xaxis_title,
            resolved_yaxis_title,
        )
        fig.update_xaxes(title="", row=1, col=1)
        fig.update_xaxes(title=xaxis_title, row=4, col=1)
        fig.update_layout(showlegend=False)
        fig.show("png", width=width, height=height)

    def plot_time_series_boxplot(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        agg_freq: str = "ME",
        top_n_periods: int | None = None,
        plot_title: str = "",
        width: int = 1600,
        height: int = 850,
        xaxis_title: str = "Period",
        yaxis_title: str = "",
        boxpoints: str = "outliers",
        jitter: float = 0.2,
        pointpos: float = 0.0,
    ) -> None:
        """Create boxplots of a feature for aggregated time periods.

        Args:
            data (pd.DataFrame): Input DataFrame.
            date_feature (str): Datetime column name.
            feature (str): Value feature to display as boxplots.
            agg_freq (str, optional): Time grouping frequency. Defaults to "ME".
            top_n_periods (int | None, optional): Number of most recent periods to keep.
                Use None to show all periods. Defaults to None.
            plot_title (str, optional): Plot title. Defaults to "".
            width (int, optional): Plot width. Defaults to 1600.
            height (int, optional): Plot height. Defaults to 850.
            xaxis_title (str, optional): X axis title. Defaults to "Period".
            yaxis_title (str, optional): Y axis title. Defaults to "".
            boxpoints (str, optional): Box points style. Defaults to "outliers".
            jitter (float, optional): Point jitter. Defaults to 0.2.
            pointpos (float, optional): Point offset. Defaults to 0.0.

        Raises:
            ValueError: If top_n_periods is not greater than 0 when provided.
        """
        data_copy = self._prepare_time_series_data(data, date_feature, feature)
        grouped = [
            (period, values.dropna())
            for period, values in data_copy[feature].resample(agg_freq)
            if not values.dropna().empty
        ]

        if top_n_periods is not None:
            if top_n_periods <= 0:
                raise ValueError("top_n_periods must be greater than 0 when provided")
            grouped = grouped[-top_n_periods:]

        resolved_yaxis_title = yaxis_title or feature
        colors = px.colors.qualitative.Pastel

        fig = go.Figure()
        for index, (period, values) in enumerate(grouped):
            period_color = colors[index % len(colors)]
            fig.add_trace(
                go.Box(
                    y=values.values,
                    name=self._format_period_label(period),
                    marker=dict(color=period_color),
                    fillcolor=period_color,
                    boxpoints=boxpoints,
                    jitter=jitter,
                    pointpos=pointpos,
                    line=dict(color="black", width=1),
                    opacity=0.85,
                    showlegend=False,
                )
            )

        self.apply_default_layout(
            fig,
            plot_title or f"{feature} Distribution by {agg_freq}",
            width,
            height,
            xaxis_title,
            resolved_yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def plot_feature_distribution_by_category_over_time(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        category_feature: str,
        agg_freq: str = "ME",
        top_n_categories: int = 5,
        plot_title: str = "",
        width: int = 1600,
        height: int = 850,
        xaxis_title: str = "Period",
        yaxis_title: str = "",
        show_points: bool = True,
        opacity: float = 0.75,
    ) -> None:
        """Plot per-category feature distributions over time in separate plots.

        Args:
            data (pd.DataFrame): Input DataFrame.
            date_feature (str): Datetime column name.
            feature (str): Value feature to model.
            category_feature (str): Category feature used for split plots.
            agg_freq (str, optional): Time grouping frequency. Defaults to "ME".
            top_n_categories (int, optional): Number of categories to show. Defaults to 5.
            plot_title (str, optional): Base title for all generated figures. Defaults to "".
            width (int, optional): Plot width. Defaults to 1600.
            height (int, optional): Plot height. Defaults to 850.
            xaxis_title (str, optional): X axis title. Defaults to "Period".
            yaxis_title (str, optional): Y axis title. Defaults to "".
            show_points (bool, optional): Show points inside distributions. Defaults to True.
            opacity (float, optional): Trace opacity. Defaults to 0.75.

        Raises:
            ValueError: If distribution_kind is unsupported.
            KeyError: If category_feature is missing.
        """
        if category_feature not in data.columns:
            raise KeyError(f"Missing required columns: {category_feature}")

        prepared = self._prepare_time_series_data(
            data,
            date_feature,
            [feature, category_feature],
        )
        merged = prepared.reset_index().dropna(subset=[category_feature])
        top_categories = self._get_top_n_categories(
            merged, category_feature, top_n_categories
        )
        color_sequence = px.colors.qualitative.Pastel
        resolved_yaxis_title = yaxis_title or feature

        for index, category in enumerate(top_categories):
            category_frame = merged[
                merged[category_feature].astype(str) == category
            ].copy()
            category_frame = category_frame.sort_values(date_feature)
            category_frame["period"] = self._build_period_labels(
                category_frame[date_feature],
                agg_freq,
            )

            fig = go.Figure()
            marker_color = color_sequence[index % len(color_sequence)]
            fig.add_trace(
                go.Box(
                    x=category_frame["period"],
                    y=category_frame[feature],
                    marker=dict(color=marker_color),
                    fillcolor=marker_color,
                    boxpoints="outliers" if show_points else False,
                    line=dict(color="black", width=1),
                    opacity=opacity,
                    showlegend=False,
                    name=category,
                )
            )

            resolved_title = (
                f"{plot_title} - {category_feature}: {category}"
                if plot_title
                else f"{feature} by {category}"
            )
            self.apply_default_layout(
                fig,
                resolved_title,
                width,
                height,
                xaxis_title,
                resolved_yaxis_title,
            )
            fig.show("png", width=width, height=height)

    def plot_feature_distribution_by_numeric_feature_over_time(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        numeric_feature: str,
        agg_freq: str = "ME",
        n_bins: int = 4,
        binning_strategy: str = "quantile",
        plot_title: str = "",
        width: int = 1600,
        height: int = 850,
        xaxis_title: str = "Period",
        yaxis_title: str = "",
        show_points: bool = True,
        opacity: float = 0.75,
    ) -> None:
        """Plot feature distributions over time by numeric-feature bins.

        Args:
            data (pd.DataFrame): Input DataFrame.
            date_feature (str): Datetime column name.
            feature (str): Value feature to model.
            numeric_feature (str): Numeric feature used to build bins.
            agg_freq (str, optional): Time grouping frequency. Defaults to "ME".
            n_bins (int, optional): Number of bins. Defaults to 4.
            binning_strategy (str, optional): "quantile" or "uniform". Defaults to "quantile".
            plot_title (str, optional): Base title for generated figures. Defaults to "".
            width (int, optional): Plot width. Defaults to 1600.
            height (int, optional): Plot height. Defaults to 850.
            xaxis_title (str, optional): X axis title. Defaults to "Period".
            yaxis_title (str, optional): Y axis title. Defaults to "".
            show_points (bool, optional): Show points inside distributions. Defaults to True.
            opacity (float, optional): Trace opacity. Defaults to 0.75.

        Raises:
            KeyError: If numeric_feature is missing.
            ValueError: If strategy or plot kind is unsupported.
        """
        if numeric_feature not in data.columns:
            raise KeyError(f"Missing required columns: {numeric_feature}")
        if binning_strategy not in {"quantile", "uniform"}:
            raise ValueError("binning_strategy must be one of: 'quantile', 'uniform'")

        prepared = self._prepare_time_series_data(
            data,
            date_feature,
            [feature, numeric_feature],
        )
        merged = prepared.reset_index().dropna(subset=[numeric_feature])

        if binning_strategy == "quantile":
            merged["numeric_bin"] = pd.qcut(
                merged[numeric_feature],
                q=n_bins,
                duplicates="drop",
            ).astype(str)
        else:
            merged["numeric_bin"] = pd.cut(
                merged[numeric_feature],
                bins=n_bins,
                duplicates="drop",
            ).astype(str)

        color_sequence = px.colors.qualitative.Pastel
        resolved_yaxis_title = yaxis_title or feature
        bin_labels = list(merged["numeric_bin"].value_counts().sort_index().index)

        for index, bin_label in enumerate(bin_labels):
            bin_frame = merged[merged["numeric_bin"] == bin_label].copy()
            bin_frame = bin_frame.sort_values(date_feature)
            bin_frame["period"] = self._build_period_labels(
                bin_frame[date_feature],
                agg_freq,
            )

            marker_color = color_sequence[index % len(color_sequence)]
            fig = go.Figure()
            fig.add_trace(
                go.Box(
                    x=bin_frame["period"],
                    y=bin_frame[feature],
                    marker=dict(color=marker_color),
                    fillcolor=marker_color,
                    boxpoints="outliers" if show_points else False,
                    line=dict(color="black", width=1),
                    opacity=opacity,
                    showlegend=False,
                    name=bin_label,
                )
            )

            resolved_title = (
                f"{plot_title} - {numeric_feature}: {bin_label}"
                if plot_title
                else f"{feature} by {numeric_feature} bin {bin_label}"
            )
            self.apply_default_layout(
                fig,
                resolved_title,
                width,
                height,
                xaxis_title,
                resolved_yaxis_title,
            )
            fig.show("png", width=width, height=height)

    def plot_auto_and_partial_correlation(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        max_lag: int = 40,
        plot_title: str = "",
        width: int = 1600,
        height: int = 750,
        xaxis_title: str = "Lag",
        yaxis_title: str = "Correlation",
        acf_color: str = "",
        pacf_color: str = "",
        confidence_alpha: float = 0.05,
    ) -> None:
        """Create ACF and PACF bar plots in one figure.

        Args:
            data (pd.DataFrame): Input DataFrame.
            date_feature (str): Datetime column name.
            feature (str): Value feature used for correlation diagnostics.
            max_lag (int, optional): Maximum lag. Defaults to 40.
            plot_title (str, optional): Plot title. Defaults to "".
            width (int, optional): Plot width. Defaults to 1600.
            height (int, optional): Plot height. Defaults to 750.
            xaxis_title (str, optional): X axis title. Defaults to "Lag".
            yaxis_title (str, optional): Y axis title. Defaults to "Correlation".
            acf_color (str, optional): ACF bar color. Defaults to "".
            pacf_color (str, optional): PACF bar color. Defaults to "".
            confidence_alpha (float, optional): Significance level. Defaults to 0.05.
        """
        data_copy = self._prepare_time_series_data(data, date_feature, feature)
        values = data_copy[feature].dropna().values

        effective_lag = min(max_lag, max(len(values) - 1, 1))
        acf_values = self._compute_autocorrelation(values, effective_lag)
        pacf_values = self._compute_partial_autocorrelation(values, effective_lag)
        lags = np.arange(effective_lag + 1)

        confidence_multiplier = 1.96 if confidence_alpha == 0.05 else 2.58
        confidence_bound = confidence_multiplier / np.sqrt(len(values))

        resolved_acf_color = acf_color or px.colors.qualitative.Pastel[1]
        resolved_pacf_color = pacf_color or px.colors.qualitative.Pastel[4]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Autocorrelation", "Partial Autocorrelation"),
        )
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acf_values,
                marker=dict(
                    color=resolved_acf_color, line=dict(color="black", width=1)
                ),
                showlegend=False,
                opacity=0.9,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=lags,
                y=pacf_values,
                marker=dict(
                    color=resolved_pacf_color, line=dict(color="black", width=1)
                ),
                showlegend=False,
                opacity=0.9,
            ),
            row=1,
            col=2,
        )

        fig.add_hline(
            y=confidence_bound, line_dash="dash", line_color="firebrick", row=1, col=1
        )
        fig.add_hline(
            y=-confidence_bound, line_dash="dash", line_color="firebrick", row=1, col=1
        )
        fig.add_hline(
            y=confidence_bound, line_dash="dash", line_color="firebrick", row=1, col=2
        )
        fig.add_hline(
            y=-confidence_bound, line_dash="dash", line_color="firebrick", row=1, col=2
        )

        self.apply_default_layout(
            fig,
            plot_title or f"ACF and PACF for {feature}",
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.update_xaxes(title=xaxis_title, row=1, col=1)
        fig.update_xaxes(title=xaxis_title, row=1, col=2)
        fig.update_yaxes(title=yaxis_title, row=1, col=1)
        fig.update_yaxes(title=yaxis_title, row=1, col=2)
        fig.show("png", width=width, height=height)

    def plot_cross_correlation_heatmap(
        self,
        data: pd.DataFrame,
        date_feature: str,
        feature: str,
        comparison_features: Sequence[str],
        agg_freq: str = "ME",
        max_lag: int = 24,
        corr_method: str = "pearson",
        absolute_values: bool = False,
        plot_title: str = "",
        width: int = 1600,
        height: int = 850,
        xaxis_title: str = "Lag",
        yaxis_title: str = "Feature",
    ) -> None:
        """Create cross-correlation heatmap across lagged feature pairs.

        Args:
            data (pd.DataFrame): Input DataFrame.
            date_feature (str): Datetime column name.
            feature (str): Reference value feature.
            comparison_features (Sequence[str]): Features compared with lag shifts.
            agg_freq (str, optional): Frequency for pre-aggregation. Defaults to "ME".
            max_lag (int, optional): Maximum lag in each direction. Defaults to 24.
            corr_method (str, optional): Correlation method. Defaults to "pearson".
            absolute_values (bool, optional): Whether to display absolute values. Defaults to False.
            plot_title (str, optional): Plot title. Defaults to "".
            width (int, optional): Plot width. Defaults to 1600.
            height (int, optional): Plot height. Defaults to 850.
            xaxis_title (str, optional): X axis title. Defaults to "Lag".
            yaxis_title (str, optional): Y axis title. Defaults to "Feature".

        Raises:
            ValueError: If no comparison features are provided.
        """
        if len(comparison_features) == 0:
            raise ValueError("comparison_features must contain at least one feature")

        all_features = [feature] + list(comparison_features)
        data_copy = self._prepare_time_series_data(data, date_feature, all_features)
        resampled = data_copy.resample(agg_freq).mean().dropna(how="all")
        lags = list(range(-max_lag, max_lag + 1))

        heatmap_rows: list[list[float]] = []
        for compared_feature in comparison_features:
            row_values: list[float] = []
            for lag in lags:
                shifted_feature = resampled[compared_feature].shift(lag)
                corr_value = resampled[feature].corr(
                    shifted_feature, method=corr_method
                )
                if pd.isna(corr_value):
                    row_values.append(np.nan)
                else:
                    row_values.append(
                        abs(float(corr_value)) if absolute_values else float(corr_value)
                    )
            heatmap_rows.append(row_values)

        colorscale = "YlGnBu" if absolute_values else "RdBu"
        z_mid = None if absolute_values else 0

        fig = go.Figure(
            data=go.Heatmap(
                z=np.array(heatmap_rows),
                x=lags,
                y=list(comparison_features),
                colorscale=colorscale,
                zmid=z_mid,
                colorbar=dict(title="|corr|" if absolute_values else "corr"),
                text=np.round(np.array(heatmap_rows), 2),
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        self.apply_default_layout(
            fig,
            plot_title or f"Cross-Correlation Heatmap vs {feature}",
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(42)
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=offset) for offset in range(365)]

    trend = np.linspace(100, 200, 365)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)
    noise = np.random.normal(0, 10, 365)
    values_1 = trend + seasonal + noise
    values_2 = trend * 0.8 + seasonal * 1.5 + np.random.normal(0, 8, 365)

    sample_data = pd.DataFrame(
        {
            "date": dates,
            "metric1": values_1,
            "metric2": values_2,
            "metric3": np.random.normal(50, 15, 365),
            "store": np.random.choice(["A", "B", "C", "D", "E"], 365),
            "promotion": np.random.uniform(0, 100, 365),
        }
    )

    ts_plots = TimeSeriesPlots()
    ts_plots.plot_time_series_mean(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        agg_freq="ME",
        plot_title="Monthly Mean Time Series",
        line_color="rgb(77,97,143)",
        show_markers=True,
    )
    ts_plots.plot_time_series_multiple_metrics(
        data=sample_data,
        date_feature="date",
        features=["metric1", "metric2"],
        agg_freq="ME",
        agg_func="mean",
        plot_title="Multiple Metrics Comparison",
    )
    ts_plots.plot_time_series_with_trend(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        window_size=30,
        plot_title="Time Series with Moving Average",
    )
    ts_plots.plot_seasonal_decomposition(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        freq=90,
        plot_title="Seasonal Decomposition Analysis",
    )
    ts_plots.plot_time_series_boxplot(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        agg_freq="ME",
        plot_title="Distribution by Month",
    )
    ts_plots.plot_feature_distribution_by_category_over_time(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        category_feature="store",
        agg_freq="ME",
        top_n_categories=3,
        plot_title="Store Distribution Through Time",
    )
    ts_plots.plot_feature_distribution_by_numeric_feature_over_time(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        numeric_feature="promotion",
        n_bins=3,
        agg_freq="ME",
        plot_title="Promotion Bins Through Time",
    )
    ts_plots.plot_auto_and_partial_correlation(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        max_lag=24,
    )
    ts_plots.plot_cross_correlation_heatmap(
        data=sample_data,
        date_feature="date",
        feature="metric1",
        comparison_features=["metric2", "metric3"],
        agg_freq="ME",
        max_lag=10,
    )
