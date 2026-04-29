import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..base_plots import BasePlots


class TimeSeriesPlots(BasePlots):
    """Class for creating forecasting evaluation visualizations.

    This class provides plotting utilities for validating forecasting behavior over time,
    including actual-vs-predicted comparisons, residual diagnostics, prediction intervals,
    and rolling error trends.
    """

    def __init__(self) -> None:
        """Initialize forecasting-specific visual defaults."""
        super().__init__()
        self.default_time_series_colors = {
            "actual": "rgba(255, 167, 38, 0.98)",
            "predicted": "rgba(33, 150, 243, 0.98)",
            "future_forecast": "rgba(0, 121, 107, 0.98)",
            "split_line": "rgba(220, 53, 69, 0.95)",
            "residual_positive": "rgba(126, 189, 56, 0.95)",
            "residual_negative": "rgba(240, 84, 104, 0.95)",
            "interval_line": "rgba(47, 128, 237, 0.95)",
            "interval_fill": "rgba(47, 128, 237, 0.30)",
            "rolling_error": "rgba(28, 97, 180, 0.95)",
        }

    def _parse_datetime_with_dayfirst_fallback(
        self,
        values: pd.Index | pd.Series,
    ) -> pd.DatetimeIndex | pd.Series:
        """Parse datetime values with a day-first fallback for ambiguous strings."""
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

    def _validate_forecasting_inputs(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        time_index: pd.Index | pd.Series | np.ndarray | list[object],
    ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """Validate forecasting arrays and aligned time index.

        Note: O(n) time, O(n) space.

        Args:
            y_true (np.ndarray | pd.Series | list[object]): Observed target values.
            y_pred (np.ndarray | pd.Series | list[object]): Predicted target values.
            time_index (pd.Index | pd.Series | np.ndarray | list[object]): Timestamps.

        Returns:
            tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]: Validated and time-sorted arrays.

        Raises:
            ValueError: If lengths differ, input is empty, contains invalid values,
                or timestamps cannot be parsed.
        """
        y_true_array = np.asarray(y_true)
        y_pred_array = np.asarray(y_pred)
        index_array = np.asarray(time_index)

        if y_true_array.ndim > 1:
            y_true_array = y_true_array.squeeze()
        if y_pred_array.ndim > 1:
            y_pred_array = y_pred_array.squeeze()
        if index_array.ndim > 1:
            index_array = index_array.squeeze()

        if len(y_true_array) != len(y_pred_array):
            raise ValueError("y_true and y_pred must have the same length")
        if len(y_true_array) != len(index_array):
            raise ValueError(
                "time_index must have the same length as y_true and y_pred"
            )
        if len(y_true_array) == 0:
            raise ValueError("Input arrays cannot be empty")

        try:
            y_true_numeric = y_true_array.astype(float)
            y_pred_numeric = y_pred_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "y_true and y_pred must contain only numeric values"
            ) from exc

        parsed_index = self._parse_datetime_with_dayfirst_fallback(
            pd.Index(index_array)
        )
        if not isinstance(parsed_index, pd.DatetimeIndex):
            parsed_index = pd.DatetimeIndex(parsed_index)

        valid_mask = (
            np.isfinite(y_true_numeric)
            & np.isfinite(y_pred_numeric)
            & (~parsed_index.isna())
        )

        if not bool(np.any(valid_mask)):
            raise ValueError("No valid aligned observations available after filtering")

        y_true_valid = y_true_numeric[valid_mask]
        y_pred_valid = y_pred_numeric[valid_mask]
        index_valid = parsed_index[valid_mask]

        sort_order = np.argsort(index_valid.values)
        y_true_sorted = y_true_valid[sort_order]
        y_pred_sorted = y_pred_valid[sort_order]
        index_sorted = index_valid[sort_order]

        return y_true_sorted, y_pred_sorted, index_sorted

    def _calculate_metric_value(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric_name: str,
    ) -> float:
        """Calculate supported metric value in a case-insensitive way."""
        metric_lookup = {name.upper(): func for name, func in self.metrics_dict.items()}
        metric_key = metric_name.upper()

        if metric_key not in metric_lookup:
            raise ValueError(
                f"Unsupported metric: {metric_name}. Available metrics: "
                f"{list(self.metrics_dict.keys())}"
            )

        return float(metric_lookup[metric_key](y_true, y_pred))

    def _resolve_split_timestamp(
        self,
        split_point: int | str | pd.Timestamp | None,
        time_index: pd.DatetimeIndex,
    ) -> pd.Timestamp | None:
        """Resolve split specification into a concrete timestamp."""
        if split_point is None:
            return None

        if isinstance(split_point, (int, np.integer)):
            if split_point < 0 or split_point >= len(time_index):
                raise ValueError(
                    "split_point index must be between 0 and len(time_index)-1"
                )
            return pd.Timestamp(time_index[int(split_point)]).tz_convert(None)

        parsed_split = self._parse_datetime_with_dayfirst_fallback(
            pd.Index([split_point])
        )
        if parsed_split.isna().any():
            raise ValueError("split_point could not be parsed as datetime")

        split_ts = pd.Timestamp(parsed_split[0]).tz_convert(None)
        min_ts = pd.Timestamp(time_index.min()).tz_convert(None)
        max_ts = pd.Timestamp(time_index.max()).tz_convert(None)

        if split_ts < min_ts or split_ts > max_ts:
            raise ValueError(
                "split_point timestamp must lie within the time_index range"
            )

        return split_ts

    def _validate_prediction_interval(
        self,
        lower_bound: np.ndarray | pd.Series | list[object] | None,
        upper_bound: np.ndarray | pd.Series | list[object] | None,
        expected_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate lower and upper prediction interval bounds.

        Note: O(n) time, O(n) space.

        Args:
            lower_bound (np.ndarray | pd.Series | list[object] | None): Lower interval bound.
            upper_bound (np.ndarray | pd.Series | list[object] | None): Upper interval bound.
            expected_length (int): Required number of samples.

        Returns:
            tuple[np.ndarray, np.ndarray]: Validated lower and upper arrays.

        Raises:
            ValueError: If only one bound is provided, lengths mismatch,
                or lower bound exceeds upper bound.
        """
        if (lower_bound is None) != (upper_bound is None):
            raise ValueError(
                "Both lower_bound and upper_bound must be provided together"
            )

        if lower_bound is None or upper_bound is None:
            raise ValueError("Both lower_bound and upper_bound must be provided")

        lower_array = np.asarray(lower_bound)
        upper_array = np.asarray(upper_bound)

        if lower_array.ndim > 1:
            lower_array = lower_array.squeeze()
        if upper_array.ndim > 1:
            upper_array = upper_array.squeeze()

        if len(lower_array) != expected_length or len(upper_array) != expected_length:
            raise ValueError("Prediction interval bounds must match prediction length")

        try:
            lower_numeric = lower_array.astype(float)
            upper_numeric = upper_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Prediction interval bounds must contain numeric values"
            ) from exc

        if not np.all(lower_numeric <= upper_numeric):
            raise ValueError(
                "Each lower_bound value must be less than or equal to upper_bound"
            )

        return lower_numeric, upper_numeric

    def _validate_residual_inputs(
        self,
        residuals: np.ndarray | pd.Series | list[object],
        time_index: pd.Index | pd.Series | np.ndarray | list[object],
    ) -> tuple[np.ndarray, pd.DatetimeIndex]:
        """Validate residual values and aligned time index.

        Note: O(n) time, O(n) space.

        Args:
            residuals (np.ndarray | pd.Series | list[object]): Residual values.
            time_index (pd.Index | pd.Series | np.ndarray | list[object]): Timestamps.

        Returns:
            tuple[np.ndarray, pd.DatetimeIndex]: Validated residual values and sorted timestamps.

        Raises:
            ValueError: If lengths differ, input is empty, contains invalid values,
                or timestamps cannot be parsed.
        """
        residuals_array = np.asarray(residuals)
        index_array = np.asarray(time_index)

        if residuals_array.ndim > 1:
            residuals_array = residuals_array.squeeze()
        if index_array.ndim > 1:
            index_array = index_array.squeeze()

        if len(residuals_array) != len(index_array):
            raise ValueError("residuals and time_index must have the same length")
        if len(residuals_array) == 0:
            raise ValueError("residuals cannot be empty")

        try:
            residuals_numeric = residuals_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("residuals must contain only numeric values") from exc

        parsed_index = self._parse_datetime_with_dayfirst_fallback(
            pd.Index(index_array)
        )
        if not isinstance(parsed_index, pd.DatetimeIndex):
            parsed_index = pd.DatetimeIndex(parsed_index)

        valid_mask = np.isfinite(residuals_numeric) & (~parsed_index.isna())
        if not bool(np.any(valid_mask)):
            raise ValueError("No valid residual observations available after filtering")

        residuals_valid = residuals_numeric[valid_mask]
        index_valid = parsed_index[valid_mask]

        sort_order = np.argsort(index_valid.values)
        residuals_sorted = residuals_valid[sort_order]
        index_sorted = index_valid[sort_order]

        return residuals_sorted, index_sorted

    def _calculate_residual_metric_value(
        self,
        residuals: np.ndarray,
        metric_name: str,
    ) -> float:
        """Calculate residual-only metrics.

        Args:
            residuals (np.ndarray): Residual values.
            metric_name (str): Metric to compute (RMSE, MAE, or MSE).

        Returns:
            float: Calculated metric value.

        Raises:
            ValueError: If metric is not supported for residual-only mode.
        """
        metric_key = metric_name.upper()
        if metric_key == "RMSE":
            return float(np.sqrt(np.mean(np.square(residuals))))
        if metric_key == "MAE":
            return float(np.mean(np.abs(residuals)))
        if metric_key == "MSE":
            return float(np.mean(np.square(residuals)))
        raise ValueError(
            "When residuals are provided, metric_name must be one of: RMSE, MAE, MSE"
        )

    def _add_split_marker(
        self,
        fig: go.Figure,
        split_timestamp: pd.Timestamp | None,
        split_label: str,
        split_color: str,
        split_line_width: int,
    ) -> None:
        """Add a forecast split marker to a figure."""
        if split_timestamp is None:
            return

        fig.add_vline(
            x=split_timestamp,
            line_color=split_color,
            line_width=split_line_width,
            line_dash="dash",
        )
        fig.add_annotation(
            x=split_timestamp,
            y=1,
            xref="x",
            yref="paper",
            text=split_label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color=split_color, size=20),
        )

    def plot_actual_vs_predicted_over_time(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        time_index: pd.Index | pd.Series | np.ndarray | list[object],
        plot_title: str = "",
        width: int = 1600,
        height: int = 700,
        xaxis_title: str = "Date",
        yaxis_title: str = "Value",
        metric_name: str = "RMSE",
        metric_value: float | None = None,
        actual_color: str | None = None,
        predicted_color: str | None = None,
        actual_mode: str = "markers",
        actual_marker_symbol: str = "star",
        actual_marker_size: int = 10,
        actual_line_width: int = 1,
        predicted_line_width: int = 4,
        split_point: int | str | pd.Timestamp | None = None,
        split_label: str = "Forecast",
        split_color: str | None = None,
        split_line_width: int = 3,
        show_legend: bool = True,
    ) -> None:
        """Plot actual and predicted series over time.

        Note: O(n) time, O(n) space.

        Args:
            y_true (np.ndarray | pd.Series | list[object]): Observed target values.
            y_pred (np.ndarray | pd.Series | list[object]): Predicted target values.
            time_index (pd.Index | pd.Series | np.ndarray | list[object]): Timestamps.
            plot_title (str, optional): Plot title prefix. Defaults to "".
            width (int, optional): Figure width. Defaults to 1600.
            height (int, optional): Figure height. Defaults to 700.
            xaxis_title (str, optional): X-axis title. Defaults to "Date".
            yaxis_title (str, optional): Y-axis title. Defaults to "Value".
            metric_name (str, optional): Metric name shown in title. Defaults to "RMSE".
            metric_value (float | None, optional): Precomputed metric value. Defaults to None.
            actual_color (str | None, optional): Color for actual values. Defaults to None.
            predicted_color (str | None, optional): Color for predicted line. Defaults to None.
            actual_mode (str, optional): Plotly mode for actual trace. Defaults to "markers".
            actual_marker_symbol (str, optional): Plotly marker symbol for actual trace.
                Defaults to "star".
            actual_marker_size (int, optional): Marker size for actual trace. Defaults to 10.
            actual_line_width (int, optional): Actual trace line width when line mode is used.
                Defaults to 1.
            predicted_line_width (int, optional): Predicted line width. Defaults to 4.
            split_point (int | str | pd.Timestamp | None, optional): Forecast split location.
                Defaults to None.
            split_label (str, optional): Split label text. Defaults to "Forecast".
            split_color (str | None, optional): Color for split marker. Defaults to None.
            split_line_width (int, optional): Split line width. Defaults to 3.
            show_legend (bool, optional): Whether to show legend. Defaults to True.
        """
        y_true_array, y_pred_array, time_values = self._validate_forecasting_inputs(
            y_true,
            y_pred,
            time_index,
        )

        metric_display_value = metric_value
        if metric_display_value is None:
            metric_display_value = self._calculate_metric_value(
                y_true_array,
                y_pred_array,
                metric_name,
            )

        if plot_title:
            full_title = (
                f"{plot_title} ({metric_name.upper()}: {metric_display_value:.4f})"
            )
        else:
            full_title = (
                "Actual vs Predicted Over Time "
                f"({metric_name.upper()}: {metric_display_value:.4f})"
            )

        actual_line_color = actual_color or self.default_time_series_colors["actual"]
        predicted_line_color = (
            predicted_color or self.default_time_series_colors["predicted"]
        )
        split_line_color = split_color or self.default_time_series_colors["split_line"]
        split_timestamp = self._resolve_split_timestamp(split_point, time_values)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=y_true_array,
                mode=actual_mode,
                line=dict(color=actual_line_color, width=actual_line_width),
                marker=dict(
                    size=actual_marker_size,
                    symbol=actual_marker_symbol,
                    color=actual_line_color,
                ),
                name="Actual",
                showlegend=show_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=y_pred_array,
                mode="lines",
                line=dict(
                    color=predicted_line_color, width=predicted_line_width, dash="dot"
                ),
                name="Predicted",
                showlegend=show_legend,
            )
        )

        self._add_split_marker(
            fig,
            split_timestamp,
            split_label,
            split_line_color,
            split_line_width,
        )

        self.apply_default_layout(
            fig,
            full_title,
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.update_layout(showlegend=show_legend)
        fig.show("png", width=width, height=height)

    def plot_residuals_over_time(
        self,
        y_true: np.ndarray | pd.Series | list[object] | None = None,
        y_pred: np.ndarray | pd.Series | list[object] | None = None,
        time_index: pd.Index | pd.Series | np.ndarray | list[object] | None = None,
        residuals: np.ndarray | pd.Series | list[object] | None = None,
        plot_title: str = "",
        width: int = 1600,
        height: int = 700,
        xaxis_title: str = "Date",
        yaxis_title: str = "Residual",
        metric_name: str = "RMSE",
        metric_value: float | None = None,
        positive_color: str | None = None,
        negative_color: str | None = None,
        bar_opacity: float = 0.95,
        show_zero_line: bool = True,
        zero_line_color: str = "rgba(220, 53, 69, 0.80)",
        zero_line_width: int = 2,
        split_point: int | str | pd.Timestamp | None = None,
        split_label: str = "Forecast",
        split_color: str | None = None,
        split_line_width: int = 3,
    ) -> None:
        """Plot residuals over time as colored bars.

        Note: O(n) time, O(n) space.

        Args:
            y_true (np.ndarray | pd.Series | list[object] | None, optional): True values.
                Required with y_pred when residuals is not provided.
            y_pred (np.ndarray | pd.Series | list[object] | None, optional): Predicted values.
                Required with y_true when residuals is not provided.
            time_index (pd.Index | pd.Series | np.ndarray | list[object]): Timestamps.
            residuals (np.ndarray | pd.Series | list[object] | None, optional):
                Precomputed residual values. Defaults to None.

        Raises:
            ValueError: If input mode is ambiguous or incomplete.
        """
        if time_index is None:
            raise ValueError("time_index is required")

        if residuals is not None and (y_true is not None or y_pred is not None):
            raise ValueError(
                "Provide either residuals or y_true and y_pred, but not both"
            )
        if residuals is None and (y_true is None or y_pred is None):
            raise ValueError("Provide residuals or both y_true and y_pred")

        if residuals is not None:
            residual_values, time_values = self._validate_residual_inputs(
                residuals,
                time_index,
            )
            metric_display_value = metric_value
            if metric_display_value is None:
                metric_display_value = self._calculate_residual_metric_value(
                    residual_values,
                    metric_name,
                )
        else:
            y_true_array, y_pred_array, time_values = self._validate_forecasting_inputs(
                y_true,
                y_pred,
                time_index,
            )
            residual_values = y_true_array - y_pred_array
            metric_display_value = metric_value
            if metric_display_value is None:
                metric_display_value = self._calculate_metric_value(
                    y_true_array,
                    y_pred_array,
                    metric_name,
                )

        if plot_title:
            full_title = (
                f"{plot_title} ({metric_name.upper()}: {metric_display_value:.4f})"
            )
        else:
            full_title = (
                "Residuals Over Time "
                f"({metric_name.upper()}: {metric_display_value:.4f})"
            )

        resolved_positive = (
            positive_color or self.default_time_series_colors["residual_positive"]
        )
        resolved_negative = (
            negative_color or self.default_time_series_colors["residual_negative"]
        )
        bar_colors = [
            resolved_positive if value >= 0 else resolved_negative
            for value in residual_values
        ]

        split_line_color = split_color or self.default_time_series_colors["split_line"]
        split_timestamp = self._resolve_split_timestamp(split_point, time_values)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=time_values,
                y=residual_values,
                marker=dict(color=bar_colors),
                opacity=bar_opacity,
                showlegend=False,
                name="Residuals",
            )
        )

        if show_zero_line:
            fig.add_hline(
                y=0,
                line_color=zero_line_color,
                line_width=zero_line_width,
                line_dash="dash",
            )

        self._add_split_marker(
            fig,
            split_timestamp,
            split_label,
            split_line_color,
            split_line_width,
        )

        self.apply_default_layout(
            fig,
            full_title,
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def plot_prediction_interval_fan(
        self,
        y_pred: np.ndarray | pd.Series | list[object],
        time_index: pd.Index | pd.Series | np.ndarray | list[object],
        lower_bound: np.ndarray | pd.Series | list[object],
        upper_bound: np.ndarray | pd.Series | list[object],
        y_true: np.ndarray | pd.Series | list[object] | None = None,
        plot_title: str = "",
        width: int = 1600,
        height: int = 700,
        xaxis_title: str = "Date",
        yaxis_title: str = "Value",
        actual_mode: str = "markers",
        actual_marker_symbol: str = "star",
        actual_marker_size: int = 10,
        predicted_color: str | None = None,
        interval_fill_color: str | None = None,
        interval_line_color: str | None = None,
        forecast_line_width: int = 4,
        split_point: int | str | pd.Timestamp | None = None,
        split_label: str = "Forecast",
        split_color: str | None = None,
        split_line_width: int = 3,
        show_legend: bool = True,
    ) -> None:
        """Plot forecast line with a shaded prediction interval band.

        Note: O(n) time, O(n) space.
        """
        y_pred_array = np.asarray(y_pred)
        if y_pred_array.ndim > 1:
            y_pred_array = y_pred_array.squeeze()

        if len(y_pred_array) == 0:
            raise ValueError("y_pred cannot be empty")

        try:
            y_pred_numeric = y_pred_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_pred must contain only numeric values") from exc

        parsed_index = self._parse_datetime_with_dayfirst_fallback(pd.Index(time_index))
        if not isinstance(parsed_index, pd.DatetimeIndex):
            parsed_index = pd.DatetimeIndex(parsed_index)

        if len(parsed_index) != len(y_pred_numeric):
            raise ValueError("time_index must have the same length as y_pred")

        valid_index_mask = ~parsed_index.isna()
        if not bool(np.any(valid_index_mask)):
            raise ValueError("No valid datetime values in time_index")

        time_valid = parsed_index[valid_index_mask]
        y_pred_valid = y_pred_numeric[valid_index_mask]

        sort_order = np.argsort(time_valid.values)
        time_values = time_valid[sort_order]
        y_pred_sorted = y_pred_valid[sort_order]

        lower_numeric, upper_numeric = self._validate_prediction_interval(
            lower_bound,
            upper_bound,
            len(y_pred_numeric),
        )
        lower_sorted = lower_numeric[valid_index_mask][sort_order]
        upper_sorted = upper_numeric[valid_index_mask][sort_order]

        y_true_sorted: np.ndarray | None = None
        if y_true is not None:
            y_true_array = np.asarray(y_true)
            if y_true_array.ndim > 1:
                y_true_array = y_true_array.squeeze()
            if len(y_true_array) != len(y_pred_numeric):
                raise ValueError("y_true must have the same length as y_pred")
            try:
                y_true_numeric = y_true_array.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("y_true must contain only numeric values") from exc
            y_true_sorted = y_true_numeric[valid_index_mask][sort_order]

        resolved_predicted_color = (
            predicted_color or self.default_time_series_colors["interval_line"]
        )
        resolved_interval_fill = (
            interval_fill_color or self.default_time_series_colors["interval_fill"]
        )
        resolved_interval_line = (
            interval_line_color or self.default_time_series_colors["interval_line"]
        )

        split_line_color = split_color or self.default_time_series_colors["split_line"]
        split_timestamp = self._resolve_split_timestamp(split_point, time_values)

        full_title = plot_title or "Forecast with Prediction Interval"

        fig = go.Figure()
        if y_true_sorted is not None:
            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=y_true_sorted,
                    mode=actual_mode,
                    line=dict(color=self.default_time_series_colors["actual"], width=1),
                    marker=dict(
                        size=actual_marker_size,
                        symbol=actual_marker_symbol,
                        color=self.default_time_series_colors["actual"],
                    ),
                    name="Actual",
                    showlegend=show_legend,
                )
            )

        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=upper_sorted,
                mode="lines",
                line=dict(color=resolved_interval_line, width=0),
                name="Upper Bound",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=lower_sorted,
                mode="lines",
                fill="tonexty",
                fillcolor=resolved_interval_fill,
                line=dict(color=resolved_interval_line, width=0),
                name="Prediction Interval",
                showlegend=show_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=y_pred_sorted,
                mode="lines+markers",
                line=dict(color=resolved_predicted_color, width=forecast_line_width),
                marker=dict(size=6, color=resolved_predicted_color),
                name="Forecast",
                showlegend=show_legend,
            )
        )

        self._add_split_marker(
            fig,
            split_timestamp,
            split_label,
            split_line_color,
            split_line_width,
        )

        self.apply_default_layout(
            fig,
            full_title,
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.update_layout(showlegend=show_legend)
        fig.show("png", width=width, height=height)

    def plot_rolling_error_over_time(
        self,
        y_true: np.ndarray | pd.Series | list[object] | None = None,
        y_pred: np.ndarray | pd.Series | list[object] | None = None,
        time_index: pd.Index | pd.Series | np.ndarray | list[object] | None = None,
        residuals: np.ndarray | pd.Series | list[object] | None = None,
        rolling_window: int = 7,
        rolling_metric: str = "RMSE",
        plot_title: str = "",
        width: int = 1600,
        height: int = 700,
        xaxis_title: str = "Date",
        yaxis_title: str = "Rolling Error",
        line_color: str | None = None,
        line_width: int = 3,
        split_point: int | str | pd.Timestamp | None = None,
        split_label: str = "Forecast",
        split_color: str | None = None,
        split_line_width: int = 3,
    ) -> None:
        """Plot rolling forecasting error over time.

        Note: O(n) time, O(n) space.

        Args:
            y_true (np.ndarray | pd.Series | list[object] | None, optional): True values.
                Required with y_pred when residuals is not provided.
            y_pred (np.ndarray | pd.Series | list[object] | None, optional): Predicted values.
                Required with y_true when residuals is not provided.
            time_index (pd.Index | pd.Series | np.ndarray | list[object]): Timestamps.
            residuals (np.ndarray | pd.Series | list[object] | None, optional):
                Precomputed residual values. Defaults to None.

        Raises:
            ValueError: If input mode is ambiguous or incomplete.
        """
        if time_index is None:
            raise ValueError("time_index is required")

        if residuals is not None and (y_true is not None or y_pred is not None):
            raise ValueError(
                "Provide either residuals or y_true and y_pred, but not both"
            )
        if residuals is None and (y_true is None or y_pred is None):
            raise ValueError("Provide residuals or both y_true and y_pred")

        if residuals is not None:
            residual_values, time_values = self._validate_residual_inputs(
                residuals,
                time_index,
            )
        else:
            y_true_array, y_pred_array, time_values = self._validate_forecasting_inputs(
                y_true,
                y_pred,
                time_index,
            )
            residual_values = y_true_array - y_pred_array

        if rolling_window <= 0:
            raise ValueError("rolling_window must be a positive integer")
        if rolling_window > len(residual_values):
            raise ValueError("rolling_window cannot exceed the number of observations")

        metric_upper = rolling_metric.upper()
        if metric_upper == "RMSE":
            base_errors = np.square(residual_values)
            rolling_error = np.sqrt(
                pd.Series(base_errors)
                .rolling(window=rolling_window, min_periods=rolling_window)
                .mean()
            )
        elif metric_upper == "MAE":
            base_errors = np.abs(residual_values)
            rolling_error = (
                pd.Series(base_errors)
                .rolling(
                    window=rolling_window,
                    min_periods=rolling_window,
                )
                .mean()
            )
        else:
            raise ValueError("rolling_metric must be one of: RMSE, MAE")

        valid_mask = ~rolling_error.isna().to_numpy()
        x_values = time_values[valid_mask]
        y_values = rolling_error.to_numpy()[valid_mask]

        if plot_title:
            full_title = plot_title
        else:
            full_title = f"Rolling {metric_upper} Over Time (window={rolling_window})"

        split_line_color = split_color or self.default_time_series_colors["split_line"]
        split_timestamp = self._resolve_split_timestamp(split_point, time_values)

        resolved_line_color = (
            line_color or self.default_time_series_colors["rolling_error"]
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="lines",
                line=dict(color=resolved_line_color, width=line_width),
                name=f"Rolling {metric_upper}",
                showlegend=False,
            )
        )

        self._add_split_marker(
            fig,
            split_timestamp,
            split_label,
            split_line_color,
            split_line_width,
        )

        self.apply_default_layout(
            fig,
            full_title,
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def plot_historical_and_future_forecast(
        self,
        y_true: np.ndarray | pd.Series | list[object],
        y_pred: np.ndarray | pd.Series | list[object],
        y_forecast: np.ndarray | pd.Series | list[object],
        historical_time_index: pd.Index | pd.Series | np.ndarray | list[object],
        forecast_time_index: pd.Index | pd.Series | np.ndarray | list[object],
        plot_title: str = "",
        width: int = 1600,
        height: int = 700,
        xaxis_title: str = "Date",
        yaxis_title: str = "Value",
        metric_name: str = "RMSE",
        metric_value: float | None = None,
        actual_color: str | None = None,
        predicted_color: str | None = None,
        forecast_color: str | None = None,
        actual_mode: str = "markers",
        actual_marker_symbol: str = "star",
        actual_marker_size: int = 10,
        predicted_line_width: int = 4,
        forecast_line_width: int = 5,
        split_label: str = "Forecast",
        split_color: str | None = None,
        split_line_width: int = 3,
        show_legend: bool = True,
    ) -> None:
        """Plot historical actual/predicted values with future forecast only values.

        Note: O(n) time, O(n) space.

        Args:
            y_true (np.ndarray | pd.Series | list[object]): Historical actual values.
            y_pred (np.ndarray | pd.Series | list[object]): Historical predicted values.
            y_forecast (np.ndarray | pd.Series | list[object]): Future forecast-only values.
            historical_time_index (pd.Index | pd.Series | np.ndarray | list[object]):
                Historical timestamps.
            forecast_time_index (pd.Index | pd.Series | np.ndarray | list[object]):
                Future timestamps.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Figure width. Defaults to 1600.
            height (int, optional): Figure height. Defaults to 700.
            xaxis_title (str, optional): X-axis title. Defaults to "Date".
            yaxis_title (str, optional): Y-axis title. Defaults to "Value".
            metric_name (str, optional): Metric computed on historical fit. Defaults to "RMSE".
            metric_value (float | None, optional): Precomputed metric value. Defaults to None.
            actual_color (str | None, optional): Actual trace color. Defaults to None.
            predicted_color (str | None, optional): Historical predicted line color.
                Defaults to None.
            forecast_color (str | None, optional): Future forecast line color. Defaults to None.
            actual_mode (str, optional): Plotly mode for actual trace. Defaults to "markers".
            actual_marker_symbol (str, optional): Marker symbol for actual trace.
                Defaults to "star".
            actual_marker_size (int, optional): Marker size for actual trace. Defaults to 10.
            predicted_line_width (int, optional): Historical predicted line width. Defaults to 4.
            forecast_line_width (int, optional): Future forecast line width. Defaults to 5.
            split_label (str, optional): Split marker label. Defaults to "Forecast".
            split_color (str | None, optional): Split marker color. Defaults to None.
            split_line_width (int, optional): Split marker line width. Defaults to 3.
            show_legend (bool, optional): Whether to show legend. Defaults to True.

        Raises:
            ValueError: If forecast arrays are empty, non-numeric, have mismatched lengths,
                contain invalid datetimes, or overlap with historical range.
        """
        y_true_sorted, y_pred_sorted, historical_index = (
            self._validate_forecasting_inputs(
                y_true,
                y_pred,
                historical_time_index,
            )
        )

        y_forecast_array = np.asarray(y_forecast)
        if y_forecast_array.ndim > 1:
            y_forecast_array = y_forecast_array.squeeze()
        if len(y_forecast_array) == 0:
            raise ValueError("y_forecast cannot be empty")

        try:
            y_forecast_numeric = y_forecast_array.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_forecast must contain only numeric values") from exc

        forecast_index_parsed = self._parse_datetime_with_dayfirst_fallback(
            pd.Index(forecast_time_index)
        )
        if not isinstance(forecast_index_parsed, pd.DatetimeIndex):
            forecast_index_parsed = pd.DatetimeIndex(forecast_index_parsed)

        if len(forecast_index_parsed) != len(y_forecast_numeric):
            raise ValueError(
                "forecast_time_index must have the same length as y_forecast"
            )

        valid_forecast_mask = (~forecast_index_parsed.isna()) & np.isfinite(
            y_forecast_numeric
        )
        if not bool(np.any(valid_forecast_mask)):
            raise ValueError(
                "No valid future forecast observations available after filtering"
            )

        forecast_values = y_forecast_numeric[valid_forecast_mask]
        forecast_index_values = forecast_index_parsed[valid_forecast_mask]
        forecast_sort = np.argsort(forecast_index_values.values)
        forecast_values_sorted = forecast_values[forecast_sort]
        forecast_index_sorted = forecast_index_values[forecast_sort]

        historical_max = pd.Timestamp(historical_index.max()).tz_convert(None)
        future_min = pd.Timestamp(forecast_index_sorted.min()).tz_convert(None)
        if future_min <= historical_max:
            raise ValueError(
                "forecast_time_index must start strictly after historical_time_index"
            )

        resolved_metric = metric_value
        if resolved_metric is None:
            resolved_metric = self._calculate_metric_value(
                y_true_sorted,
                y_pred_sorted,
                metric_name,
            )

        if plot_title:
            full_title = f"{plot_title} ({metric_name.upper()}: {resolved_metric:.4f})"
        else:
            full_title = (
                "Historical Fit and Future Forecast "
                f"({metric_name.upper()}: {resolved_metric:.4f})"
            )

        actual_line_color = actual_color or self.default_time_series_colors["actual"]
        predicted_line_color = (
            predicted_color or self.default_time_series_colors["predicted"]
        )
        future_line_color = (
            forecast_color or self.default_time_series_colors["future_forecast"]
        )
        split_line_color = split_color or self.default_time_series_colors["split_line"]
        split_timestamp = pd.Timestamp(forecast_index_sorted.min()).tz_convert(None)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=historical_index,
                y=y_true_sorted,
                mode=actual_mode,
                line=dict(color=actual_line_color, width=1),
                marker=dict(
                    size=actual_marker_size,
                    symbol=actual_marker_symbol,
                    color=actual_line_color,
                ),
                name="Actual",
                showlegend=show_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=historical_index,
                y=y_pred_sorted,
                mode="lines",
                line=dict(
                    color=predicted_line_color, width=predicted_line_width, dash="dot"
                ),
                name="Predicted (Historical)",
                showlegend=show_legend,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_index_sorted,
                y=forecast_values_sorted,
                mode="lines+markers",
                line=dict(color=future_line_color, width=forecast_line_width),
                marker=dict(size=7, color=future_line_color),
                name="Forecast (Future)",
                showlegend=show_legend,
            )
        )

        self._add_split_marker(
            fig,
            split_timestamp,
            split_label,
            split_line_color,
            split_line_width,
        )

        self.apply_default_layout(
            fig,
            full_title,
            width,
            height,
            xaxis_title,
            yaxis_title,
        )
        fig.update_layout(showlegend=show_legend)
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(42)
    N_POINTS = 60
    time_points = pd.date_range("2024-01-01", periods=N_POINTS, freq="W")
    signal = np.cumsum(np.random.normal(loc=0.25, scale=0.8, size=N_POINTS)) + 20
    forecast = signal + np.random.normal(loc=0.0, scale=0.6, size=N_POINTS)
    lower = forecast - 1.8
    upper = forecast + 1.8

    plots = TimeSeriesPlots()
    plots.plot_actual_vs_predicted_over_time(
        y_true=signal,
        y_pred=forecast,
        time_index=time_points,
        split_point=40,
        width=900,
        height=500,
    )
    plots.plot_residuals_over_time(
        y_true=signal,
        y_pred=forecast,
        time_index=time_points,
        split_point=40,
        width=900,
        height=500,
    )
    plots.plot_prediction_interval_fan(
        y_pred=forecast,
        time_index=time_points,
        lower_bound=lower,
        upper_bound=upper,
        y_true=signal,
        split_point=40,
        width=900,
        height=500,
    )
    plots.plot_rolling_error_over_time(
        y_true=signal,
        y_pred=forecast,
        time_index=time_points,
        rolling_window=6,
        rolling_metric="RMSE",
        split_point=40,
        width=900,
        height=500,
    )

    FUTURE_POINTS = 12
    future_index = pd.date_range(
        time_points[-1] + pd.Timedelta(weeks=1), periods=FUTURE_POINTS, freq="W"
    )
    future_forecast = np.linspace(forecast[-1] + 0.2, forecast[-1] + 2.6, FUTURE_POINTS)
    plots.plot_historical_and_future_forecast(
        y_true=signal,
        y_pred=forecast,
        y_forecast=future_forecast,
        historical_time_index=time_points,
        forecast_time_index=future_index,
        split_label="Forecast ->",
        width=900,
        height=500,
    )
