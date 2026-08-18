import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from kvbiii_plots.evaluation.time_series_plots import TimeSeriesPlots


@pytest.fixture
def forecast_data(
    test_settings: object,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Provide synthetic forecasting data for plotting tests."""
    np.random.seed(test_settings.SEED)
    n_points = 80
    time_index = pd.date_range("2024-01-01", periods=n_points, freq="D")
    y_true = np.cumsum(np.random.normal(0.2, 0.8, size=n_points)) + 30
    y_pred = y_true + np.random.normal(0, 0.6, size=n_points)
    return y_true, y_pred, time_index


@pytest.fixture
def forecast_data_lists(
    test_settings: object,
) -> tuple[list[float], list[float], list[str]]:
    """Provide synthetic forecasting data as lists for validation tests."""
    np.random.seed(test_settings.SEED)
    n_points = 25
    time_index = pd.date_range("2024-05-01", periods=n_points, freq="D")
    y_true = (np.cumsum(np.random.normal(0.1, 1.0, size=n_points)) + 10).tolist()
    y_pred = (np.array(y_true) + np.random.normal(0, 0.5, size=n_points)).tolist()
    time_list = time_index.strftime("%Y-%m-%d").tolist()
    return y_true, y_pred, time_list


@pytest.fixture
def interval_bounds(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> tuple[np.ndarray, np.ndarray]:
    """Provide synthetic prediction interval bounds."""
    _, y_pred, _ = forecast_data
    lower_bound = y_pred - 1.4
    upper_bound = y_pred + 1.4
    return lower_bound, upper_bound


@pytest.fixture
def future_forecast_data(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Provide synthetic future-only forecast values and timestamps."""
    _, y_pred, time_index = forecast_data
    n_future = 16
    last_value = float(y_pred[-1])
    y_future = np.linspace(last_value + 0.2, last_value + 3.2, n_future)
    future_index = pd.date_range(
        time_index[-1] + pd.Timedelta(days=1), periods=n_future, freq="D"
    )
    return y_future, future_index


def test_timeseriesplots_initialization() -> None:
    """Test class initialization and default configurations."""
    ts_plots = TimeSeriesPlots()

    if not (hasattr(ts_plots, "default_time_series_colors")):
        raise AssertionError("Assertion failed.")
    if not ("actual" in ts_plots.default_time_series_colors):
        raise AssertionError("Assertion failed.")
    if not ("predicted" in ts_plots.default_time_series_colors):
        raise AssertionError("Assertion failed.")
    if not ("split_line" in ts_plots.default_time_series_colors):
        raise AssertionError("Assertion failed.")
    if not ("rolling_error" in ts_plots.default_time_series_colors):
        raise AssertionError("Assertion failed.")


def test_timeseriesplots_validate_forecasting_inputs_handles_lists(
    forecast_data_lists: tuple[list[float], list[float], list[str]],
) -> None:
    """Test input validation with list-based values and string timestamps."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data_lists

    y_true_out, y_pred_out, index_out = ts_plots._validate_forecasting_inputs(
        y_true,
        y_pred,
        time_index,
    )

    if not (isinstance(y_true_out, np.ndarray)):
        raise AssertionError("Assertion failed.")
    if not (isinstance(y_pred_out, np.ndarray)):
        raise AssertionError("Assertion failed.")
    if not (isinstance(index_out, pd.DatetimeIndex)):
        raise AssertionError("Assertion failed.")
    if not (len(y_true_out) == len(y_pred_out) == len(index_out)):
        raise AssertionError("Assertion failed.")


def test_timeseriesplots_validate_forecasting_inputs_sorts_by_time(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test that validation sorts values by time while preserving alignment."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    shuffled_order = np.arange(len(time_index))[::-1]
    shuffled_true = y_true[shuffled_order]
    shuffled_pred = y_pred[shuffled_order]
    shuffled_time = time_index[shuffled_order]

    _, _, sorted_time = ts_plots._validate_forecasting_inputs(
        shuffled_true,
        shuffled_pred,
        shuffled_time,
    )

    if not (sorted_time.is_monotonic_increasing):
        raise AssertionError("Assertion failed.")


def test_timeseriesplots_validate_forecasting_inputs_raises_for_mismatched_lengths() -> (
    None
):
    """Test validation error for mismatched input lengths."""
    ts_plots = TimeSeriesPlots()

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        ts_plots._validate_forecasting_inputs(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.0, 2.0]),
            time_index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )


def test_timeseriesplots_validate_forecasting_inputs_raises_for_non_numeric_values() -> (
    None
):
    """Test validation error for non-numeric target arrays."""
    ts_plots = TimeSeriesPlots()

    with pytest.raises(
        ValueError,
        match="y_true and y_pred must contain only numeric values",
    ):
        ts_plots._validate_forecasting_inputs(
            y_true=np.array(["a", "b", "c"]),
            y_pred=np.array([1.0, 2.0, 3.0]),
            time_index=pd.date_range("2024-01-01", periods=3, freq="D"),
        )


def test_timeseriesplots_resolve_split_timestamp_raises_for_out_of_range_index(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test split point index bounds validation."""
    ts_plots = TimeSeriesPlots()
    _, _, time_index = forecast_data

    with pytest.raises(
        ValueError,
        match=r"split_point index must be between 0 and len\(time_index\)-1",
    ):
        ts_plots._resolve_split_timestamp(len(time_index), time_index)


def test_timeseriesplots_validate_prediction_interval_raises_for_partial_bounds(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test interval validation error when only one bound is provided."""
    ts_plots = TimeSeriesPlots()
    _, y_pred, _ = forecast_data

    with pytest.raises(
        ValueError,
        match="Both lower_bound and upper_bound must be provided together",
    ):
        ts_plots._validate_prediction_interval(
            lower_bound=y_pred - 1.0,
            upper_bound=None,
            expected_length=len(y_pred),
        )


def test_timeseriesplots_plot_actual_vs_predicted_over_time_executes(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test that actual-vs-predicted plot executes with custom settings."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    ts_plots.plot_actual_vs_predicted_over_time(
        y_true=y_true,
        y_pred=y_pred,
        time_index=time_index,
        metric_name="RMSE",
        split_point=50,
        width=800,
        height=500,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_actual_vs_predicted_uses_serializable_datetime_values(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Plotly receives serializable values for datetime plot elements."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    captured_figures: list[go.Figure] = []

    def capture_figure(figure: go.Figure, *args: object, **kwargs: object) -> None:
        captured_figures.append(figure)

    monkeypatch.setattr(go.Figure, "show", capture_figure)

    ts_plots.plot_actual_vs_predicted_over_time(
        y_true=y_true,
        y_pred=y_pred,
        time_index=time_index,
        split_point=50,
    )

    figure = captured_figures[0]
    assert all(isinstance(value, str) for value in figure.data[0].x)
    assert all(isinstance(value, str) for value in figure.data[1].x)
    assert isinstance(figure.layout.shapes[0].x0, str)
    assert isinstance(figure.layout.annotations[0].x, str)


def test_timeseriesplots_plot_residuals_over_time_executes(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test that residual-over-time plot executes and supports split marker."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    ts_plots.plot_residuals_over_time(
        y_true=y_true,
        y_pred=y_pred,
        time_index=time_index,
        split_point=time_index[40],
        width=800,
        height=450,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_residuals_over_time_accepts_precomputed_residuals(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test residual plotting with precomputed residual input only."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    residual_values = y_true - y_pred

    ts_plots.plot_residuals_over_time(
        residuals=residual_values,
        time_index=time_index,
        metric_name="RMSE",
        split_point=30,
        width=800,
        height=450,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_residuals_over_time_raises_for_ambiguous_inputs(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test residual plotting rejects mixed input modes."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    residual_values = y_true - y_pred

    with pytest.raises(
        ValueError,
        match="Provide either residuals or y_true and y_pred, but not both",
    ):
        ts_plots.plot_residuals_over_time(
            y_true=y_true,
            y_pred=y_pred,
            residuals=residual_values,
            time_index=time_index,
        )


def test_timeseriesplots_plot_prediction_interval_fan_executes(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
    interval_bounds: tuple[np.ndarray, np.ndarray],
) -> None:
    """Test that prediction interval fan chart executes with actual overlay."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    lower_bound, upper_bound = interval_bounds

    ts_plots.plot_prediction_interval_fan(
        y_pred=y_pred,
        time_index=time_index,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        y_true=y_true,
        split_point=45,
        width=900,
        height=500,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_prediction_interval_fan_raises_when_bounds_invalid(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test interval error when lower bound is greater than upper bound."""
    ts_plots = TimeSeriesPlots()
    _, y_pred, time_index = forecast_data

    lower_bound = y_pred + 0.5
    upper_bound = y_pred - 0.5

    with pytest.raises(
        ValueError,
        match="Each lower_bound value must be less than or equal to upper_bound",
    ):
        ts_plots.plot_prediction_interval_fan(
            y_pred=y_pred,
            time_index=time_index,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )


def test_timeseriesplots_plot_rolling_error_over_time_executes_for_rmse_and_mae(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test rolling error plot for both RMSE and MAE options."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    ts_plots.plot_rolling_error_over_time(
        y_true=y_true,
        y_pred=y_pred,
        time_index=time_index,
        rolling_window=7,
        rolling_metric="RMSE",
        split_point=40,
        width=900,
        height=450,
    )

    ts_plots.plot_rolling_error_over_time(
        y_true=y_true,
        y_pred=y_pred,
        time_index=time_index,
        rolling_window=7,
        rolling_metric="MAE",
        split_point=40,
        width=900,
        height=450,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_rolling_error_over_time_accepts_precomputed_residuals(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test rolling error plotting with precomputed residual input only."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    residual_values = y_true - y_pred

    ts_plots.plot_rolling_error_over_time(
        residuals=residual_values,
        time_index=time_index,
        rolling_window=7,
        rolling_metric="RMSE",
        split_point=40,
        width=900,
        height=450,
    )

    ts_plots.plot_rolling_error_over_time(
        residuals=residual_values,
        time_index=time_index,
        rolling_window=7,
        rolling_metric="MAE",
        split_point=40,
        width=900,
        height=450,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_rolling_error_over_time_raises_for_ambiguous_inputs(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test rolling error plotting rejects mixed input modes."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data
    residual_values = y_true - y_pred

    with pytest.raises(
        ValueError,
        match="Provide either residuals or y_true and y_pred, but not both",
    ):
        ts_plots.plot_rolling_error_over_time(
            y_true=y_true,
            y_pred=y_pred,
            residuals=residual_values,
            time_index=time_index,
        )


def test_timeseriesplots_plot_rolling_error_over_time_raises_for_invalid_metric(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test rolling error validation for unsupported rolling metric."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    with pytest.raises(ValueError, match="rolling_metric must be one of: RMSE, MAE"):
        ts_plots.plot_rolling_error_over_time(
            y_true=y_true,
            y_pred=y_pred,
            time_index=time_index,
            rolling_window=7,
            rolling_metric="R2",
        )


def test_timeseriesplots_plot_rolling_error_over_time_raises_for_invalid_window(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test rolling window validation for positive bounded integer window."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, time_index = forecast_data

    with pytest.raises(ValueError, match="rolling_window must be a positive integer"):
        ts_plots.plot_rolling_error_over_time(
            y_true=y_true,
            y_pred=y_pred,
            time_index=time_index,
            rolling_window=0,
        )


def test_timeseriesplots_plot_historical_and_future_forecast_executes(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
    future_forecast_data: tuple[np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test historical actual/predicted plus future forecast plot execution."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, historical_index = forecast_data
    y_forecast, future_index = future_forecast_data

    ts_plots.plot_historical_and_future_forecast(
        y_true=y_true,
        y_pred=y_pred,
        y_forecast=y_forecast,
        historical_time_index=historical_index,
        forecast_time_index=future_index,
        split_label="forecast ->",
        width=900,
        height=500,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_timeseriesplots_plot_historical_and_future_forecast_raises_for_overlap(
    forecast_data: tuple[np.ndarray, np.ndarray, pd.DatetimeIndex],
    future_forecast_data: tuple[np.ndarray, pd.DatetimeIndex],
) -> None:
    """Test forecast time range validation for overlap with historical index."""
    ts_plots = TimeSeriesPlots()
    y_true, y_pred, historical_index = forecast_data
    y_forecast, future_index = future_forecast_data

    overlapping_index = future_index.copy()
    overlapping_index = overlapping_index.insert(0, historical_index[-1])
    overlapping_forecast = np.insert(y_forecast, 0, y_pred[-1])

    with pytest.raises(
        ValueError,
        match="forecast_time_index must start strictly after historical_time_index",
    ):
        ts_plots.plot_historical_and_future_forecast(
            y_true=y_true,
            y_pred=y_pred,
            y_forecast=overlapping_forecast,
            historical_time_index=historical_index,
            forecast_time_index=overlapping_index,
        )
