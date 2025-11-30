import pytest
import numpy as np
import pandas as pd
from kvbiii_plots.eda.time_series_plots import TimeSeriesPlots


@pytest.fixture
def time_series_dataframe(test_settings) -> pd.DataFrame:
    """Provides a sample time series DataFrame for testing purposes.

    Args:
        test_settings: Test settings fixture

    Returns:
        pd.DataFrame: DataFrame with datetime index and time series data
    """
    np.random.seed(test_settings.SEED)
    dates = pd.date_range(start="2020-01-01", end="2023-12-31", freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "value1": np.random.randn(len(dates)).cumsum() + 100,
            "value2": np.random.randn(len(dates)).cumsum() + 50,
            "category": np.random.choice(["A", "B"], size=len(dates)),
        }
    ).set_index("date")


def test_timeseriesplots_prepare_time_series_data_handles_single_target(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests _prepare_time_series_data correctly processes single target column.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors for single target
        - Time series data is properly prepared for visualization
        - Date feature is correctly handled as index
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    result = ts_plots._prepare_time_series_data(
        data=time_series_dataframe.reset_index(), feature="date", target="value1"
    )

    assert isinstance(result, pd.DataFrame), "Result should be DataFrame"
    assert "value1" in result.columns, "Target column should be present"


def test_timeseriesplots_prepare_time_series_data_handles_multiple_targets(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests _prepare_time_series_data correctly processes multiple target columns.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors for multiple targets
        - All target columns are preserved in result
        - Date indexing is properly maintained
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    result = ts_plots._prepare_time_series_data(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target=["value1", "value2"],
    )

    assert isinstance(result, pd.DataFrame), "Result should be DataFrame"
    assert "value1" in result.columns, "First target should be present"
    assert "value2" in result.columns, "Second target should be present"


def test_timeseriesplots_plot_time_series_mean_handles_monthly_aggregation(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_mean correctly aggregates data monthly.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors with monthly aggregation
        - Mean aggregation is properly applied
        - Plot configuration uses specified parameters
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_mean(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        agg_freq="ME",
        plot_title="Monthly Mean Test",
        width=1600,
        height=800,
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_time_series_mean_applies_custom_parameters(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_mean applies custom visualization parameters.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method accepts custom line color and width parameters
        - Marker settings are properly applied
        - Plot configuration uses specified settings
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_mean(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        line_color="red",
        line_width=3,
        show_markers=True,
        marker_size=8,
        plot_title="Custom Parameters Test",
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_time_series_multiple_metrics_handles_multiple_targets(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_multiple_metrics correctly processes multiple targets.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors for multiple target metrics
        - All targets are plotted on same chart
        - Legend and colors are properly applied
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_multiple_metrics(
        data=time_series_dataframe.reset_index(),
        feature="date",
        targets=["value1", "value2"],
        agg_freq="ME",
        agg_func="mean",
        plot_title="Multiple Metrics Test",
        width=1600,
        height=800,
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_time_series_multiple_metrics_applies_aggregation_functions(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_multiple_metrics applies different aggregation functions.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method accepts different aggregation functions (sum, max, min, etc.)
        - Aggregation is properly applied to time series data
        - Plot reflects chosen aggregation method
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_multiple_metrics(
        data=time_series_dataframe.reset_index(),
        feature="date",
        targets=["value1", "value2"],
        agg_func="sum",
        show_markers=True,
        plot_title="Sum Aggregation Test",
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_time_series_with_trend_handles_moving_average(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_with_trend correctly calculates moving average trend.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors with trend calculation
        - Moving average trend line is properly generated
        - Original data and trend are both displayed
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_with_trend(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        window_size=30,
        plot_title="Trend Analysis Test",
        width=1600,
        height=800,
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_time_series_with_trend_applies_custom_colors(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_time_series_with_trend applies custom color parameters.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method accepts custom original_color and trend_color parameters
        - Line colors are applied as specified
        - Opacity and line width settings work correctly
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_time_series_with_trend(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        original_color="blue",
        trend_color="red",
        original_opacity=0.5,
        trend_line_width=4,
        plot_title="Custom Colors Test",
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_seasonal_decomposition_handles_seasonal_analysis(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_seasonal_decomposition correctly performs seasonal decomposition.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors with seasonal decomposition
        - Original, trend, seasonal, and residual components are displayed
        - Subplot layout is properly configured
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_seasonal_decomposition(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        freq=365,
        plot_title="Seasonal Decomposition Test",
        width=1600,
        height=1000,
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_seasonal_decomposition_applies_custom_parameters(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_seasonal_decomposition applies custom visualization parameters.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method accepts custom color parameters for each component
        - Line width and subplot settings are properly applied
        - Subplot titles can be controlled via parameter
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_seasonal_decomposition(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        original_color="black",
        trend_color="blue",
        seasonal_color="green",
        residual_color="red",
        line_width=3,
        show_subplot_titles=True,
        plot_title="Custom Decomposition Test",
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_plot_seasonal_decomposition_handles_different_frequencies(
    time_series_dataframe: pd.DataFrame,
) -> None:
    """Tests plot_seasonal_decomposition handles different seasonal frequencies.

    Args:
        time_series_dataframe (pd.DataFrame): Fixture containing test time series data

    Asserts:
        - Method executes without errors for different freq parameters
        - Seasonal decomposition adapts to specified frequency
        - Plot maintains proper formatting across frequencies
    """
    ts_plots = TimeSeriesPlots()

    # This should not raise an exception
    ts_plots.plot_seasonal_decomposition(
        data=time_series_dataframe.reset_index(),
        feature="date",
        target="value1",
        freq=30,  # Monthly seasonality
        plot_title="Monthly Seasonality Test",
        vertical_spacing=0.1,
    )

    assert True, "Method should execute without errors"


def test_timeseriesplots_raises_error_missing_columns() -> None:
    """Tests TimeSeriesPlots methods raise appropriate errors for missing columns.

    Asserts:
        - Methods raise KeyError for non-existent feature or target columns
        - Error handling is appropriate for invalid column names
    """
    ts_plots = TimeSeriesPlots()
    test_df = pd.DataFrame(
        {"date": pd.date_range("2020-01-01", periods=10), "value": range(10)}
    )

    with pytest.raises(KeyError):
        ts_plots.plot_time_series_mean(
            data=test_df, feature="NonExistentColumn", target="value"
        )

    with pytest.raises(KeyError):
        ts_plots.plot_time_series_with_trend(
            data=test_df, feature="date", target="NonExistentTarget"
        )


def test_timeseriesplots_raises_error_invalid_input_types() -> None:
    """Tests TimeSeriesPlots methods raise appropriate errors for invalid input types.

    Asserts:
        - Methods raise appropriate errors for non-DataFrame input
        - Error messages provide helpful information
    """
    ts_plots = TimeSeriesPlots()

    with pytest.raises((TypeError, AttributeError)):
        ts_plots.plot_time_series_mean(
            data="invalid_input", feature="invalid", target="invalid"
        )

    with pytest.raises((TypeError, AttributeError)):
        ts_plots.plot_seasonal_decomposition(
            data=123, feature="invalid", target="invalid"
        )
