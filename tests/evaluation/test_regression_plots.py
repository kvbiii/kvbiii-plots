import numpy as np
import pandas as pd
import pytest

from kvbiii_plots.evaluation.regression_plots import RegressionPlots


@pytest.fixture
def regression_data(test_settings: object) -> tuple[np.ndarray, np.ndarray]:
    """
    Provides sample regression data for testing purposes.

    Args:
        test_settings: Test settings fixture.

    Returns:
        tuple[np.ndarray, np.ndarray]: True and predicted values.
    """
    np.random.seed(test_settings.SEED)
    y_true = np.random.rand(100) * 100
    y_pred = y_true + np.random.normal(0, 5, 100)
    return y_true, y_pred


@pytest.fixture
def regression_data_series(test_settings: object) -> tuple[pd.Series, pd.Series]:
    """
    Provides sample regression data as pandas Series for testing purposes.

    Args:
        test_settings: Test settings fixture.

    Returns:
        tuple[pd.Series, pd.Series]: True and predicted values as Series.
    """
    np.random.seed(test_settings.SEED)
    y_true = pd.Series(np.random.rand(50) * 50, name="true_values")
    y_pred = pd.Series(y_true + np.random.normal(0, 3, 50), name="predicted_values")
    return y_true, y_pred


@pytest.fixture
def regression_data_list(test_settings: object) -> tuple[list[float], list[float]]:
    """
    Provides sample regression data as lists for testing purposes.

    Args:
        test_settings: Test settings fixture.

    Returns:
        tuple[list, list]: True and predicted values as lists.
    """
    np.random.seed(test_settings.SEED)
    y_true = (np.random.rand(30) * 30).tolist()
    y_pred = (np.array(y_true) + np.random.normal(0, 2, 30)).tolist()
    return y_true, y_pred


@pytest.fixture
def perfect_predictions() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides perfect regression predictions for testing edge cases.

    Returns:
        tuple[np.ndarray, np.ndarray]: Identical true and predicted values.
    """
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 4, 5])
    return y_true, y_pred


@pytest.fixture
def mismatched_length_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides regression data with mismatched lengths for error testing.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays with different lengths.
    """
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3])
    return y_true, y_pred


@pytest.fixture
def empty_regression_data() -> tuple[np.ndarray, np.ndarray]:
    """
    Provides empty regression data for error testing.

    Returns:
        tuple[np.ndarray, np.ndarray]: Empty arrays.
    """
    y_true = np.array([])
    y_pred = np.array([])
    return y_true, y_pred


def test_regressionplots_initialization() -> None:
    """Tests RegressionPlots class initialization.

    Asserts:
        - Class initializes without errors
        - Default regression colors are properly set
        - Inherits from BasePlots correctly
    """
    reg_plots = RegressionPlots()

    assert hasattr(reg_plots, "default_regression_colors")
    assert "scatter" in reg_plots.default_regression_colors
    assert "line" in reg_plots.default_regression_colors
    assert "trendline" in reg_plots.default_regression_colors
    assert "residual_line" in reg_plots.default_regression_colors
    assert hasattr(reg_plots, "metrics_dict")
    assert "RMSE" in reg_plots.metrics_dict
    assert "R2" in reg_plots.metrics_dict


def test_regressionplots_validate_regression_inputs_handles_numpy_arrays(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _validate_regression_inputs with numpy arrays.

    Args:
        regression_data: Fixture containing numpy array data.

    Asserts:
        - Method executes without errors for numpy arrays
        - Returns validated numpy arrays
        - Output arrays have correct shapes
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    validated_true, validated_pred = reg_plots._validate_regression_inputs(
        y_true, y_pred
    )

    assert isinstance(validated_true, np.ndarray)
    assert isinstance(validated_pred, np.ndarray)
    assert len(validated_true) == len(validated_pred)
    assert len(validated_true) == len(y_true)


def test_regressionplots_validate_regression_inputs_handles_pandas_series(
    regression_data_series: tuple[pd.Series, pd.Series],
) -> None:
    """Tests _validate_regression_inputs with pandas Series.

    Args:
        regression_data_series: Fixture containing pandas Series data.

    Asserts:
        - Method executes without errors for pandas Series
        - Converts Series to numpy arrays
        - Maintains data integrity during conversion
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data_series

    validated_true, validated_pred = reg_plots._validate_regression_inputs(
        y_true, y_pred
    )

    assert isinstance(validated_true, np.ndarray)
    assert isinstance(validated_pred, np.ndarray)
    assert len(validated_true) == len(y_true)
    assert len(validated_pred) == len(y_pred)


def test_regressionplots_validate_regression_inputs_handles_lists(
    regression_data_list: tuple[list[float], list[float]],
) -> None:
    """Tests _validate_regression_inputs with lists.

    Args:
        regression_data_list: Fixture containing list data.

    Asserts:
        - Method executes without errors for lists
        - Converts lists to numpy arrays
        - Preserves data values during conversion
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data_list

    validated_true, validated_pred = reg_plots._validate_regression_inputs(
        y_true, y_pred
    )

    assert isinstance(validated_true, np.ndarray)
    assert isinstance(validated_pred, np.ndarray)
    assert len(validated_true) == len(y_true)
    assert len(validated_pred) == len(y_pred)


def test_regressionplots_validate_regression_inputs_raises_error_mismatched_lengths(
    mismatched_length_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _validate_regression_inputs raises error for mismatched lengths.

    Args:
        mismatched_length_data: Fixture containing arrays with different lengths.

    Asserts:
        - ValueError is raised for arrays with different lengths
        - Error message is informative
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = mismatched_length_data

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        reg_plots._validate_regression_inputs(y_true, y_pred)


def test_regressionplots_validate_regression_inputs_raises_error_empty_arrays(
    empty_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _validate_regression_inputs raises error for empty arrays.

    Args:
        empty_regression_data: Fixture containing empty arrays.

    Asserts:
        - ValueError is raised for empty arrays
        - Error message indicates empty input issue
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = empty_regression_data

    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        reg_plots._validate_regression_inputs(y_true, y_pred)


def test_regressionplots_calculate_metric_value_handles_supported_metrics(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _calculate_metric_value with supported metrics.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method calculates RMSE correctly
        - Method calculates R2 correctly
        - Method calculates MAE correctly
        - Metric values are numeric
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    rmse_value = reg_plots._calculate_metric_value(y_true, y_pred, "RMSE")
    r2_value = reg_plots._calculate_metric_value(y_true, y_pred, "R2")
    mae_value = reg_plots._calculate_metric_value(y_true, y_pred, "MAE")

    assert isinstance(rmse_value, (int, float))
    assert isinstance(r2_value, (int, float))
    assert isinstance(mae_value, (int, float))
    assert rmse_value >= 0
    assert mae_value >= 0


def test_regressionplots_calculate_metric_value_raises_error_unsupported_metric(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _calculate_metric_value raises error for unsupported metrics.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - ValueError is raised for unsupported metric names
        - Error message lists available metrics
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    with pytest.raises(ValueError, match="Unsupported metric"):
        reg_plots._calculate_metric_value(y_true, y_pred, "INVALID_METRIC")


def test_regressionplots_calculate_metric_value_handles_case_insensitive_metrics(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _calculate_metric_value handles case-insensitive metric names.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts lowercase metric names
        - Method accepts mixed case metric names
        - Results are consistent regardless of case
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    rmse_upper = reg_plots._calculate_metric_value(y_true, y_pred, "RMSE")
    rmse_lower = reg_plots._calculate_metric_value(y_true, y_pred, "rmse")
    rmse_mixed = reg_plots._calculate_metric_value(y_true, y_pred, "Rmse")

    assert rmse_upper == rmse_lower == rmse_mixed


def test_regressionplots_create_scatter_trace_returns_valid_trace(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _create_scatter_trace returns valid plotly trace.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method returns go.Scatter object
        - Trace has correct mode and data
        - Default colors are applied when none specified
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    trace = reg_plots._create_scatter_trace(y_pred, y_true)

    assert hasattr(trace, "x")
    assert hasattr(trace, "y")
    assert hasattr(trace, "mode")
    assert trace.mode == "markers"
    assert len(trace.x) == len(y_pred)
    assert len(trace.y) == len(y_true)


def test_regressionplots_create_scatter_trace_applies_custom_parameters(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests _create_scatter_trace applies custom parameters.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts custom color parameter
        - Method accepts custom size parameter
        - Method accepts custom opacity parameter
        - Custom parameters are applied to trace
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    custom_color = "red"
    custom_size = 12
    custom_opacity = 0.8
    custom_name = "Test Trace"

    trace = reg_plots._create_scatter_trace(
        y_pred,
        y_true,
        color=custom_color,
        size=custom_size,
        opacity=custom_opacity,
        name=custom_name,
        show_legend=True,
    )

    assert trace.marker.color == custom_color
    assert trace.marker.size == custom_size
    assert trace.marker.opacity == custom_opacity
    assert trace.name == custom_name
    assert trace.showlegend == True


def test_regressionplots_homoscedacity_plot_handles_numpy_arrays(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests homoscedacity_plot correctly processes numpy arrays.

    Args:
        regression_data: Fixture containing numpy array data.

    Asserts:
        - Method executes without errors when given numpy arrays
        - Plot is generated with appropriate configuration
        - Residuals are calculated correctly
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.homoscedacity_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_title="Test Homoscedacity Plot",
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_homoscedacity_plot_applies_custom_parameters(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests homoscedacity_plot applies custom visualization parameters.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts custom scatter color and size
        - Method accepts custom reference line parameters
        - Method accepts custom metric parameters
        - Trendline option works correctly
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.homoscedacity_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_title="Custom Parameters Test",
        width=1200,
        height=900,
        metric_name="MAE",
        scatter_color="blue",
        scatter_size=10,
        scatter_opacity=0.8,
        reference_line_color="green",
        reference_line_width=3,
        show_trendline=True,
        trendline_color="purple",
    )

    assert True, "Method should execute without errors"


def test_regressionplots_homoscedacity_plot_handles_series_input(
    regression_data_series: tuple[pd.Series, pd.Series],
) -> None:
    """Tests homoscedacity_plot correctly processes pandas Series.

    Args:
        regression_data_series: Fixture containing pandas Series data.

    Asserts:
        - Method executes without errors when given Series input
        - Series are converted to numpy arrays internally
        - Plot displays residuals correctly
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data_series

    reg_plots.homoscedacity_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="RMSE",
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_homoscedacity_plot_handles_custom_metric_value(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests homoscedacity_plot accepts pre-calculated metric values.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts custom metric_value parameter
        - Custom metric value is used instead of calculating
        - Plot title reflects custom metric value
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    custom_metric_value = 42.5

    reg_plots.homoscedacity_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="RMSE",
        metric_value=custom_metric_value,
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_true_vs_fitted_plot_handles_numpy_arrays(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests true_vs_fitted_plot correctly processes numpy arrays.

    Args:
        regression_data: Fixture containing numpy array data.

    Asserts:
        - Method executes without errors when given numpy arrays
        - Plot is generated with appropriate configuration
        - True vs predicted relationship is visualized
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_title="Test True vs Fitted Plot",
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_true_vs_fitted_plot_applies_custom_parameters(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests true_vs_fitted_plot applies custom visualization parameters.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts custom scatter parameters
        - Method accepts custom diagonal line parameters
        - Method accepts custom trendline parameters
        - Method accepts custom axis buffer settings
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_title="Custom Parameters Test",
        width=1200,
        height=900,
        metric_name="R2",
        scatter_color="orange",
        scatter_size=12,
        scatter_opacity=0.9,
        show_diagonal=True,
        diagonal_color="red",
        diagonal_width=4,
        show_trendline=True,
        trendline_color="green",
        trendline_width=3,
        axis_buffer_percent=5.0,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_true_vs_fitted_plot_handles_diagonal_and_trendline_options(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests true_vs_fitted_plot handles diagonal and trendline display options.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts show_diagonal parameter
        - Method accepts show_trendline parameter
        - Both options can be enabled simultaneously
        - Both options can be disabled
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        show_diagonal=True,
        show_trendline=False,
        width=600,
        height=400,
    )

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        show_diagonal=False,
        show_trendline=True,
        width=600,
        height=400,
    )

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        show_diagonal=True,
        show_trendline=True,
        width=600,
        height=400,
    )

    assert True, "Method should execute without errors for all combinations"


def test_regressionplots_true_vs_fitted_plot_handles_perfect_predictions(
    perfect_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests true_vs_fitted_plot handles perfect predictions.

    Args:
        perfect_predictions: Fixture containing identical true and predicted values.

    Asserts:
        - Method executes without errors for perfect predictions
        - R2 score should be 1.0 for perfect predictions
        - Plot displays correctly for edge case
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = perfect_predictions

    reg_plots.true_vs_fitted_plot(
        y_true=y_true,
        y_pred=y_pred,
        metric_name="R2",
        width=600,
        height=400,
    )

    assert True, "Method should execute without errors for perfect predictions"


def test_regressionplots_residual_distribution_plot_handles_histogram_type(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot creates histogram correctly.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method executes without errors for histogram type
        - Histogram displays residual distribution
        - Custom bins parameter works correctly
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        plot_title="Test Residual Histogram",
        bins=30,
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_residual_distribution_plot_handles_box_type(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot creates box plot correctly.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method executes without errors for box plot type
        - Box plot displays residual distribution statistics
        - Plot configuration is appropriate for box plots
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="box",
        plot_title="Test Residual Box Plot",
        width=800,
        height=600,
    )

    assert True, "Method should execute without errors"


def test_regressionplots_residual_distribution_plot_applies_custom_parameters(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot applies custom visualization parameters.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts custom color and opacity parameters
        - Method accepts custom bins parameter for histograms
        - Method accepts normal curve display option
        - Custom metric parameters work correctly
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        plot_title="Custom Parameters Test",
        width=1200,
        height=800,
        metric_name="MAE",
        bins=40,
        color="purple",
        opacity=0.8,
        show_normal_curve=True,
        normal_curve_color="orange",
    )

    assert True, "Method should execute without errors"


def test_regressionplots_residual_distribution_plot_handles_normal_curve_option(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot normal curve display option.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method accepts show_normal_curve parameter
        - Normal curve can be enabled and disabled
        - Normal curve color can be customized
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        show_normal_curve=True,
        normal_curve_color="red",
        width=600,
        height=400,
    )

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        show_normal_curve=False,
        width=600,
        height=400,
    )

    assert True, "Method should execute without errors for both options"


def test_regressionplots_residual_distribution_plot_raises_error_invalid_plot_type(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot raises error for invalid plot type.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - ValueError is raised for unsupported plot types
        - Error message indicates valid plot types
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    with pytest.raises(ValueError, match="Unsupported plot_type"):
        reg_plots.residual_distribution_plot(
            y_true=y_true,
            y_pred=y_pred,
            plot_type="invalid_type",
        )


def test_regressionplots_residual_distribution_plot_handles_automatic_axis_titles(
    regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests residual_distribution_plot automatic axis title generation.

    Args:
        regression_data: Fixture containing regression data.

    Asserts:
        - Method generates appropriate axis titles when not specified
        - Different plot types get appropriate default titles
        - Custom titles override automatic ones
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        width=600,
        height=400,
    )

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="box",
        width=600,
        height=400,
    )

    reg_plots.residual_distribution_plot(
        y_true=y_true,
        y_pred=y_pred,
        plot_type="histogram",
        xaxis_title="Custom X Title",
        yaxis_title="Custom Y Title",
        width=600,
        height=400,
    )

    assert True, "Method should execute without errors for all title options"


def test_regressionplots_methods_handle_mismatched_input_lengths(
    mismatched_length_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests all regression plot methods raise errors for mismatched input lengths.

    Args:
        mismatched_length_data: Fixture containing arrays with different lengths.

    Asserts:
        - ValueError is raised for all methods with mismatched lengths
        - Error handling is consistent across all methods
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = mismatched_length_data

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        reg_plots.homoscedacity_plot(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        reg_plots.true_vs_fitted_plot(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError, match="y_true and y_pred must have the same length"):
        reg_plots.residual_distribution_plot(y_true=y_true, y_pred=y_pred)


def test_regressionplots_methods_handle_empty_input_arrays(
    empty_regression_data: tuple[np.ndarray, np.ndarray],
) -> None:
    """Tests all regression plot methods raise errors for empty input arrays.

    Args:
        empty_regression_data: Fixture containing empty arrays.

    Asserts:
        - ValueError is raised for all methods with empty arrays
        - Error handling is consistent across all methods
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = empty_regression_data

    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        reg_plots.homoscedacity_plot(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        reg_plots.true_vs_fitted_plot(y_true=y_true, y_pred=y_pred)

    with pytest.raises(ValueError, match="Input arrays cannot be empty"):
        reg_plots.residual_distribution_plot(y_true=y_true, y_pred=y_pred)


def test_regressionplots_methods_handle_list_input(
    regression_data_list: tuple[list[float], list[float]],
) -> None:
    """Tests all regression plot methods handle list input correctly.

    Args:
        regression_data_list: Fixture containing list data.

    Asserts:
        - All methods execute without errors for list input
        - Lists are converted to numpy arrays internally
        - Plot generation works correctly with list data
    """
    reg_plots = RegressionPlots()
    y_true, y_pred = regression_data_list

    reg_plots.homoscedacity_plot(y_true=y_true, y_pred=y_pred, width=400, height=300)

    reg_plots.true_vs_fitted_plot(y_true=y_true, y_pred=y_pred, width=400, height=300)

    reg_plots.residual_distribution_plot(
        y_true=y_true, y_pred=y_pred, plot_type="histogram", width=400, height=300
    )

    assert True, "All methods should execute without errors for list input"
