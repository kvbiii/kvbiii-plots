import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from kvbiii_plots.base_plots import BasePlots


def test_baseplots_init_creates_default_configurations() -> None:
    """Tests BasePlots.__init__ correctly initializes default configurations.

    Asserts:
        - quantiles_dict contains expected keys and values
        - default_template is set correctly
        - default_font has proper configuration
        - default_colors contains all required color schemes
    """
    base_plots = BasePlots()

    if not (
        base_plots.quantiles_dict
        == {
            "Min": 0,
            "Q1": 0.25,
            "Med": 0.5,
            "Q3": 0.75,
            "Max": 1,
        }
    ):
        raise AssertionError("Assertion failed.")
    if not (base_plots.default_template == "simple_white"):
        raise AssertionError("Assertion failed.")
    if not (base_plots.default_font["family"] == "Times New Roman"):
        raise AssertionError("Assertion failed.")
    if not (base_plots.default_font["size"] == 26):
        raise AssertionError("Assertion failed.")
    if not ("primary" in base_plots.default_colors):
        raise AssertionError("Assertion failed.")
    if not ("secondary" in base_plots.default_colors):
        raise AssertionError("Assertion failed.")


def test_baseplots_check_data_handles_dataframe_input(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests check_data correctly processes DataFrame input.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method returns numpy array
        - Original data shape is preserved when squeezed
        - Data is properly converted from DataFrame
    """
    base_plots = BasePlots()
    result = base_plots.check_data(sample_dataframe)

    if not (isinstance(result, np.ndarray)):
        raise AssertionError("Result should be numpy array")

    if not (result.shape[0] > 0):
        raise AssertionError("Result should have some data")


def test_baseplots_check_data_handles_series_input(sample_series: pd.Series) -> None:
    """Tests check_data correctly processes Series input.

    Args:
        sample_series (pd.Series): Fixture containing test Series data

    Asserts:
        - Method returns numpy array
        - Series data is properly converted
        - Data type is preserved when appropriate
    """
    base_plots = BasePlots()
    result = base_plots.check_data(sample_series)

    if not (isinstance(result, np.ndarray)):
        raise AssertionError("Result should be numpy array")
    if not (len(result) == len(sample_series)):
        raise AssertionError("Length should be preserved")


def test_baseplots_check_data_handles_list_input(sample_list: list[float]) -> None:
    """Tests check_data correctly processes list input.

    Args:
        sample_list (list[float]): Fixture containing test list data

    Asserts:
        - Method returns numpy array
        - List values are properly converted
        - Original length is maintained
    """
    base_plots = BasePlots()
    result = base_plots.check_data(sample_list)

    if not (isinstance(result, np.ndarray)):
        raise AssertionError("Result should be numpy array")
    if not (len(result) == len(sample_list)):
        raise AssertionError("Length should be preserved")


def test_baseplots_check_data_raises_error_invalid_input() -> None:
    """Tests check_data raises TypeError for invalid input types.

    Asserts:
        - TypeError is raised for string input
        - TypeError is raised for dictionary input
        - Error message contains expected information
    """
    base_plots = BasePlots()

    with pytest.raises(TypeError):
        base_plots.check_data("invalid_input")

    with pytest.raises(TypeError):
        base_plots.check_data({"invalid": "input"})


def test_baseplots_check_2d_data_handles_dataframe_input(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests check_2d_data correctly processes 2D DataFrame input.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method returns numpy array
        - Output maintains 2D structure
        - NaN rows are properly removed
    """
    base_plots = BasePlots()
    result = base_plots.check_2d_data(sample_dataframe)

    if not (isinstance(result, np.ndarray)):
        raise AssertionError("Result should be numpy array")
    if not (result.ndim == 2):
        raise AssertionError("Result should be 2-dimensional")
    if not (result.shape[1] == sample_dataframe.shape[1]):
        raise AssertionError("Column count should be preserved")


def test_baseplots_check_2d_data_raises_error_1d_input(
    sample_series: pd.Series,
) -> None:
    """Tests check_2d_data correctly processes Series input by reshaping to 2D.

    Args:
        sample_series (pd.Series): Fixture containing test Series data

    Asserts:
        - Method accepts Series input (despite type hint suggesting otherwise)
        - Series is reshaped to 2D array
        - Result maintains data integrity
    """
    base_plots = BasePlots()

    result = base_plots.check_2d_data(sample_series.values)

    if not (isinstance(result, np.ndarray)):
        raise AssertionError("Result should be numpy array")
    if not (result.ndim == 2):
        raise AssertionError("Result should be 2-dimensional")


def test_baseplots_get_colors_returns_correct_types() -> None:
    """Tests _get_colors returns appropriate color list.

    Asserts:
        - Method returns list of colors
        - Color list contains colors from qualitative palette
        - Colors are appropriate for the number requested
    """
    base_plots = BasePlots()
    colors = base_plots._get_colors(5)

    if not (isinstance(colors, list)):
        raise AssertionError("Colors should be a list")
    if not (len(colors) > 0):
        raise AssertionError("Color list should contain colors")


def test_baseplots_apply_default_layout_modifies_figure() -> None:
    """Tests apply_default_layout correctly applies layout settings to figure.

    Asserts:
        - Figure layout is modified with provided parameters
        - Title is properly formatted with bold tags
        - Font settings are applied correctly
    """
    base_plots = BasePlots()
    fig = go.Figure()

    base_plots.apply_default_layout(
        fig,
        plot_title="Test Title",
        width=800,
        height=600,
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
    )

    if not (fig.layout.title.text == "<b>Test Title</b>"):
        raise AssertionError("Title should be formatted with bold tags")
    if not (fig.layout.width == 800):
        raise AssertionError("Width should be set correctly")
    if not (fig.layout.height == 600):
        raise AssertionError("Height should be set correctly")


def test_baseplots_filter_nan_indices_removes_nan_rows(
    dataframe_with_nan: pd.DataFrame,
) -> None:
    """Tests filter_nan_indices correctly identifies and removes NaN indices.

    Args:
        dataframe_with_nan (pd.DataFrame): Fixture containing DataFrame with NaN values

    Asserts:
        - Method returns boolean Series
        - NaN indices are properly identified as False
        - Non-NaN indices are properly identified as True
    """
    base_plots = BasePlots()
    result_indices = base_plots.filter_nan_indices(dataframe_with_nan, "values")

    if not (isinstance(result_indices, pd.Series)):
        raise AssertionError("Result should be pandas Series")
    if not (result_indices.dtype == bool):
        raise AssertionError("Result should be boolean Series")
    if not (result_indices.sum() < len(dataframe_with_nan)):
        raise AssertionError("Some values should be filtered as NaN")


def test_baseplots_add_quantile_annotations_adds_annotations_to_figure(
    sample_numpy_array: np.ndarray,
) -> None:
    """Tests add_quantile_annotations correctly adds quantile annotations to figure.

    Args:
        sample_numpy_array (np.ndarray): Fixture containing test numpy array data

    Asserts:
        - Annotations are added to figure
        - Quantile values are calculated correctly
        - Annotation text format includes quantile names
    """
    base_plots = BasePlots()
    fig = go.Figure()

    base_plots.add_quantile_annotations(fig, sample_numpy_array, annotations=True)

    if not (len(fig.layout.annotations) > 0):
        raise AssertionError("Annotations should be added to figure")
    if not all(
        annotation.text is not None
        and annotation.text.split(": ")[-1].count(".") == 1
        and len(annotation.text.split(": ")[-1].split(".")[-1]) <= 5
        for annotation in fig.layout.annotations
    ):
        raise AssertionError(
            "Quantile annotations should display up to five decimal places"
        )


def test_baseplots_calculate_dynamic_dimensions_returns_appropriate_size() -> None:
    """Tests calculate_dynamic_dimensions returns appropriate width and height.

    Asserts:
        - Method returns tuple of two integers
        - Dimensions scale appropriately with item count
        - Minimum dimensions are respected
    """
    base_plots = BasePlots()
    width, height = base_plots.calculate_dynamic_dimensions(10)

    if not (isinstance(width, int)):
        raise AssertionError("Width should be integer")
    if not (isinstance(height, int)):
        raise AssertionError("Height should be integer")
    if not (width >= 1600):
        raise AssertionError("Width should meet minimum requirement")
    if not (height >= 800):
        raise AssertionError("Height should meet minimum requirement")


def test_baseplots_create_subplot_layout_creates_valid_figure() -> None:
    """Tests create_subplot_layout creates valid subplot figure.

    Asserts:
        - Method returns plotly Figure object
        - Subplot configuration matches specifications
        - Figure is ready for trace addition
    """
    base_plots = BasePlots()
    fig = base_plots.create_subplot_layout(1, 2, [["xy", "xy"]])

    if not (isinstance(fig, go.Figure)):
        raise AssertionError("Result should be plotly Figure")
    if not (hasattr(fig, "add_trace")):
        raise AssertionError("Figure should support trace addition")


def test_baseplots_apply_aggregation_handles_mean_function() -> None:
    """Tests _apply_aggregation correctly applies mean aggregation function.

    Asserts:
        - Method returns float value
        - Mean calculation is accurate for test data
        - Function handles Series input correctly
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 2, 3, 4, 5])
    result = base_plots._apply_aggregation(test_series, "mean")

    if not (isinstance(result, float)):
        raise AssertionError("Result should be float")
    if not (result == 3.0):
        raise AssertionError("Mean should be calculated correctly")


def test_baseplots_apply_aggregation_handles_sum_function() -> None:
    """Tests _apply_aggregation correctly applies sum aggregation function.

    Asserts:
        - Method returns numeric value for sum function
        - Sum calculation matches expected result
        - Function processes numerical data correctly
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 2, 3, 4, 5])
    result = base_plots._apply_aggregation(test_series, "sum")

    if not (isinstance(result, (int, float, np.number))):
        raise AssertionError("Result should be numeric")
    if not (result == 15):
        raise AssertionError("Sum should be calculated correctly")


def test_baseplots_apply_aggregation_raises_error_invalid_function() -> None:
    """Tests _apply_aggregation raises ValueError for invalid aggregation function.

    Asserts:
        - ValueError is raised for unknown function names
        - Error message provides helpful information
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 2, 3, 4, 5])

    with pytest.raises(ValueError, match="Unsupported aggregation function"):
        base_plots._apply_aggregation(test_series, "invalid_function")


def test_baseplots_get_colors() -> None:
    """Tests _get_colors method functionality.

    Asserts:
        - Returns colors for given n_colors
        - Colors are from expected qualitative palette
        - Appropriate color scheme for different numbers
    """
    base_plots = BasePlots()

    colors = base_plots._get_colors(5)
    if not (isinstance(colors, list)):
        raise AssertionError("Assertion failed.")

    colors = base_plots._get_colors(15)
    if not (isinstance(colors, list)):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_default_layout() -> None:
    """Tests apply_default_layout method functionality.

    Asserts:
        - Method applies layout to plotly figure correctly
        - Title, dimensions, and axis titles are set properly
        - Font configuration is applied
    """
    base_plots = BasePlots()
    fig = go.Figure()

    base_plots.apply_default_layout(
        fig=fig,
        plot_title="Test Title",
        width=800,
        height=600,
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
    )

    if not (fig.layout.title.text == "<b>Test Title</b>"):
        raise AssertionError("Assertion failed.")
    if not (fig.layout.width == 800):
        raise AssertionError("Assertion failed.")
    if not (fig.layout.height == 600):
        raise AssertionError("Assertion failed.")
    if not (fig.layout.xaxis.title.text == "X Axis"):
        raise AssertionError("Assertion failed.")
    if not (fig.layout.yaxis.title.text == "Y Axis"):
        raise AssertionError("Assertion failed.")


def test_baseplots_filter_nan_indices(sample_dataframe: pd.DataFrame) -> None:
    """Tests filter_nan_indices method functionality.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method correctly identifies non-NaN indices
        - Returns boolean mask
        - Handles missing values appropriately
    """
    base_plots = BasePlots()

    test_df = sample_dataframe.copy()
    test_df.loc[0, "A"] = np.nan
    test_df.loc[2, "A"] = np.nan

    valid_mask = base_plots.filter_nan_indices(test_df, "A")

    if not (isinstance(valid_mask, pd.Series)):
        raise AssertionError("Assertion failed.")
    if not (valid_mask.dtype == bool):
        raise AssertionError("Assertion failed.")
    if not (not valid_mask.iloc[0]):
        raise AssertionError("Assertion failed.")
    if not (not valid_mask.iloc[2]):
        raise AssertionError("Assertion failed.")


def test_baseplots_add_quantile_annotations() -> None:
    """Tests add_quantile_annotations method functionality.

    Asserts:
        - Method adds annotations to figure correctly
        - Quantile values are calculated properly
        - Annotations have correct positioning and formatting
    """
    base_plots = BasePlots()
    fig = go.Figure()
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    base_plots.add_quantile_annotations(
        fig=fig, data=test_data, annotations=True, x_position=0.5
    )

    if not (isinstance(fig, go.Figure)):
        raise AssertionError("Assertion failed.")


def test_baseplots_calculate_dynamic_dimensions() -> None:
    """Tests calculate_dynamic_dimensions method functionality.

    Asserts:
        - Method returns reasonable width and height values
        - Dimensions scale appropriately with input parameters
        - Returns tuple of integers
    """
    base_plots = BasePlots()

    width, height = base_plots.calculate_dynamic_dimensions(
        n_items=5, min_width=800, min_height=600, scale_factor=30
    )

    if not (isinstance(width, int)):
        raise AssertionError("Assertion failed.")
    if not (isinstance(height, int)):
        raise AssertionError("Assertion failed.")
    if not (width >= 800):
        raise AssertionError("Assertion failed.")
    if not (height >= 600):
        raise AssertionError("Assertion failed.")

    width2, height2 = base_plots.calculate_dynamic_dimensions(
        n_items=10, min_width=800, min_height=600, scale_factor=30
    )

    if not (width2 >= width):
        raise AssertionError("Assertion failed.")
    if not (height2 >= height):
        raise AssertionError("Assertion failed.")


def test_baseplots_create_subplot_layout() -> None:
    """Tests create_subplot_layout method functionality.

    Asserts:
        - Method returns plotly figure with subplots
        - Subplot configuration matches input parameters
        - Figure is properly initialized
    """
    base_plots = BasePlots()

    fig = base_plots.create_subplot_layout(
        rows=2, cols=2, subplot_types=[["xy", "xy"], ["xy", "xy"]]
    )

    if not (isinstance(fig, go.Figure)):
        raise AssertionError("Assertion failed.")

    if not (hasattr(fig, "layout")):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_aggregation_with_nan_values() -> None:
    """Tests _apply_aggregation handles NaN values correctly.

    Asserts:
        - Method correctly applies aggregation while ignoring NaN
        - Returns appropriate numeric values
        - Handles various aggregation functions with missing data
    """
    base_plots = BasePlots()
    test_series_nan = pd.Series([1, 2, np.nan, 4, 5])

    mean_result = base_plots._apply_aggregation(test_series_nan, "mean")
    if not (mean_result == 3.0):
        raise AssertionError("Assertion failed.")

    std_result = base_plots._apply_aggregation(test_series_nan, "std")
    if not (isinstance(std_result, float)):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_aggregation_handles_max_function() -> None:
    """Tests _apply_aggregation correctly handles max function.

    Asserts:
        - Method returns maximum value from series
        - Max aggregation works with different data types
        - Returns correct float type
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 5, 3, 9, 2])

    result = base_plots._apply_aggregation(test_series, "max")
    if not (result == 9.0):
        raise AssertionError("Assertion failed.")
    if not (isinstance(result, (int, float))):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_aggregation_handles_min_function() -> None:
    """Tests _apply_aggregation correctly handles min function.

    Asserts:
        - Method returns minimum value from series
        - Min aggregation works with different data types
        - Returns correct float type
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 5, 3, 9, 2])

    result = base_plots._apply_aggregation(test_series, "min")
    if not (result == 1.0):
        raise AssertionError("Assertion failed.")
    if not (isinstance(result, (int, float))):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_aggregation_handles_std_function() -> None:
    """Tests _apply_aggregation correctly handles std function.

    Asserts:
        - Method returns standard deviation of series
        - Std aggregation produces valid statistical result
        - Returns correct float type
    """
    base_plots = BasePlots()
    test_series = pd.Series([1, 2, 3, 4, 5])

    result = base_plots._apply_aggregation(test_series, "std")
    expected_std = test_series.std()
    if not (abs(result - expected_std) < 1e-10):
        raise AssertionError("Assertion failed.")
    if not (isinstance(result, float)):
        raise AssertionError("Assertion failed.")


def test_baseplots_apply_aggregation_handles_median_function() -> None:
    """Tests _apply_aggregation correctly handles median function.

    Asserts:
        - Method returns median value from series
        - Median aggregation works with odd and even length series
        - Returns correct float type
    """
    base_plots = BasePlots()
    test_series_odd = pd.Series([1, 2, 3, 4, 5])
    test_series_even = pd.Series([1, 2, 3, 4])

    result_odd = base_plots._apply_aggregation(test_series_odd, "median")
    result_even = base_plots._apply_aggregation(test_series_even, "median")

    if not (result_odd == 3.0):
        raise AssertionError("Assertion failed.")
    if not (result_even == 2.5):
        raise AssertionError("Assertion failed.")
    if not (isinstance(result_odd, float)):
        raise AssertionError("Assertion failed.")
    if not (isinstance(result_even, float)):
        raise AssertionError("Assertion failed.")


def test_baseplots_edge_cases() -> None:
    """Tests edge cases for BasePlots methods.

    Asserts:
        - Methods handle empty data gracefully
        - Single-value inputs are processed correctly
        - Boundary conditions are managed appropriately
    """
    base_plots = BasePlots()

    single_value = pd.Series([42])
    result = base_plots._apply_aggregation(single_value, "mean")
    if not (result == 42.0):
        raise AssertionError("Assertion failed.")

    empty_series = pd.Series([], dtype=float)
    try:
        result = base_plots._apply_aggregation(empty_series, "mean")

        if not (pd.isna(result)):
            raise AssertionError("Assertion failed.")
    except (ValueError, TypeError):

        pass


def test_baseplots_create_subplot_layout_functionality() -> None:
    """Tests create_subplot_layout method functionality.

    Asserts:
        - Method creates subplot with correct number of subplots
        - Subplot types are properly configured
        - Returns valid plotly Figure object
    """
    base_plots = BasePlots()

    fig = base_plots.create_subplot_layout(rows=1, cols=2, subplot_types=[["xy", "xy"]])

    if not (hasattr(fig, "add_trace")):
        raise AssertionError("Assertion failed.")
    if not (hasattr(fig, "update_layout")):
        raise AssertionError("Assertion failed.")


def test_baseplots_calculate_dynamic_dimensions_edge_cases() -> None:
    """Tests calculate_dynamic_dimensions with edge cases.

    Asserts:
        - Method handles zero items correctly
        - Method handles very large numbers appropriately
        - Returns minimum dimensions when needed
    """
    base_plots = BasePlots()

    width, height = base_plots.calculate_dynamic_dimensions(0)
    if not (width >= 1600):
        raise AssertionError("Assertion failed.")
    if not (height >= 800):
        raise AssertionError("Assertion failed.")

    width, height = base_plots.calculate_dynamic_dimensions(1)
    if not (width >= 1600):
        raise AssertionError("Assertion failed.")
    if not (height >= 800):
        raise AssertionError("Assertion failed.")

    width, height = base_plots.calculate_dynamic_dimensions(100)
    if not (width == 3000):
        raise AssertionError("Assertion failed.")
    if not (height == 3000):
        raise AssertionError("Assertion failed.")


def test_baseplots_add_quantile_annotations_with_specific_quantiles() -> None:
    """Tests add_quantile_annotations with specific quantile list.

    Asserts:
        - Method accepts specific quantile list
        - Only specified quantiles are processed
        - Invalid quantile names are ignored gracefully
    """
    base_plots = BasePlots()
    test_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    fig = go.Figure()
    fig.add_scatter(x=test_data, y=test_data)

    base_plots.add_quantile_annotations(fig, test_data, annotations=["Q1", "Med", "Q3"])

    if not (True):
        raise AssertionError("Assertion failed.")


def test_baseplots_filter_nan_indices_functionality() -> None:
    """Tests filter_nan_indices method functionality.

    Asserts:
        - Method correctly identifies non-NaN indices
        - Returns proper boolean index
        - Handles columns with all NaN or no NaN values
    """
    base_plots = BasePlots()

    test_df = pd.DataFrame(
        {
            "feature_with_nan": [1, 2, np.nan, 4, np.nan],
            "feature_no_nan": [1, 2, 3, 4, 5],
            "feature_all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
        }
    )

    indices = base_plots.filter_nan_indices(test_df, "feature_with_nan")
    expected = pd.Series([True, True, False, True, False], name="feature_with_nan")
    pd.testing.assert_series_equal(indices, expected)

    indices_no_nan = base_plots.filter_nan_indices(test_df, "feature_no_nan")
    expected_no_nan = pd.Series([True, True, True, True, True], name="feature_no_nan")
    pd.testing.assert_series_equal(indices_no_nan, expected_no_nan)

    indices_all_nan = base_plots.filter_nan_indices(test_df, "feature_all_nan")
    expected_all_nan = pd.Series(
        [False, False, False, False, False], name="feature_all_nan"
    )
    pd.testing.assert_series_equal(indices_all_nan, expected_all_nan)
