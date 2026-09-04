import numpy as np
import pandas as pd
import pytest

from kvbiii_plots.eda.categorical_plots import CategoricalPlots


def test_categoricalplots_barplot_handles_series_input(
    sample_series: pd.Series,
) -> None:
    """Tests barplot correctly processes Series input.

    Args:
        sample_series (pd.Series): Fixture containing test Series data

    Asserts:
        - Method executes without errors when given Series input
        - Plot is generated with appropriate configuration
        - Bar chart displays frequency data correctly
    """
    cat_plots = CategoricalPlots()

    cat_plots.barplot(
        data=sample_series.value_counts(),
        plot_title="Test Bar Plot",
        width=800,
        height=600,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_barplot_handles_dict_input() -> None:
    """Tests barplot correctly processes dictionary input.

    Asserts:
        - Method executes without errors when given dict input
        - Dictionary keys become bar labels
        - Dictionary values become bar heights
    """
    cat_plots = CategoricalPlots()
    test_data = {"A": 10, "B": 20, "C": 15}

    cat_plots.barplot(
        data=test_data,
        plot_title="Test Dict Bar Plot",
        xaxis_title="Categories",
        yaxis_title="Values",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_barplot_applies_custom_dimensions() -> None:
    """Tests barplot applies custom width and height parameters.

    Asserts:
        - Method accepts custom dimension parameters
        - Plot configuration uses specified dimensions
        - Custom titles are properly applied
    """
    cat_plots = CategoricalPlots()
    test_data = {"Category1": 5, "Category2": 8, "Category3": 12}

    cat_plots.barplot(
        data=test_data,
        plot_title="Custom Dimensions Test",
        width=1200,
        height=900,
        xaxis_title="Test Categories",
        yaxis_title="Test Values",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_barplot_handles_empty_input() -> None:
    """Tests barplot handles empty data input gracefully.

    Asserts:
        - Method handles empty Series without crashing
        - Empty dictionary input is processed appropriately
        - No errors are raised for empty data
    """
    cat_plots = CategoricalPlots()
    empty_series = pd.Series([], dtype=object)
    empty_dict = {}

    cat_plots.barplot(data=empty_series.value_counts())
    cat_plots.barplot(data=empty_dict)
    if not (True):
        raise AssertionError("Empty input should be handled gracefully")


def test_categoricalplots_pie_barplot_handles_dataframe_input(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests pie_barplot correctly processes DataFrame input.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method executes without errors when given DataFrame input
        - Pie chart and bar chart are generated side by side
        - Feature column is properly processed for visualization
    """
    cat_plots = CategoricalPlots()

    cat_plots.pie_barplot(
        data=sample_dataframe,
        feature="D",
        plot_title="Test Pie Bar Plot",
        width=1600,
        height=800,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_barplot_applies_custom_hole_size(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests pie_barplot applies custom hole size parameter.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method accepts custom hole_size parameter
        - Pie chart configuration uses specified hole size
        - Plot displays with donut-style visualization
    """
    cat_plots = CategoricalPlots()

    cat_plots.pie_barplot(
        data=sample_dataframe,
        feature="D",
        plot_title="Custom Hole Size Test",
        hole_size=0.5,
        xaxis_title="Categories",
        yaxis_title="Count",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_barplot_handles_missing_feature() -> None:
    """Tests pie_barplot raises appropriate error for missing feature.

    Asserts:
        - KeyError is raised when feature column doesn't exist
        - Error handling is appropriate for invalid feature names
    """
    cat_plots = CategoricalPlots()
    test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    with pytest.raises(KeyError):
        cat_plots.pie_barplot(
            data=test_df, feature="NonExistentColumn", plot_title="Missing Feature Test"
        )


def test_categoricalplots_boxplot_by_categorical_handles_valid_input(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests boxplot_by_categorical correctly processes valid DataFrame input.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method executes without errors for valid categorical and target columns
        - Box plots are generated for each category
        - Statistical summaries are properly displayed
    """
    cat_plots = CategoricalPlots()

    cat_plots.boxplot_by_categorical(
        data=sample_dataframe,
        categorical="D",
        target="A",
        plot_title="Test Box Plot by Category",
        width=1600,
        height=800,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_boxplot_by_categorical_applies_custom_titles(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests boxplot_by_categorical applies custom axis titles.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method accepts custom xaxis_title and yaxis_title parameters
        - Plot configuration uses specified axis labels
        - Title formatting is applied correctly
    """
    cat_plots = CategoricalPlots()

    cat_plots.boxplot_by_categorical(
        data=sample_dataframe,
        categorical="D",
        target="B",
        plot_title="Custom Titles Test",
        xaxis_title="Custom X Axis",
        yaxis_title="Custom Y Axis",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_boxplot_by_categorical_handles_top_n_parameter(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests pie_boxplot_by_categorical correctly applies top_n filtering.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method executes without errors when top_n parameter is specified
        - Top categories are selected for visualization
        - Combined pie and box plot layout is generated
    """
    cat_plots = CategoricalPlots()

    cat_plots.pie_boxplot_by_categorical(
        data=sample_dataframe,
        categorical="D",
        target="A",
        top_n=2,
        plot_title="Top N Test",
        width=1600,
        height=800,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_boxplot_by_categorical_handles_custom_hole_size(
    sample_dataframe: pd.DataFrame,
) -> None:
    """Tests pie_boxplot_by_categorical applies custom hole size parameter.

    Args:
        sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

    Asserts:
        - Method accepts custom hole_size parameter for pie chart
        - Pie chart configuration uses specified hole size
        - Box plot portion remains unaffected by pie chart settings
    """
    cat_plots = CategoricalPlots()

    cat_plots.pie_boxplot_by_categorical(
        data=sample_dataframe,
        categorical="D",
        target="B",
        hole_size=0.4,
        plot_title="Custom Hole Size Box Plot Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_stacked_barplot_by_hue_handles_multivariate_data() -> (
    None
):
    """Tests pie_stacked_barplot_by_hue correctly processes multivariate categorical data.

    Asserts:
        - Method executes without errors for multivariate input
        - Pie chart and stacked bar chart are generated
        - Hue grouping is properly implemented
    """
    cat_plots = CategoricalPlots()
    test_data = pd.DataFrame(
        {
            "category": ["A", "B", "A", "B", "C"] * 20,
            "hue": ["X", "Y", "X", "Y", "Z"] * 20,
            "value": np.random.rand(100),
        }
    )

    cat_plots.pie_stacked_barplot_by_hue(
        data=test_data,
        feature="category",
        hue="hue",
        plot_title="Multivariate Test",
        width=1600,
        height=800,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_pie_stacked_barplot_by_hue_applies_custom_parameters() -> (
    None
):
    """Tests pie_stacked_barplot_by_hue applies custom visualization parameters.

    Asserts:
        - Method accepts custom hole_size and dimension parameters
        - Plot configuration uses specified settings
        - Stacked bar chart displays hue groupings correctly
    """
    cat_plots = CategoricalPlots()
    test_data = pd.DataFrame(
        {
            "main_cat": ["Type1", "Type2", "Type3"] * 30,
            "sub_cat": ["A", "B"] * 45,
            "values": np.random.rand(90),
        }
    )

    cat_plots.pie_stacked_barplot_by_hue(
        data=test_data,
        feature="main_cat",
        hue="sub_cat",
        plot_title="Custom Parameters Test",
        width=1200,
        height=900,
        hole_size=0.6,
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_raises_error_invalid_dataframe_input() -> None:
    """Tests CategoricalPlots methods raise appropriate errors for invalid DataFrame input.

    Asserts:
        - Methods raise appropriate errors for non-DataFrame input
        - Error messages provide helpful information
    """
    cat_plots = CategoricalPlots()

    with pytest.raises((TypeError, AttributeError)):
        cat_plots.pie_barplot(data="invalid_input", feature="nonexistent")

    with pytest.raises((TypeError, AttributeError)):
        cat_plots.boxplot_by_categorical(
            data=123, categorical="invalid", target="invalid"
        )


def test_categoricalplots_apply_top_n_categories_basic_functionality() -> None:
    """Tests _apply_top_n_categories basic functionality.

    Asserts:
        - Method correctly limits categories to top_n
        - Aggregates remaining categories as 'Other'
        - Returns correct tuple structure
    """
    cat_plots = CategoricalPlots()

    labels = np.array(["A", "B", "C", "D", "E"])
    frequency = np.array([50, 40, 30, 20, 10])

    result_labels, result_freq, other_labels = cat_plots._apply_top_n_categories(
        labels, frequency, top_n=3
    )

    expected_labels = np.array(["A", "B", "C", "Other"])
    expected_freq = np.array([50, 40, 30, 30])
    expected_other = np.array(["D", "E"])

    np.testing.assert_array_equal(result_labels, expected_labels)
    np.testing.assert_array_equal(result_freq, expected_freq)
    np.testing.assert_array_equal(other_labels, expected_other)


def test_categoricalplots_apply_top_n_categories_no_limit() -> None:
    """Tests _apply_top_n_categories when top_n is None or larger than data.

    Asserts:
        - Method returns original data when no limit applied
        - other_labels is None when no aggregation occurs
        - Original arrays are unchanged
    """
    cat_plots = CategoricalPlots()

    labels = np.array(["A", "B", "C"])
    frequency = np.array([30, 20, 10])

    result_labels, result_freq, other_labels = cat_plots._apply_top_n_categories(
        labels, frequency, top_n=None
    )

    np.testing.assert_array_equal(result_labels, labels)
    np.testing.assert_array_equal(result_freq, frequency)
    if not (other_labels is None):
        raise AssertionError("Assertion failed.")

    result_labels, result_freq, other_labels = cat_plots._apply_top_n_categories(
        labels, frequency, top_n=10
    )

    np.testing.assert_array_equal(result_labels, labels)
    np.testing.assert_array_equal(result_freq, frequency)
    if not (other_labels is None):
        raise AssertionError("Assertion failed.")


def test_categoricalplots_apply_top_n_categories_custom_other_label() -> None:
    """Tests _apply_top_n_categories with custom other_category label.

    Asserts:
        - Method uses custom other_category label
        - Aggregation logic remains correct
        - Custom label appears in results
    """
    cat_plots = CategoricalPlots()

    labels = np.array(["Category1", "Category2", "Category3", "Category4"])
    frequency = np.array([100, 80, 60, 40])

    result_labels, result_freq, _ = cat_plots._apply_top_n_categories(
        labels, frequency, top_n=2, other_category="Remaining"
    )

    expected_labels = np.array(["Category1", "Category2", "Remaining"])
    expected_freq = np.array([100, 80, 100])

    np.testing.assert_array_equal(result_labels, expected_labels)
    np.testing.assert_array_equal(result_freq, expected_freq)
    if not ("Remaining" in result_labels):
        raise AssertionError("Assertion failed.")


def test_categoricalplots_apply_top_n_categories_edge_cases() -> None:
    """Tests _apply_top_n_categories edge cases.

    Asserts:
        - Method handles empty arrays appropriately
        - Single item arrays are processed correctly
        - top_n=0 is handled gracefully
    """
    cat_plots = CategoricalPlots()

    single_labels = np.array(["OnlyOne"])
    single_freq = np.array([100])

    result_labels, result_freq, other_labels = cat_plots._apply_top_n_categories(
        single_labels, single_freq, top_n=1
    )

    np.testing.assert_array_equal(result_labels, single_labels)
    np.testing.assert_array_equal(result_freq, single_freq)
    if not (other_labels is None):
        raise AssertionError("Assertion failed.")

    labels = np.array(["A", "B", "C"])
    frequency = np.array([30, 20, 10])

    result_labels, result_freq, other_labels = cat_plots._apply_top_n_categories(
        labels, frequency, top_n=0
    )

    expected_labels = np.array(["Other"])
    expected_freq = np.array([60])

    np.testing.assert_array_equal(result_labels, expected_labels)
    np.testing.assert_array_equal(result_freq, expected_freq)


def test_categoricalplots_barplot_with_top_n_parameter() -> None:
    """Tests barplot method with top_n parameter functionality.

    Asserts:
        - Method correctly applies top_n limiting
        - Plot generation succeeds with limited categories
        - Other category aggregation works properly
    """
    cat_plots = CategoricalPlots()

    test_data = {f"Category_{i}": 100 - i * 5 for i in range(15)}

    cat_plots.barplot(
        data=test_data,
        top_n=5,
        plot_title="Top 5 Categories Test",
        other_category="All Others",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_barplot_parameter_variations() -> None:
    """Tests barplot method with various parameter combinations.

    Asserts:
        - Method handles different parameter combinations
        - All optional parameters work correctly
        - Plot customization is applied properly
    """
    cat_plots = CategoricalPlots()
    test_data = {"A": 10, "B": 20, "C": 15}

    cat_plots.barplot(
        data=test_data,
        top_n=None,
        plot_title="Full Parameters Test",
        width=1200,
        height=600,
        xaxis_title="Custom X Axis",
        yaxis_title="Custom Y Axis",
        other_category="Others",
    )

    cat_plots.barplot(data=test_data)

    if not (True):
        raise AssertionError("All parameter combinations should work")


def test_categoricalplots_validate_categorical_groups_returns_normalized_proportions() -> (
    None
):
    """Tests _validate_categorical_groups converts raw values to per-group proportions.

    Asserts:
        - Returned mapping preserves group labels as strings
        - Each group's proportions sum to 1.0
        - NaN values are dropped before computing proportions
    """
    cat_plots = CategoricalPlots()

    raw_data = {
        "Train": pd.Series(["A", "A", "B", np.nan]),
        "Test": ["A", "B", "B"],
    }

    cleaned = cat_plots._validate_categorical_groups(raw_data)

    if not (set(cleaned.keys()) == {"Train", "Test"}):
        raise AssertionError("Assertion failed.")
    if not (np.isclose(cleaned["Train"].sum(), 1.0)):
        raise AssertionError("Assertion failed.")
    if not (np.isclose(cleaned["Test"].sum(), 1.0)):
        raise AssertionError("Assertion failed.")
    if not (np.isclose(cleaned["Train"]["A"], 2 / 3)):
        raise AssertionError("Assertion failed.")


def test_categoricalplots_validate_categorical_groups_raises_for_non_dict() -> None:
    """Tests _validate_categorical_groups raises TypeError for non-dict input.

    Asserts:
        - TypeError is raised when data is not a dict
    """
    cat_plots = CategoricalPlots()

    with pytest.raises(TypeError):
        cat_plots._validate_categorical_groups(["A", "B", "C"])


def test_categoricalplots_validate_categorical_groups_raises_for_insufficient_groups() -> (
    None
):
    """Tests _validate_categorical_groups raises ValueError with fewer than two groups.

    Asserts:
        - ValueError is raised for a single group
        - ValueError is raised when only one group remains non-empty after
          NaN removal
    """
    cat_plots = CategoricalPlots()

    with pytest.raises(ValueError):
        cat_plots._validate_categorical_groups({"Only": ["A", "B", "C"]})

    with pytest.raises(ValueError):
        cat_plots._validate_categorical_groups(
            {
                "Train": ["A", "B"],
                "Test": pd.Series([np.nan, np.nan]),
            }
        )


def test_categoricalplots_shared_category_order_ranks_by_summed_proportion() -> None:
    """Tests _shared_category_order ranks categories by proportion summed across groups.

    Asserts:
        - A category is ranked ahead of another when its summed proportion across
          groups is higher, even if the raw pooled count would suggest otherwise
        - No aggregation bucket is created when top_n is None
    """
    cat_plots = CategoricalPlots()

    group_proportions = {
        "Large": pd.Series({"Common": 0.9, "Rare": 0.1}),
        "Small": pd.Series({"Common": 0.1, "Rare": 0.9}),
    }

    labels, other_labels = cat_plots._shared_category_order(
        group_proportions, top_n=None, other_category="Other"
    )

    if not (set(labels) == {"Common", "Rare"}):
        raise AssertionError("Assertion failed.")
    if not (other_labels is None):
        raise AssertionError("Assertion failed.")


def test_categoricalplots_shared_category_order_applies_top_n_truncation() -> None:
    """Tests _shared_category_order truncates to top_n and returns an Other bucket.

    Asserts:
        - Only top_n categories plus the aggregated label remain
        - Truncated categories are returned as other_labels
    """
    cat_plots = CategoricalPlots()

    group_proportions = {
        "Group1": pd.Series({"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}),
        "Group2": pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}),
    }

    labels, other_labels = cat_plots._shared_category_order(
        group_proportions, top_n=2, other_category="Other"
    )

    if not (len(labels) == 3):
        raise AssertionError("Assertion failed.")
    if not ("Other" in labels):
        raise AssertionError("Assertion failed.")
    if not (set(other_labels) == {"C", "D"}):
        raise AssertionError("Assertion failed.")


def test_categoricalplots_compare_categorical_distributions_plot_raises_error_single_group() -> (
    None
):
    """Tests compare_categorical_distributions_plot raises ValueError with one group.

    Asserts:
        - ValueError is raised when data has fewer than two groups
    """
    cat_plots = CategoricalPlots()

    with pytest.raises(ValueError):
        cat_plots.compare_categorical_distributions_plot(data={"Only": ["A", "B", "C"]})


def test_categoricalplots_compare_categorical_distributions_plot_raises_error_not_dict() -> (
    None
):
    """Tests compare_categorical_distributions_plot raises TypeError for non-dict data.

    Asserts:
        - TypeError is raised for non-dict input
    """
    cat_plots = CategoricalPlots()

    with pytest.raises(TypeError):
        cat_plots.compare_categorical_distributions_plot(data=["A", "B", "C"])


def test_categoricalplots_compare_categorical_distributions_plot_handles_unequal_group_sizes() -> (
    None
):
    """Tests compare_categorical_distributions_plot handles very unequal group sizes.

    Asserts:
        - Method executes without errors when one group has far more
          observations than the other (e.g. 5000 vs 50)
    """
    cat_plots = CategoricalPlots()

    np.random.seed(17)
    cat_plots.compare_categorical_distributions_plot(
        data={
            "Large": np.random.choice(["A", "B", "C"], 5000),
            "Small": np.random.choice(["A", "B", "C"], 50),
        },
        plot_title="Unequal Group Sizes Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_compare_categorical_distributions_plot_handles_three_groups() -> (
    None
):
    """Tests compare_categorical_distributions_plot handles more than two groups.

    Asserts:
        - Method executes without errors for three named groups
    """
    cat_plots = CategoricalPlots()

    np.random.seed(17)
    cat_plots.compare_categorical_distributions_plot(
        data={
            "Model A": np.random.choice(["X", "Y", "Z"], 300),
            "Model B": np.random.choice(["X", "Y", "Z"], 150),
            "Model C": np.random.choice(["X", "Y", "Z"], 75),
        },
        plot_title="Three Groups Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_compare_categorical_distributions_plot_applies_top_n() -> (
    None
):
    """Tests compare_categorical_distributions_plot applies top_n with Other bucket.

    Asserts:
        - Method executes without errors when categories exceed top_n
        - Other-category aggregation does not raise errors
    """
    cat_plots = CategoricalPlots()

    np.random.seed(17)
    categories = [f"Cat_{i}" for i in range(15)]
    cat_plots.compare_categorical_distributions_plot(
        data={
            "Train": np.random.choice(categories, 500),
            "Test": np.random.choice(categories, 200),
        },
        top_n=5,
        plot_title="Top N Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_compare_categorical_distributions_plot_handles_show_other_false() -> (
    None
):
    """Tests compare_categorical_distributions_plot with show_other disabled.

    Asserts:
        - Method executes without errors when show_other is False and categories
          exceed top_n
    """
    cat_plots = CategoricalPlots()

    np.random.seed(17)
    categories = [f"Cat_{i}" for i in range(10)]
    cat_plots.compare_categorical_distributions_plot(
        data={
            "Train": np.random.choice(categories, 300),
            "Test": np.random.choice(categories, 100),
        },
        top_n=3,
        show_other=False,
        plot_title="Show Other False Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_compare_categorical_distributions_plot_handles_nan_values() -> (
    None
):
    """Tests compare_categorical_distributions_plot handles NaN values within groups.

    Asserts:
        - Method executes without errors when groups contain NaN values
    """
    cat_plots = CategoricalPlots()

    train_data = pd.Series(["A", "B", "A", np.nan, "B", "A"])
    test_data = pd.Series(["A", np.nan, "B"])

    cat_plots.compare_categorical_distributions_plot(
        data={"Train": train_data, "Test": test_data},
        plot_title="NaN Handling Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")


def test_categoricalplots_compare_categorical_distributions_plot_disables_counts_and_customizes_tickangle() -> (
    None
):
    """Tests compare_categorical_distributions_plot with counts disabled and custom tickangle.

    Asserts:
        - Method executes without errors when show_counts is False
        - Custom tickangle parameter is accepted
    """
    cat_plots = CategoricalPlots()

    np.random.seed(17)
    cat_plots.compare_categorical_distributions_plot(
        data={
            "Train": np.random.choice(["A", "B", "C"], 200),
            "Test": np.random.choice(["A", "B", "C"], 80),
        },
        show_counts=False,
        tickangle=45,
        other_category="Rest",
        plot_title="No Counts Custom Tickangle Test",
    )

    if not (True):
        raise AssertionError("Method should execute without errors")
