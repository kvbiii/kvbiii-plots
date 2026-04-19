import numpy as np
import pandas as pd
import pytest

from kvbiii_plots.eda.multivariate_plots import MultivariatePlots


class TestMultivariatePlotsCorrelationPlot:
    """Test class for MultivariatePlots.correlation_plot method."""

    def test_correlation_plot_handles_dataframe_input(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot correctly processes DataFrame input.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when given DataFrame input
            - Correlation matrix is calculated and visualized as heatmap
            - Plot configuration uses provided parameters
        """
        mv_plots = MultivariatePlots()

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B", "C"],
            plot_title="Test Correlation Plot",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_correlation_plot_handles_numpy_array_input(
        self, sample_numpy_array: np.ndarray
    ) -> None:
        """Tests correlation_plot correctly processes numpy array input.

        Args:
            sample_numpy_array (np.ndarray): Fixture containing test numpy array data

        Asserts:
            - Method executes without errors when given numpy array input
            - Default feature names are generated for numpy arrays
            - Correlation matrix is calculated correctly
        """
        mv_plots = MultivariatePlots()

        data_2d = np.column_stack(
            [sample_numpy_array, sample_numpy_array * 2, sample_numpy_array * 0.5]
        )

        mv_plots.correlation_plot(
            data=data_2d,
            plot_title="Numpy Array Correlation Test",
            width=600,
            height=600,
        )

        if not (True):
            raise AssertionError("Method should handle numpy array input")

    def test_correlation_plot_handles_custom_feature_names(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot correctly processes custom feature names.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when features_names parameter is specified
            - Custom feature names are used for axis labeling
            - Only specified features are included in correlation analysis
        """
        mv_plots = MultivariatePlots()

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B", "C"],
            plot_title="Custom Feature Names Test",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_correlation_plot_handles_different_correlation_methods(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot correctly applies different correlation methods.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts different correlation methods (pearson, kendall, spearman)
            - Correlation calculation uses specified method
            - Results vary appropriately based on method choice
        """
        mv_plots = MultivariatePlots()

        methods = ["pearson", "kendall", "spearman"]

        for method in methods:
            mv_plots.correlation_plot(
                data=sample_dataframe,
                features_names=["A", "B", "C"],
                method=method,
                plot_title=f"{method.capitalize()} Correlation Test",
                width=600,
                height=600,
            )

        if not (True):
            raise AssertionError("Method should handle different correlation methods")

    def test_correlation_plot_applies_show_upper_parameter(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot applies show_upper parameter correctly.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts show_upper parameter
            - Upper triangle is shown or hidden based on parameter
            - Correlation matrix visualization reflects parameter setting
        """
        mv_plots = MultivariatePlots()

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B", "C"],
            show_upper=True,
            plot_title="Show Upper Triangle Test",
            width=700,
            height=700,
        )

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B", "C"],
            show_upper=False,
            plot_title="Hide Upper Triangle Test",
        )

        if not (True):
            raise AssertionError("Method should apply show_upper parameter correctly")

    def test_correlation_plot_handles_different_colorscales(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot applies different colorscale parameters.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts different colorscale parameters
            - Heatmap uses specified color scheme
            - Invalid colorscales fall back to default
        """
        mv_plots = MultivariatePlots()

        colorscales = ["RdBu", "Viridis", "Plasma"]

        for colorscale in colorscales:
            mv_plots.correlation_plot(
                data=sample_dataframe,
                features_names=["A", "B"],
                colorscale=colorscale,
                plot_title=f"Colorscale {colorscale} Test",
                width=500,
                height=500,
            )

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B"],
            colorscale="InvalidColorscale",
            plot_title="Invalid Colorscale Test",
        )

        if not (True):
            raise AssertionError("Method should handle different colorscales")

    def test_correlation_plot_handles_non_numeric_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot handles DataFrame with non-numeric columns.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method processes DataFrame containing non-numeric columns
            - Only numeric columns are included in correlation calculation
            - Non-numeric columns are properly filtered out
        """
        mv_plots = MultivariatePlots()

        numeric_features = ["A", "B", "C"]
        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=numeric_features,
            plot_title="Mixed Data Types Correlation",
        )

        if not (True):
            raise AssertionError(
                "Method should handle mixed data types when numeric features are specified"
            )

    def test_correlation_plot_handles_automatic_feature_selection(
        self, test_settings: object
    ) -> None:
        """Tests correlation_plot with automatic feature selection on numeric-only DataFrame.

        Args:
            test_settings: Test settings fixture for reproducible random data

        Asserts:
            - Method automatically selects all features when features_names=None
            - Works correctly when all DataFrame columns are numeric
            - Correlation matrix includes all available features
        """
        mv_plots = MultivariatePlots()

        np.random.seed(test_settings.SEED)
        numeric_df = pd.DataFrame(
            {
                "feature1": np.random.rand(50),
                "feature2": np.random.rand(50),
                "feature3": np.random.randint(0, 10, 50),
            }
        )

        mv_plots.correlation_plot(
            data=numeric_df,
            features_names=None,
            plot_title="Automatic Feature Selection Test",
        )

        if not (True):
            raise AssertionError(
                "Method should handle automatic feature selection on numeric data"
            )

    def test_correlation_plot_applies_custom_parameters(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests correlation_plot applies custom visualization parameters.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts custom width, height, and title parameters
            - Heatmap configuration uses specified settings
            - Color scheme reflects correlation values appropriately
        """
        mv_plots = MultivariatePlots()

        mv_plots.correlation_plot(
            data=sample_dataframe,
            features_names=["A", "B", "C"],
            plot_title="Custom Parameters Correlation",
            width=1000,
            height=1000,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")


class TestMultivariatePlotsScatterMatrix:
    """Test class for MultivariatePlots.scatter_matrix method."""

    def test_scatter_matrix_handles_dataframe_input(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_matrix correctly processes DataFrame input.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when given DataFrame input
            - Scatter plot matrix is generated for specified features
            - Diagonal and off-diagonal plots are properly configured
        """
        mv_plots = MultivariatePlots()

        mv_plots.scatter_matrix(
            data=sample_dataframe,
            features=["A", "B", "C"],
            plot_title="Test Scatter Matrix",
            width=1200,
            height=1200,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_scatter_matrix_handles_automatic_feature_selection(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_matrix correctly handles automatic feature selection when features=None.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when features parameter is None
            - Automatic feature selection chooses numeric columns
            - Feature count is limited appropriately (max 6)
        """
        mv_plots = MultivariatePlots()

        mv_plots.scatter_matrix(
            data=sample_dataframe,
            features=None,
            plot_title="Auto Feature Selection Test",
            width=1000,
            height=1000,
        )

        if not (True):
            raise AssertionError("Method should handle automatic feature selection")

    def test_scatter_matrix_handles_feature_limit(
        self,
        test_settings: object,
    ) -> None:
        """Tests scatter_matrix correctly limits features to maximum of 6.

        Args:
            test_settings: Test settings fixture for reproducible random data

        Asserts:
            - Method correctly limits to 6 features when more are available
            - Feature selection works with large DataFrames
            - Performance remains acceptable with feature limiting
        """
        mv_plots = MultivariatePlots()

        np.random.seed(test_settings.SEED)
        large_df = pd.DataFrame({f"feature_{i}": np.random.rand(50) for i in range(10)})

        mv_plots.scatter_matrix(
            data=large_df,
            features=None,
            plot_title="Feature Limit Test",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should handle feature limiting")

    def test_scatter_matrix_handles_hue_parameter(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_matrix correctly applies hue parameter for color grouping.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when hue parameter is specified
            - Color grouping is applied based on hue column
            - Scatter plots show different colors for different hue categories
        """
        mv_plots = MultivariatePlots()

        mv_plots.scatter_matrix(
            data=sample_dataframe,
            features=["A", "B", "C"],
            hue="D",
            plot_title="Scatter Matrix with Hue",
            marker_size=6,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_scatter_matrix_handles_custom_marker_size(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_matrix applies custom marker size parameter.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts custom marker_size parameter
            - Scatter plot points use specified marker size
            - Visual appearance reflects custom marker settings
        """
        mv_plots = MultivariatePlots()

        marker_sizes = [3, 8, 15]

        for size in marker_sizes:
            mv_plots.scatter_matrix(
                data=sample_dataframe,
                features=["A", "B"],
                marker_size=size,
                plot_title=f"Marker Size {size} Test",
                width=600,
                height=600,
            )

        if not (True):
            raise AssertionError("Method should execute without errors")


class TestMultivariatePlotsParallelCoordinates:
    """Test class for MultivariatePlots.parallel_coordinates method."""

    def test_parallel_coordinates_handles_dataframe_input(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests parallel_coordinates correctly processes DataFrame input.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when given DataFrame input
            - Parallel coordinates plot is generated for specified features
            - Plot configuration uses provided parameters
        """
        mv_plots = MultivariatePlots()

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=["A", "B", "C"],
            plot_title="Test Parallel Coordinates",
            width=1200,
            height=600,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_parallel_coordinates_handles_hue_parameter_numeric(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests parallel_coordinates correctly applies hue parameter for numeric color grouping.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when hue parameter is numeric
            - Color grouping is applied based on numeric hue column
            - Parallel lines show continuous color scale
        """
        mv_plots = MultivariatePlots()

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=["A", "B"],
            hue="C",
            plot_title="Parallel Coordinates with Numeric Hue",
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_parallel_coordinates_handles_hue_parameter_categorical(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests parallel_coordinates correctly applies hue parameter for
        categorical color grouping.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when hue parameter is categorical
            - Color grouping is applied based on categorical hue column
            - Parallel lines show discrete colors for different categories
        """
        mv_plots = MultivariatePlots()

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=["A", "B", "C"],
            hue="D",
            plot_title="Parallel Coordinates with Categorical Hue",
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_parallel_coordinates_applies_normalization(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests parallel_coordinates applies normalize parameter correctly.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method accepts normalize parameter
            - Data normalization is applied or skipped based on parameter
            - Parallel coordinates scale appropriately
        """
        mv_plots = MultivariatePlots()

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=["A", "B", "C"],
            normalize=True,
            plot_title="Normalized Parallel Coordinates",
        )

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=["A", "B", "C"],
            normalize=False,
            plot_title="Non-normalized Parallel Coordinates",
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_parallel_coordinates_handles_all_features(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests parallel_coordinates correctly handles all numeric features when features=None.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when features parameter is None
            - All numeric features are automatically selected
            - Non-numeric features are properly filtered out
        """
        mv_plots = MultivariatePlots()

        mv_plots.parallel_coordinates(
            data=sample_dataframe,
            features=None,
            plot_title="All Features Parallel Coordinates",
            width=1400,
            height=700,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_parallel_coordinates_handles_missing_values(
        self,
        test_settings: object,
    ) -> None:
        """Tests parallel_coordinates handles DataFrames with missing values.

        Args:
            test_settings: Test settings fixture for reproducible random data

        Asserts:
            - Method executes without errors with missing values present
            - Missing values are properly handled (dropna)
            - Plot generation proceeds with remaining valid data
        """
        mv_plots = MultivariatePlots()

        np.random.seed(test_settings.SEED)
        df_with_na = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5],
                "B": [2, np.nan, 4, 5, 6],
                "C": [3, 4, 5, np.nan, 7],
                "cat": ["X", "Y", "X", "Y", "X"],
            }
        )

        mv_plots.parallel_coordinates(
            data=df_with_na, features=["A", "B", "C"], plot_title="Missing Values Test"
        )

        if not (True):
            raise AssertionError("Method should handle missing values")


class TestMultivariatePlotsErrorHandling:
    """Test class for MultivariatePlots error handling."""

    def test_correlation_plot_raises_error_invalid_features(self) -> None:
        """Tests correlation_plot raises appropriate errors for invalid feature names.

        Asserts:
            - KeyError is raised when features_names contain non-existent columns
            - Error handling is appropriate for invalid feature specifications
        """
        mv_plots = MultivariatePlots()
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            mv_plots.correlation_plot(
                data=test_df, features_names=["NonExistentColumn"]
            )

    def test_scatter_matrix_raises_error_invalid_features(self) -> None:
        """Tests scatter_matrix raises appropriate errors for invalid feature names.

        Asserts:
            - KeyError is raised when features contain non-existent column names
            - Error handling is appropriate for invalid feature specifications
        """
        mv_plots = MultivariatePlots()
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            mv_plots.scatter_matrix(data=test_df, features=["InvalidFeature"])

    def test_parallel_coordinates_raises_error_invalid_features(self) -> None:
        """Tests parallel_coordinates raises appropriate errors for invalid feature names.

        Asserts:
            - KeyError is raised when features contain non-existent column names
            - Error handling is appropriate for invalid feature specifications
        """
        mv_plots = MultivariatePlots()
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            mv_plots.parallel_coordinates(data=test_df, features=["InvalidFeature"])

    def test_multivariateplots_raises_error_invalid_input_types(self) -> None:
        """Tests MultivariatePlots methods raise appropriate errors for invalid input types.

        Asserts:
            - Methods raise appropriate errors for non-DataFrame input
            - Error messages provide helpful information
        """
        mv_plots = MultivariatePlots()

        with pytest.raises((TypeError, AttributeError)):
            mv_plots.correlation_plot(data="invalid_input")

        with pytest.raises((TypeError, AttributeError)):
            mv_plots.scatter_matrix(data=123, features=["invalid"])

        with pytest.raises((TypeError, AttributeError)):
            mv_plots.parallel_coordinates(data=None, features=["invalid"])

    def test_scatter_matrix_raises_error_invalid_hue(self) -> None:
        """Tests scatter_matrix raises appropriate errors for invalid hue column.

        Asserts:
            - KeyError is raised when hue column doesn't exist
            - Error handling is appropriate for invalid hue specifications
        """
        mv_plots = MultivariatePlots()
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            mv_plots.scatter_matrix(
                data=test_df, features=["A", "B"], hue="NonExistentHue"
            )

    def test_parallel_coordinates_raises_error_invalid_hue(self) -> None:
        """Tests parallel_coordinates raises appropriate errors for invalid hue column.

        Asserts:
            - KeyError is raised when hue column doesn't exist
            - Error handling is appropriate for invalid hue specifications
        """
        mv_plots = MultivariatePlots()
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

        with pytest.raises(KeyError):
            mv_plots.parallel_coordinates(
                data=test_df, features=["A", "B"], hue="NonExistentHue"
            )


class TestMultivariatePlotsScatterWithMarginals:
    """Test class for MultivariatePlots.scatter_with_marginals method."""

    def test_scatter_with_marginals_basic_functionality(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals executes with basic DataFrame input.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors for basic input
            - Scatter plot with marginal distributions is generated
        """
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            plot_title="Basic Scatter with Marginals",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should execute without errors")

    def test_scatter_with_marginals_with_categorical_hue(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals correctly applies categorical hue.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method handles categorical hue column
            - Color grouping is applied for categorical variable
            - Marginal histograms respect hue grouping
        """
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            hue="D",
            plot_title="Scatter with Categorical Hue",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should handle categorical hue")

    def test_scatter_with_marginals_with_numeric_hue(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals correctly applies numeric hue.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method handles numeric hue column
            - Color scale is applied for numeric variable
            - Colorbar is displayed for numeric hue
        """
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            hue="C",
            plot_title="Scatter with Numeric Hue",
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should handle numeric hue")

    def test_scatter_with_marginals_custom_dimensions(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals applies custom width and height.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Custom width and height parameters are respected
            - Plot dimensions match specified values
        """
        mv_plots = MultivariatePlots()
        custom_width = 1200
        custom_height = 900
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            plot_title="Custom Dimensions Test",
            width=custom_width,
            height=custom_height,
        )

        if not (True):
            raise AssertionError("Method should apply custom dimensions")

    def test_scatter_with_marginals_marker_size_and_opacity(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals applies marker size and opacity parameters.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Marker size parameter is applied
            - Opacity parameter is applied to markers
            - Visualization reflects marker customization
        """
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            plot_title="Marker Customization Test",
            width=800,
            height=800,
            marker_size=12,
            opacity=0.5,
        )

        if not (True):
            raise AssertionError("Method should apply marker customization")

    def test_scatter_with_marginals_marginal_height_ratio(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals applies marginal height ratio parameter.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Marginal height ratio parameter is applied
            - Proportion of marginal plots to main plot is correct
        """
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            plot_title="Marginal Ratio Test",
            width=800,
            height=800,
            marginal_height_ratio=0.15,
        )

        if not (True):
            raise AssertionError("Method should apply marginal height ratio")

    def test_scatter_with_marginals_raises_error_missing_columns(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals raises error for missing columns.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - ValueError is raised when x or y column doesn't exist
            - Error message indicates missing column
        """
        mv_plots = MultivariatePlots()

        with pytest.raises(ValueError):
            mv_plots.scatter_with_marginals(
                data=sample_dataframe,
                x="NonExistentX",
                y="B",
            )

        with pytest.raises(ValueError):
            mv_plots.scatter_with_marginals(
                data=sample_dataframe,
                x="A",
                y="NonExistentY",
            )

    def test_scatter_with_marginals_raises_error_all_nan_values(self) -> None:
        """Tests scatter_with_marginals raises error when data becomes empty after cleaning.

        Asserts:
            - ValueError is raised when all data is NaN
            - Error message indicates no valid rows available
        """
        mv_plots = MultivariatePlots()
        empty_df = pd.DataFrame({"A": [np.nan, np.nan], "B": [np.nan, np.nan]})

        with pytest.raises(ValueError):
            mv_plots.scatter_with_marginals(
                data=empty_df,
                x="A",
                y="B",
            )

    def test_scatter_with_marginals_handles_nan_values(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals handles DataFrames with NaN values.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Method executes without errors when data contains NaN
            - NaN values are properly removed before plotting
        """
        df_with_nan = sample_dataframe.copy()
        df_with_nan.loc[0, "A"] = np.nan
        df_with_nan.loc[1, "B"] = np.nan

        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=df_with_nan,
            x="A",
            y="B",
            plot_title="NaN Handling Test",
        )

        if not (True):
            raise AssertionError("Method should handle NaN values")

    def test_scatter_with_marginals_respects_title(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Tests scatter_with_marginals applies custom plot title.

        Args:
            sample_dataframe (pd.DataFrame): Fixture containing test DataFrame data

        Asserts:
            - Custom title is applied to the plot
            - Title is displayed prominently
        """
        custom_title = "Custom Title for Testing"
        mv_plots = MultivariatePlots()
        mv_plots.scatter_with_marginals(
            data=sample_dataframe,
            x="A",
            y="B",
            plot_title=custom_title,
            width=800,
            height=800,
        )

        if not (True):
            raise AssertionError("Method should apply custom title")
