import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from kvbiii_plots.base_plots import BasePlots


class MultivariatePlots(BasePlots):
    """Class for creating multivariate analysis visualizations.

    This class inherits from BasePlots and provides specialized methods
    for visualizing relationships between multiple variables including
    correlation heatmaps, scatter plot matrices, and parallel coordinates.
    """

    def correlation_plot(
        self,
        data: np.ndarray | pd.DataFrame,
        features_names: list[str] | None = None,
        method: str = "spearman",
        plot_title: str = "",
        width: int = 1500,
        height: int = 1500,
        show_upper: bool = False,
        colorscale: str = "RdBu",
    ) -> None:
        """
        Create a correlation heatmap for multivariate data analysis.

        Args:
            data (np.ndarray | pd.DataFrame): Input data as a 2D array or DataFrame.
            features_names (list[str] | None): List of feature names for axis labels. If None, uses
            DataFrame columns or generic names.
            method (str): Correlation method. One of 'spearman', 'pearson', or 'kendall'.
            plot_title (str): Plot title.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.
            show_upper (bool): If True, show upper triangle of correlation
                matrix. If False, mask upper triangle.
            colorscale (str): Plotly diverging color scale name.

        Returns:
            None
        """
        if isinstance(data, pd.DataFrame):
            if features_names is None:
                features_names = data.columns.tolist()
            data_df = data
        else:
            data = self.check_2d_data(data=data)
            if features_names is None:
                features_names = [f"Feature_{i}" for i in range(data.shape[1])]
            data_df = pd.DataFrame(data, columns=features_names)

        corr = np.round(data_df[features_names].corr(method=method), 3)

        if not show_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))
            data_mask = corr.mask(mask)
        else:
            data_mask = corr

        if hasattr(px.colors.diverging, colorscale):
            color_scale = getattr(px.colors.diverging, colorscale)
        else:
            print(
                f"Warning: colorscale '{colorscale}' not found. Using default 'RdBu'."
            )
            color_scale = px.colors.diverging.RdBu

        fig = ff.create_annotated_heatmap(
            z=data_mask.to_numpy(),
            x=data_mask.columns.tolist(),
            y=data_mask.columns.tolist(),
            colorscale=color_scale,
            hoverinfo="none",
            showscale=True,
            ygap=1,
            xgap=1,
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(
            width=width,
            height=height,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            yaxis_autorange="reversed",
            template="plotly_white",
            title=f"<b>{plot_title}<b>" if plot_title else "",
            title_x=0.5,
            font=dict(family="Times New Roman", size=26, color="Black"),
        )

        for _, annotation in enumerate(fig.layout.annotations):
            if annotation.text == "nan":
                annotation.text = ""

        fig.show("png", width=width, height=height)

    def matrix_plot(
        self,
        matrix: np.ndarray | pd.DataFrame,
        features_names: list[str] | None = None,
        plot_title: str = "",
        width: int = 1500,
        height: int = 1500,
        show_upper: bool = False,
        colorscale: str = "RdBu",
        vmin: float | None = None,
        vmax: float | None = None,
        max_label_length: int = 15,
        auto_font_size: bool = True,
    ) -> None:
        """
        Plot a heatmap for a precomputed correlation or association matrix.

        Args:
            matrix (np.ndarray | pd.DataFrame): Precomputed matrix
                (e.g., Cramér's V, correlation).
            features_names (list[str] | None): Feature names for axis labels. If None,
                uses DataFrame columns or generic names.
            plot_title (str): Plot title.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.
            show_upper (bool): If True, show upper triangle. If False, mask upper triangle.
            colorscale (str): Plotly diverging color scale name.
            vmin (float | None): Minimum value for color scale. If None, uses matrix min.
            vmax (float | None): Maximum value for color scale. If None, uses matrix max.
            max_label_length (int): Maximum length for feature labels before truncation.
            auto_font_size (bool): If True, automatically adjust font size
                based on matrix dimensions.

        Returns:
            None
        """
        if isinstance(matrix, pd.DataFrame):
            mat = matrix.copy()
            if features_names is None:
                features_names = mat.columns.tolist()
            mat = mat.loc[features_names, features_names]
        else:
            mat = np.asarray(matrix)
            if features_names is None:
                features_names = [f"Feature_{i}" for i in range(mat.shape[0])]
            mat = pd.DataFrame(mat, index=features_names, columns=features_names)

        original_names = mat.columns.tolist()
        shortened_names = []
        seen_names = {}

        for name in original_names:
            if len(name) <= max_label_length:
                short_name = name
            else:
                short_name = f"{name[:max_label_length-2]}.."

            if short_name in seen_names:
                seen_names[short_name] += 1
                short_name = f"{short_name[:-2]}_{seen_names[short_name]}"
            else:
                seen_names[short_name] = 0

            shortened_names.append(short_name)

        mat.columns = shortened_names
        mat.index = shortened_names

        if not show_upper:
            mask = np.triu(np.ones_like(mat, dtype=bool))
            mat_masked = mat.mask(mask)
        else:
            mat_masked = mat

        mat_masked_rounded = np.round(mat_masked.to_numpy(), 2)

        zmin = vmin if vmin is not None else np.nanmin(mat_masked_rounded)
        zmax = vmax if vmax is not None else np.nanmax(mat_masked_rounded)

        if auto_font_size:
            n_features = len(mat)
            font_size = max(8, min(26, int(1500 / n_features)))
            annotation_font_size = max(6, min(12, int(800 / n_features)))
        else:
            font_size = 26
            annotation_font_size = 10

        fig = ff.create_annotated_heatmap(
            z=mat_masked_rounded,
            x=mat_masked.columns.tolist(),
            y=mat_masked.index.tolist(),
            colorscale=colorscale,
            hoverinfo="none",
            showscale=True,
            ygap=1,
            xgap=1,
            zmin=zmin,
            zmax=zmax,
        )

        fig.update_xaxes(side="bottom")
        fig.update_layout(
            width=width,
            height=height,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis_zeroline=False,
            yaxis_zeroline=False,
            yaxis_autorange="reversed",
            template="plotly_white",
            title=f"<b>{plot_title}<b>" if plot_title else "",
            title_x=0.5,
            font=dict(family="Times New Roman", size=font_size, color="Black"),
        )

        for _, annotation in enumerate(fig.layout.annotations):
            if annotation.text == "nan":
                annotation.text = ""
            else:
                annotation.font.size = annotation_font_size

        fig.show("png", width=width, height=height)

    def scatter_matrix(
        self,
        data: pd.DataFrame,
        features: list[str] | None = None,
        hue: str | None = None,
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        marker_size: int = 5,
    ) -> None:
        """
        Create a scatter plot matrix for multivariate analysis.

        Args:
            data (pd.DataFrame): Input DataFrame with features.
            features (list[str] | None): List of feature columns to include. If None,
                uses up to 6 numeric columns.
            hue (str | None): Column name for color grouping. If None, no color grouping is used.
            plot_title (str): Plot title.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.
            marker_size (int): Marker size.

        Returns:
            None
        """
        if features is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols[: min(6, len(numeric_cols))]

        data_subset = data[features + ([hue] if hue else [])].dropna()

        if hue:
            fig = px.scatter_matrix(
                data_subset,
                dimensions=features,
                color=hue,
                title=plot_title,
                width=width,
                height=height,
            )
            fig.update_traces(marker=dict(size=marker_size))
        else:
            fig = px.scatter_matrix(
                data_subset,
                dimensions=features,
                title=plot_title,
                width=width,
                height=height,
            )
            fig.update_traces(
                marker=dict(size=marker_size, color=self.default_colors["primary"])
            )

        fig.update_layout(
            template="simple_white",
            title_x=0.5,
            font=dict(family="Times New Roman", size=16, color="Black"),
        )

        fig.show("png", width=width, height=height)

    def heatmap(
        self,
        data: pd.DataFrame,
        plot_title: str = "",
        width: int = 1000,
        height: int = 1000,
        xaxis_title: str = "",
        yaxis_title: str = "",
        highlights: list[tuple[str, str]] | None = None,
    ) -> go.Figure:
        """
        Plot a styled heatmap with optional highlight markers for imputed cells.

        Args:
            data (pd.DataFrame): 2D table with rows as x-axis categories and
                columns as y-axis categories.
            plot_title (str): Figure title.
            width (int): Figure width in pixels.
            height (int): Figure height in pixels.
            xaxis_title (str): X-axis title.
            yaxis_title (str): Y-axis title.
            highlights (list[tuple[str, str]] | None): List of (x_value, y_value) pairs to mark as
            imputation targets.

        Returns:
            go.Figure: Rendered Plotly figure.
        """
        values = np.asarray(data, dtype=float).T
        x_labels = list(map(str, data.index.tolist()))
        y_labels = list(map(str, data.columns.tolist()))

        text_vals = values.astype(int)
        text = [[str(v) if v > 0 else "" for v in row] for row in text_vals]

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=x_labels,
                y=y_labels,
                colorscale="Blues",
                colorbar=dict(title="Count", len=0.75),
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=26),
                hovertemplate="<b>%{y}</b> -> <b>%{x}</b><br>Count: %{z}<extra></extra>",
                ygap=1,
                xgap=1,
                zmin=0,
            )
        )

        if highlights:
            hx = []
            hy = []
            x_label_set = set(x_labels)
            y_label_set = set(y_labels)
            for xv, yv in highlights:
                xv_s, yv_s = str(xv), str(yv)
                if xv_s in x_label_set and yv_s in y_label_set:
                    hx.append(xv_s)
                    hy.append(yv_s)
            if hx:
                fig.add_trace(
                    go.Scatter(
                        x=hx,
                        y=hy,
                        mode="markers",
                        marker=dict(
                            symbol="x",
                            size=25,
                            color="crimson",
                            line=dict(width=2, color="white"),
                        ),
                        name="Imputation target",
                        hovertemplate="Impute: <b>%{y}</b> -> <b>%{x}</b><extra></extra>",
                    )
                )

        fig.update_layout(
            template="simple_white",
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            title=dict(text=f"<b>{plot_title}</b>", x=0.5),
            font=dict(family="Times New Roman", size=26, color="Black"),
            xaxis=dict(tickangle=45, tickfont=dict(size=26), showgrid=False),
            yaxis=dict(tickfont=dict(size=26), autorange="reversed", showgrid=False),
            margin=dict(l=60, r=20, t=60, b=100),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        fig.show("png", width=width, height=height)

    def compare_distributions_plot(
        self,
        df: pd.DataFrame,
        feature_name: str,
        imputed_col: str = "imputed",
        categorical_columns: list[str] | None = None,
        width: int = 1200,
        height: int = 800,
    ) -> None:
        """
        Plot the distribution of a feature before and after imputation.

        Args:
            df (pd.DataFrame): DataFrame containing the feature and imputed column.
            feature_name (str): Name of the feature to plot.
            imputed_col (str): Name of the column with imputed values.
            categorical_columns (list[str] | None): List of categorical columns. If None, uses self
            .categorical_columns if available.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.

        Returns:
            None
        """

        if categorical_columns is None:
            categorical_columns = getattr(self, "categorical_columns", [])
        is_categorical = feature_name in categorical_columns

        if is_categorical:
            original_counts = (
                df[feature_name].value_counts(normalize=True).sort_index() * 100
            )
            imputed_counts = (
                df[imputed_col].value_counts(normalize=True).sort_index() * 100
            )
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=original_counts.index,
                    y=original_counts.values,
                    name="Before imputation",
                    marker=dict(color=px.colors.qualitative.Pastel[0]),
                    text=[f"{v:.2f}%" for v in original_counts.values],
                    textposition="auto",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=imputed_counts.index,
                    y=imputed_counts.values,
                    name="After imputation",
                    marker=dict(color=px.colors.qualitative.Pastel[1]),
                    text=[f"{v:.2f}%" for v in imputed_counts.values],
                    textposition="auto",
                )
            )
            fig.update_layout(
                template="simple_white",
                width=width,
                height=height,
                title=f"<b>Distribution of '{feature_name}' before and after imputation<b>",
                title_x=0.5,
                font=dict(family="Times New Roman", size=22, color="Black"),
                xaxis_title="Feature value",
                yaxis_title="Percentage (%)",
                barmode="group",
            )
            fig.update_xaxes(type="category")
            fig.show("png", width=width, height=height)
        else:
            original_data = df[feature_name].dropna()
            imputed_data = df[imputed_col]
            kde_orig = gaussian_kde(original_data)
            kde_imputed = gaussian_kde(imputed_data)
            x_min = min(original_data.min(), imputed_data.min())
            x_max = max(original_data.max(), imputed_data.max())
            x_vals = np.linspace(x_min, x_max, 500)
            colors = px.colors.qualitative.Pastel
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=kde_orig(x_vals),
                    mode="lines",
                    name="Before imputation",
                    line=dict(color=colors[0], width=5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=kde_imputed(x_vals),
                    mode="lines",
                    name="After imputation",
                    line=dict(color=colors[1], width=5),
                )
            )
            fig.update_layout(
                template="simple_white",
                width=width,
                height=height,
                title=f"<b>Distribution of '{feature_name}' before and after imputation<b>",
                title_x=0.5,
                font=dict(family="Times New Roman", size=22, color="Black"),
                xaxis_title="Feature value",
                yaxis_title="Density",
            )
            fig.show("png", width=width, height=height)

    def parallel_coordinates(
        self,
        data: pd.DataFrame,
        features: list[str] | None = None,
        hue: str | None = None,
        plot_title: str = "",
        width: int = 1200,
        height: int = 600,
        normalize: bool = True,
    ) -> None:
        """
        Create a parallel coordinates plot for multivariate analysis.

        Args:
            data (pd.DataFrame): Input DataFrame with features.
            features (list[str] | None): List of feature columns to include.

                If None, uses all numeric columns.
            hue (str | None): Column name for color grouping. If None, no color grouping is used.
            plot_title (str): Plot title.
            width (int): Plot width in pixels.
            height (int): Plot height in pixels.
            normalize (bool): If True, normalize features to [0, 1] scale.

        Returns:
            None
        """
        if features is None:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols

        data_subset = data[features + ([hue] if hue else [])].dropna()

        if normalize:
            for feature in features:
                min_val = data_subset[feature].min()
                max_val = data_subset[feature].max()
                if max_val != min_val:
                    data_subset[feature] = (data_subset[feature] - min_val) / (
                        max_val - min_val
                    )

        if hue:
            if not is_numeric_dtype(data_subset[hue].dtype):
                color_codes, uniques = pd.factorize(data_subset[hue])
                data_subset["_hue_code"] = color_codes
                color_arg = "_hue_code"
            else:
                color_arg = hue
            fig = px.parallel_coordinates(
                data_subset,
                dimensions=features,
                color=color_arg,
                title=plot_title,
                width=width,
                height=height,
            )
            if not is_numeric_dtype(data_subset[hue].dtype):
                fig.update_coloraxes(
                    colorbar=dict(
                        tickvals=list(range(len(uniques))),
                        ticktext=[str(u) for u in uniques],
                    )
                )
        else:
            fig = px.parallel_coordinates(
                data_subset,
                dimensions=features,
                title=plot_title,
                width=width,
                height=height,
            )

        fig.update_layout(
            template="simple_white",
            title_x=0.5,
            font=dict(family="Times New Roman", size=16, color="Black"),
        )

        fig.show("png", width=width, height=height)

    def _coerce_vector_input(
        self,
        values: pd.Series | np.ndarray | list | tuple,
        label: str,
    ) -> pd.Series:
        """Convert supported vector-like inputs to a one-dimensional Series."""
        if isinstance(values, pd.Series):
            return values.reset_index(drop=True)

        if isinstance(values, np.ndarray):
            if values.ndim != 1:
                raise ValueError(f"{label} must be one-dimensional.")
            return pd.Series(values)

        if isinstance(values, (list, tuple)):
            if len(values) == 0:
                return pd.Series(dtype=float)

            has_nested_vectors = all(
                isinstance(item, (pd.Series, np.ndarray, list, tuple))
                for item in values
            )
            if has_nested_vectors:
                series_parts = [
                    self._coerce_vector_input(item, label) for item in values
                ]
                return pd.concat(series_parts, ignore_index=True)

            return pd.Series(values)

        raise TypeError(
            f"{label} must be a pandas Series, numpy array, list, or tuple."
        )

    def scatter_with_marginals(
        self,
        data: pd.DataFrame | None = None,
        x: str | None = None,
        y: str | None = None,
        X: pd.Series | np.ndarray | list | tuple | None = None,
        Y: pd.Series | np.ndarray | list | tuple | None = None,
        hue: str | pd.Series | np.ndarray | list | tuple | None = None,
        x_label: str = "",
        y_label: str = "",
        plot_title: str = "",
        width: int = 1000,
        height: int = 1000,
        marker_size: int = 8,
        marginal_height_ratio: float = 0.2,
        opacity: float = 0.7,
        font_size: int = 22,
        showlegend: bool = True,
    ) -> None:
        """
        Create a scatter plot with marginal distributions (histograms) on axes.

        Generates a joint plot with a central scatter plot and marginal histograms
        on the top and right axes. Optionally supports color grouping by a hue column.

        Args:
            data (pd.DataFrame | None): Input DataFrame containing the variables to plot.
                Use together with x and y column names.
            x (str | None): Column name for x-axis variable when data is provided.
            y (str | None): Column name for y-axis variable when data is provided.
            X (pd.Series | np.ndarray | list | tuple | None): Direct x-values input.
                Supports one vector or a list/tuple of vectors.
            Y (pd.Series | np.ndarray | list | tuple | None): Direct y-values input.
                Supports one vector or a list/tuple of vectors.
            hue (str | pd.Series | np.ndarray | list | tuple | None): Color grouping.
                Use a column name string when data is provided, or a direct vector when
                X and Y are provided.
            x_label (str): X-axis label override. Defaults to empty string.
            y_label (str): Y-axis label override. Defaults to empty string.
            plot_title (str): Plot title. Defaults to empty string.
            width (int): Plot width in pixels. Defaults to 1000.
            height (int): Plot height in pixels. Defaults to 1000.
            marker_size (int): Size of scatter plot markers. Defaults to 8.
            marginal_height_ratio (float): Ratio of marginal plot height relative to
                scatter plot. Defaults to 0.2.
            opacity (float): Opacity of scatter plot markers. Defaults to 0.7.
            font_size (int): Font size for plot title and axis labels. Defaults to 22.
            showlegend (bool): If True, display legend for hue groups. Defaults to True.

        Returns:
            None

        Note:
            O(n) time complexity where n is the number of rows in the data.
            Handles missing values by dropping them before plotting.
        """
        if data is not None:
            if not isinstance(x, str) or not isinstance(y, str):
                raise ValueError(
                    "When data is provided, x and y must be column name strings."
                )
            if x not in data.columns or y not in data.columns:
                missing = [col for col in [x, y] if col not in data.columns]
                raise ValueError(f"Columns {missing} not found in DataFrame.")

            x_column = x
            y_column = y
            hue_column = None
            hue_label = "hue"

            cols_to_select = [x_column, y_column]
            if hue is not None:
                if not isinstance(hue, str):
                    raise TypeError(
                        "When data is provided, hue must be a column name string."
                    )
                if hue not in data.columns:
                    raise ValueError(f"Column '{hue}' not found in DataFrame.")
                cols_to_select.append(hue)
                hue_column = hue
                hue_label = hue

            data_clean = data[cols_to_select].dropna().copy()
            x_axis_label = x_label if x_label else x_column
            y_axis_label = y_label if y_label else y_column
        else:
            if X is None or Y is None:
                raise ValueError(
                    "Provide either data with x and y column names or direct X and Y vectors."
                )

            x_series = self._coerce_vector_input(X, "X")
            y_series = self._coerce_vector_input(Y, "Y")
            if len(x_series) != len(y_series):
                raise ValueError("X and Y must have the same number of values.")

            x_column = "__x"
            y_column = "__y"
            data_clean = pd.DataFrame({x_column: x_series, y_column: y_series})
            x_axis_label = x_label if x_label else (x_series.name or "X")
            y_axis_label = y_label if y_label else (y_series.name or "Y")

            hue_column = None
            hue_label = "hue"
            if hue is not None:
                if isinstance(hue, str):
                    raise ValueError(
                        "When data is not provided, hue must be a vector-like input, not a string."
                    )
                hue_series = self._coerce_vector_input(hue, "hue")
                if len(hue_series) != len(data_clean):
                    raise ValueError(
                        "hue must have the same number of values as X and Y."
                    )
                hue_column = "__hue"
                hue_label = hue_series.name or "hue"
                data_clean[hue_column] = hue_series.to_numpy()

        if data_clean.empty:
            raise ValueError("Data contains no valid rows after removing NaN values.")

        fig = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
            column_widths=[1 - marginal_height_ratio, marginal_height_ratio],
            row_heights=[marginal_height_ratio, 1 - marginal_height_ratio],
            horizontal_spacing=0.02,
            vertical_spacing=0.02,
        )

        unique_groups = None
        group_colors = None

        if hue_column:
            if not is_numeric_dtype(data_clean[hue_column].dtype):
                unique_groups = data_clean[hue_column].unique()
                group_colors = (
                    px.colors.qualitative.Set1[: len(unique_groups)]
                    + px.colors.qualitative.Set2[: max(0, len(unique_groups) - 9)]
                )
            else:
                unique_groups = None

        if hue_column and unique_groups is not None:
            for idx, group in enumerate(unique_groups):
                group_data = data_clean[data_clean[hue_column] == group]
                color = group_colors[idx % len(group_colors)]

                fig.add_trace(
                    go.Scatter(
                        x=group_data[x_column],
                        y=group_data[y_column],
                        mode="markers",
                        name=str(group),
                        marker=dict(
                            size=marker_size,
                            color=color,
                            opacity=opacity,
                            line=dict(width=0.5, color="white"),
                        ),
                        hovertemplate=f"<b>{x_axis_label}</b>: %{{x}}<br><b>{y_axis_label}</b>: %{{y}}<br><b>{hue_label}</b>: {group}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )

                fig.add_trace(
                    go.Histogram(
                        x=group_data[x_column],
                        name=f"{group} (marginal)",
                        marker=dict(color=color, opacity=opacity),
                        showlegend=False,
                        hovertemplate="<b>Count</b>: %{y}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

                fig.add_trace(
                    go.Histogram(
                        y=group_data[y_column],
                        name=f"{group} (marginal)",
                        marker=dict(color=color, opacity=opacity),
                        showlegend=False,
                        hovertemplate="<b>Count</b>: %{x}<extra></extra>",
                    ),
                    row=2,
                    col=2,
                )
        elif hue_column and unique_groups is None:
            fig.add_trace(
                go.Scatter(
                    x=data_clean[x_column],
                    y=data_clean[y_column],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=data_clean[hue_column],
                        opacity=opacity,
                        colorscale="Viridis",
                        showscale=True,
                        line=dict(width=0.5, color="white"),
                        colorbar=dict(title=hue_label, x=1.12),
                    ),
                    name="Data",
                    hovertemplate=f"<b>{x_axis_label}</b>: %{{x}}<br><b>{y_axis_label}</b>: %{{y}}<br><b>{hue_label}</b>: %{{marker.color}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    x=data_clean[x_column],
                    name="x marginal",
                    marker=dict(color=self.default_colors["primary"], opacity=opacity),
                    showlegend=False,
                    hovertemplate="<b>Count</b>: %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    y=data_clean[y_column],
                    name="y marginal",
                    marker=dict(color=self.default_colors["primary"], opacity=opacity),
                    showlegend=False,
                    hovertemplate="<b>Count</b>: %{x}<extra></extra>",
                ),
                row=2,
                col=2,
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=data_clean[x_column],
                    y=data_clean[y_column],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=self.default_colors["primary"],
                        opacity=opacity,
                        line=dict(width=0.5, color="white"),
                    ),
                    name="Data",
                    hovertemplate=f"<b>{x_axis_label}</b>: %{{x}}<br><b>{y_axis_label}</b>: %{{y}}<extra></extra>",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    x=data_clean[x_column],
                    name="x marginal",
                    marker=dict(color=self.default_colors["primary"], opacity=opacity),
                    showlegend=False,
                    hovertemplate="<b>Count</b>: %{y}<extra></extra>",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Histogram(
                    y=data_clean[y_column],
                    name="y marginal",
                    marker=dict(color=self.default_colors["primary"], opacity=opacity),
                    showlegend=False,
                    hovertemplate="<b>Count</b>: %{x}<extra></extra>",
                ),
                row=2,
                col=2,
            )

        fig.update_xaxes(title_text=x_axis_label, row=2, col=1)
        fig.update_yaxes(title_text=y_axis_label, row=2, col=1)

        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, row=2, col=2)

        fig.update_layout(
            title=f"<b>{plot_title}</b>" if plot_title else "",
            title_x=0.5,
            width=width,
            height=height,
            template="simple_white",
            font=dict(family="Times New Roman", size=font_size, color="Black"),
            showlegend=showlegend,
            hovermode="closest",
            margin=dict(l=80, r=0, t=90, b=80),
        )
        fig.update_layout(
            legend=dict(
                x=0.92,
                y=0.85,
                xanchor="right",
                yanchor="bottom",
                bordercolor="black",
                borderwidth=1,
            )
        )

        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(0)
    df = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 50),
            "feature2": np.random.normal(5, 2, 50),
            "feature3": np.random.normal(-2, 0.5, 50),
            "category1": np.random.choice(["A", "B"], 50),
            "category2": np.random.choice(["X", "Y"], 50),
        }
    )

    plots = MultivariatePlots()
    plots.correlation_plot(
        df,
        features_names=["feature1", "feature2", "feature3"],
        plot_title="Correlation Example",
        width=600,
        height=600,
    )
    plots.scatter_matrix(
        df,
        features=["feature1", "feature2"],
        hue="category1",
        plot_title="Scatter Matrix Example",
        width=600,
        height=600,
    )
    plots.parallel_coordinates(
        df,
        features=["feature1", "feature2", "feature3"],
        hue="category1",
        plot_title="Parallel Coordinates Example",
        width=800,
        height=400,
    )
    plots.heatmap(
        df[["category1", "category2"]],
        plot_title="Heatmap Example",
        width=800,
        height=600,
        xaxis_title="Features",
        yaxis_title="Categories",
    )
    plots.compare_distributions_plot(
        df,
        feature_name="feature1",
        imputed_col="feature2",
        categorical_columns=["category1", "category2"],
        width=800,
        height=600,
    )
    plots.scatter_with_marginals(
        df,
        x="feature1",
        y="feature2",
        hue="category1",
        plot_title="Scatter with Marginal Distributions Example",
        width=900,
        height=900,
        marker_size=6,
    )
    plots.scatter_with_marginals(
        X=[
            df.loc[df["category1"] == "A", "feature1"],
            df.loc[df["category1"] == "B", "feature1"],
        ],
        Y=[
            df.loc[df["category1"] == "A", "feature2"],
            df.loc[df["category1"] == "B", "feature2"],
        ],
        hue=[
            pd.Series(["A"] * (df["category1"] == "A").sum()),
            pd.Series(["B"] * (df["category1"] == "B").sum()),
        ],
        x_label="feature1",
        y_label="feature2",
        plot_title="Scatter with Marginal Distributions (Direct Vectors)",
        width=900,
        height=900,
        marker_size=6,
    )
