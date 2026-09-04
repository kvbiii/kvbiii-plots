import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

from kvbiii_plots.base_plots import BasePlots


class ContinuousPlots(BasePlots):
    """
    Class for creating plots specifically for continuous variables.

    This class inherits from BasePlots and provides specialized methods
    for visualizing continuous data including histograms, box plots,
    and regression analysis.
    """

    def _calculate_bin_size(self, data: np.ndarray) -> float:
        """
        Calculate bin size using the Freedman-Diaconis rule.

        Args:
            data (np.ndarray): Input data array.

        Returns:
            float: Calculated bin size.
        """
        clean_data = data[~np.isnan(data)]
        if len(clean_data) < 2:
            return 1.0
        q75, q25 = np.percentile(clean_data, [75, 25])
        iqr = q75 - q25
        bin_size = (
            2 * (np.std(clean_data) if iqr == 0 else iqr) / (len(clean_data) ** (1 / 3))
        )
        data_range = np.max(clean_data) - np.min(clean_data)
        return max(data_range / 100, min(bin_size, data_range / 5))

    def _validate_distribution_data(
        self,
        data: dict[str, pd.Series | np.ndarray | list[object]],
    ) -> dict[str, np.ndarray]:
        """
        Validates and cleans grouped continuous data for distribution comparison.

        Args:
            data (dict[str, pd.Series | np.ndarray | list]): Mapping of group labels
            to their continuous values.

        Returns:
            dict[str, np.ndarray]: Mapping of group labels to cleaned, non-empty
            NumPy arrays.

        Raises:
            TypeError: If data is not a dict.
            ValueError: If fewer than two non-empty groups remain after cleaning.
        """
        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dict mapping group labels to their continuous values."
            )
        cleaned_data = {
            str(group_label): self.check_data(group_values)
            for group_label, group_values in data.items()
        }
        cleaned_data = {
            group_label: group_values
            for group_label, group_values in cleaned_data.items()
            if group_values.size > 0
        }
        if len(cleaned_data) < 2:
            raise ValueError(
                "compare_distributions_plot requires at least two non-empty groups."
            )
        return cleaned_data

    def _compute_shared_bins(
        self,
        cleaned_data: dict[str, np.ndarray],
        bin_size: float | None,
    ) -> tuple[float, float, float]:
        """
        Computes shared histogram bin boundaries across all groups.

        Args:
            cleaned_data (dict[str, np.ndarray]): Mapping of group labels to cleaned
            arrays.
            bin_size (float | None): Bin width to use. If None, calculated from the
            pooled data using the Freedman-Diaconis rule.

        Returns:
            tuple[float, float, float]: Shared (start, end, size) for histogram bins.
        """
        pooled_data = np.concatenate(list(cleaned_data.values()))
        if bin_size is None:
            bin_size = self._calculate_bin_size(pooled_data)
        return float(np.min(pooled_data)), float(np.max(pooled_data)), float(bin_size)

    def compare_distributions_plot(
        self,
        data: dict[str, pd.Series | np.ndarray | list[object]],
        plot_title: str = "",
        width: int = 1200,
        height: int = 800,
        bin_size: float | None = None,
        xaxis_title: str = "Value",
        yaxis_title: str = "Density",
        alpha: float = 0.55,
        show_legend: bool = True,
    ) -> None:
        """
        Creates an overlaid, density-normalized histogram comparing continuous
        distributions across named groups of any size.

        Each group's histogram is normalized to a probability density (the area
        under its curve sums to one) and all groups share the same bin edges. This
        keeps the comparison based on each group's actual distribution shape rather
        than its raw observation count, so a group of 10 000 values and a group of
        100 values remain equally visible and comparable.

        Args:
            data (dict[str, pd.Series | np.ndarray | list]): Mapping of group labels
            to their continuous values (e.g. model probabilities, predictions, or any
            numeric feature) to compare.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1200.
            height (int, optional): Height of the plot. Defaults to 800.
            bin_size (float | None, optional): Shared bin width. If None,
            automatically calculated from the pooled data using the
            Freedman-Diaconis rule. Defaults to None.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Value".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Density".
            alpha (float, optional): Opacity for overlapping histograms (0.0 to 1.0).
            Defaults to 0.55.
            show_legend (bool, optional): Whether to show the legend. Defaults to
            True.

        Returns:
            None

        Raises:
            TypeError: If data is not a dict.
            ValueError: If fewer than two non-empty groups are provided.
        """
        cleaned_data = self._validate_distribution_data(data)
        start, end, size = self._compute_shared_bins(cleaned_data, bin_size)
        colors = self._get_colors(len(cleaned_data))

        fig = go.Figure()
        for idx, (group_label, group_values) in enumerate(cleaned_data.items()):
            fig.add_trace(
                go.Histogram(
                    x=group_values,
                    name=group_label,
                    histnorm="probability density",
                    xbins=dict(start=start, end=end, size=size),
                    marker=dict(color=colors[idx % len(colors)], opacity=alpha),
                    showlegend=show_legend,
                )
            )
        fig.update_layout(barmode="overlay")
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        if show_legend:
            fig.update_layout(
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                )
            )
        fig.show("png", width=width, height=height)

    def scatter_plot(
        self,
        data: np.ndarray,
        hue: np.ndarray | list[object],
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "Component 1",
        yaxis_title: str = "Component 2",
        marker_size: int = 8,
        show_legend: bool = True,
    ) -> None:
        """
        Create a scatter plot for 2D data colored by hue labels.

        Args:
            data (np.ndarray): 2D data of shape (n_samples, 2).
            hue (np.ndarray | list): Labels for coloring the points.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1200.
            height (int, optional): Height of the plot. Defaults to 1200.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Component 1".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Component 2".
            marker_size (int, optional): Size of the markers. Defaults to 8.
            show_legend (bool, optional): Whether to show the legend. Defaults to True.

        Returns:
            None
        """
        data = self.check_2d_data(data=data)
        hue_arr = np.array(hue).squeeze()
        fig = go.Figure()
        unique_hue, freq = np.unique(hue_arr, return_counts=True)
        hue_freq = dict(
            sorted(
                dict(zip(unique_hue, freq)).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        colors = px.colors.qualitative.Pastel
        for idx, hue_value in enumerate(hue_freq):
            fig.add_trace(
                go.Scatter(
                    x=data[hue_arr == hue_value, 0],
                    y=data[hue_arr == hue_value, 1],
                    mode="markers",
                    marker=dict(
                        color=colors[idx % len(colors)],
                        showscale=False,
                        size=marker_size,
                    ),
                    showlegend=show_legend,
                    name=str(hue_value),
                )
            )
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)
        fig.update_layout(
            template="simple_white",
            width=width,
            height=height,
            title=f"<b>{plot_title}<b>",
            title_x=0.5,
            font=dict(family="Times New Roman", size=26, color="Black"),
        )
        fig.show("png", width=width, height=height)

    def line_plot(
        self,
        x: np.ndarray | pd.Series | list[object],
        y: np.ndarray | pd.Series | list[object],
        plot_title: str = "",
        width: int = 1200,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "",
    ) -> None:
        """
        Create a line plot for continuous data.

        Args:
            x (np.ndarray | pd.Series | list): X-axis data.
            y (np.ndarray | pd.Series | list): Y-axis data.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1200.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "".

        Returns:
            None
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                marker=dict(
                    color=self.default_colors["primary"],
                    size=10,
                    symbol="diamond",
                    line=dict(width=2, color=self.default_colors["accent"]),
                ),
                line=dict(
                    color=self.default_colors["primary"], width=4, dash="dashdot"
                ),
                name="Line Plot",
                showlegend=False,
            )
        )
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(title_text=xaxis_title)
        fig.update_yaxes(title_text=yaxis_title)
        fig.show("png", width=width, height=height)

    def histogram_plot(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray | list[object],
        hue: np.ndarray | pd.Series | list[object] | None = None,
        plot_title: str = "",
        width: int = 1200,
        height: int = 800,
        bin_size: float | None = None,
        xaxis_title: str = "",
        yaxis_title: str = "Frequency",
        alpha: float = 0.7,
        show_legend: bool = True,
    ) -> None:
        """
        Create a histogram plot with optional hue-based grouping using alpha transparency.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | list): Input continuous data.
            hue (np.ndarray | pd.Series | list | None, optional): Labels for grouping data.
            Defaults to None.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1200.
            height (int, optional): Height of the plot. Defaults to 800.
            bin_size (float | None, optional): Size of histogram bins. If None, auto-calculated.
            Defaults to None.
            xaxis_title (str, optional): Title for the x-axis.
            Defaults to data name if available, otherwise "".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Frequency".
            alpha (float, optional): Transparency for overlapping histograms (0.0 to 1.0).
            Defaults to 0.7.
            show_legend (bool, optional): Whether to show the legend. Defaults to True.

        Returns:
            None
        """
        default_title = (
            str(data.name) if hasattr(data, "name") and data.name else "Value"
        )
        checked_data = self.check_data(data=data)
        if not xaxis_title:
            xaxis_title = default_title
        fig = go.Figure()
        if hue is None:
            if bin_size is None:
                bin_size = self._calculate_bin_size(np.array(checked_data))
            fig.add_trace(
                go.Histogram(
                    x=checked_data,
                    marker=dict(color=self.default_colors["primary"], opacity=alpha),
                    showlegend=False,
                    xbins=dict(size=bin_size),
                    name="Distribution",
                )
            )
        else:
            hue_arr = np.array(hue).squeeze()
            if bin_size is None:
                bin_size = self._calculate_bin_size(np.array(checked_data))
            unique_hue, freq = np.unique(hue_arr, return_counts=True)
            hue_freq = dict(
                sorted(
                    dict(zip(unique_hue, freq)).items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
            n_colors = len(unique_hue)
            colors = (
                px.colors.sample_colorscale(
                    "rainbow", list(np.linspace(0, 1, n_colors))
                )
                if n_colors > 10
                else px.colors.qualitative.Plotly
            )
            for idx, hue_value in enumerate(hue_freq):
                mask = hue_arr == hue_value
                group_data = np.array(checked_data)[mask]
                if len(group_data) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=group_data,
                            marker=dict(color=colors[idx % len(colors)], opacity=alpha),
                            showlegend=show_legend,
                            xbins=dict(size=bin_size),
                            name=str(hue_value),
                        )
                    )
            fig.update_layout(barmode="overlay")
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_yaxes(title_text=yaxis_title)
        if show_legend and hue is not None:
            fig.update_layout(
                legend=dict(
                    x=0.99,
                    y=0.99,
                    xanchor="right",
                    yanchor="top",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                )
            )
        fig.show("png", width=width, height=height)

    def histogram_and_box_plot(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray | list[object],
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        annotations: bool | list[str] | None = None,
        bin_size: float | None = None,
        xaxis_title: str = "",
        yaxis_title: str = "",
    ) -> None:
        """
        Creates a combined histogram and box plot for continuous data analysis.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | list): Input continuous data.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            annotations (bool | list[str] | None, optional): Whether to show quantile annotations
            or specific quantiles.
            Defaults to None.
            bin_size (float | None, optional): Size of histogram bins. If None, automatically
            calculated using Freedman-Diaconis rule.
            Defaults to None.
            xaxis_title (str, optional): Title for the x-axis.
            Defaults to data name if available, otherwise "".
            yaxis_title (str, optional): Title for the y-axis.
            Defaults to data name if available, otherwise "".

        Returns:
            None
        """
        default_title = (
            str(data.name) if hasattr(data, "name") and data.name else "Value"
        )
        checked_data = self.check_data(data=data)
        if bin_size is None:
            bin_size = self._calculate_bin_size(np.array(checked_data))
        if not xaxis_title:
            xaxis_title = default_title
        if not yaxis_title:
            yaxis_title = default_title
        fig = self.create_subplot_layout(1, 2, [["box", "histogram"]])
        fig.add_trace(
            go.Box(
                y=checked_data,
                name="",
                marker=dict(color=self.default_colors["primary"]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=checked_data,
                marker=dict(color=self.default_colors["primary"]),
                showlegend=False,
                xbins=dict(size=bin_size),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text=xaxis_title, row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        self.add_quantile_annotations(fig, checked_data, annotations)
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(
            row=1,
            col=1,
            showticklabels=False,
            tickvals=[],
            ticktext=[],
            title="",
        )
        fig.show("png", width=width, height=height)

    def histogram_boxplot_linear_regression(
        self,
        data: pd.DataFrame,
        feature: str,
        target: str,
        plot_title: str = "",
        width: int = 2000,
        height: int = 800,
        annotations: bool | list[str] | None = None,
        bin_size: float | None = None,
        xaxis_title: str = "",
        yaxis_title: str = "",
        show_correlation: bool = True,
    ) -> None:
        """
        Creates a comprehensive plot with histogram, box plot, and linear regression analysis.

        Args:
            data (pd.DataFrame): Input DataFrame containing the feature and target variables.
            feature (str): Name of the feature column for analysis.
            target (str): Name of the target column for regression.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 2000.
            height (int, optional): Height of the plot. Defaults to 800.
            annotations (bool | list[str] | None, optional): Whether to show quantile annotations.
            Defaults to None.
            bin_size (float | None, optional): Size of histogram bins. If None, automatically
            calculated using Freedman-Diaconis rule.
            Defaults to None.
            xaxis_title (str, optional): Title for the x-axis. Defaults to feature name.
            yaxis_title (str, optional): Title for the y-axis. Defaults to target name.
            show_correlation (bool, optional): Whether to show correlation annotation.
            Defaults to True.

        Returns:
            None
        """
        no_nan_indices = self.filter_nan_indices(data, feature)
        clean_df = data[no_nan_indices].copy()
        if bin_size is None:
            bin_size = self._calculate_bin_size(clean_df[feature].values)
        if not xaxis_title:
            xaxis_title = feature
        if not yaxis_title:
            yaxis_title = target
        model = LinearRegression()
        X = clean_df[feature].values.reshape(-1, 1)
        y = clean_df[target].values
        model.fit(X, y)
        predictions = model.predict(X)
        fig = self.create_subplot_layout(1, 3, [["box", "histogram", "scatter"]])
        fig.add_trace(
            go.Box(
                y=clean_df[feature],
                name="",
                marker=dict(color=self.default_colors["primary"]),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=clean_df[feature],
                marker=dict(color=self.default_colors["primary"]),
                showlegend=False,
                xbins=dict(size=bin_size),
            ),
            row=1,
            col=2,
        )
        fig.update_xaxes(title_text=xaxis_title, row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.add_trace(
            go.Scatter(
                x=clean_df[feature],
                y=clean_df[target],
                mode="markers",
                marker=dict(color=self.default_colors["secondary"]),
                name="Real values",
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=clean_df[feature],
                y=predictions,
                mode="lines",
                line_color=self.default_colors["accent"],
                name=feature,
                showlegend=False,
            ),
            row=1,
            col=3,
        )
        fig.update_xaxes(title_text=xaxis_title, row=1, col=3)
        fig.update_yaxes(title_text=yaxis_title, row=1, col=3)
        y_min = clean_df[target].min() - 0.01 * abs(clean_df[target].min())
        y_max = clean_df[target].max() + 0.01 * abs(clean_df[target].max())
        fig.update_yaxes(range=[y_min, y_max], row=1, col=3)
        if show_correlation:
            spearman_corr = clean_df[feature].corr(clean_df[target], method="spearman")
            fig.add_annotation(
                x=0.99,
                y=0.99,
                xref="paper",
                yref="paper",
                text=f"Spearman corr: {spearman_corr:.3f}",
                showarrow=False,
                font=dict(size=24),
                align="right",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
            )
        self.add_quantile_annotations(fig, clean_df[feature], annotations)
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(
            row=1,
            col=1,
            showticklabels=False,
            tickvals=[],
            ticktext=[],
            title="",
        )
        fig.show("png", width=width, height=height)

    def boxplot_histogram_boxplot_by_hue(
        self,
        data: pd.DataFrame,
        feature: str,
        hue: str,
        plot_title: str = "",
        width: int = 2000,
        height: int = 1000,
        annotations: bool | list[str] | None = None,
        bin_size: float | None = None,
        xaxis_title: str = "",
        yaxis_title: str = "",
    ) -> None:
        """
        Creates a comprehensive plot with box plot, histogram, and grouped box plots by hue.

        Args:
            data (pd.DataFrame): Input DataFrame containing the feature and hue columns.
            feature (str): Name of the continuous feature column.
            hue (str): Name of the categorical column for grouping.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 2000.
            height (int, optional): Height of the plot. Defaults to 1000.
            annotations (bool | list[str] | None, optional): Whether to show quantile annotations.
            Defaults to None.
            bin_size (float | None, optional): Size of histogram bins. If None, automatically
            calculated using Freedman-Diaconis rule.
            Defaults to None.
            xaxis_title (str, optional): Title for the x-axis. Defaults to feature/hue names.
            yaxis_title (str, optional): Title for the y-axis. Defaults to feature name.

        Returns:
            None
        """
        data_copy = data[[feature, hue]].copy()
        data_copy.dropna(inplace=True)
        data_copy.reset_index(drop=True, inplace=True)
        if bin_size is None:
            bin_size = self._calculate_bin_size(data_copy[feature].values)
        if not yaxis_title:
            yaxis_title = feature
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "box"}, {"type": "histogram"}, {"type": "box"}]],
        )
        fig.add_trace(
            go.Box(
                y=data_copy[feature],
                name="",
                marker=dict(color="rgb(48,70,116)"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=data_copy[feature],
                marker=dict(color="rgb(48,70,116)"),
                showlegend=False,
                xbins=dict(size=bin_size),
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text=yaxis_title, row=1, col=1)
        fig.update_xaxes(title_text=yaxis_title, row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text=yaxis_title, row=1, col=3)
        fig.update_xaxes(
            title_text=hue if not xaxis_title else xaxis_title, row=1, col=3
        )
        labels = np.array(data_copy[hue].value_counts().index)
        frequency = np.array(data_copy[hue].value_counts().values)
        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        n_colors = len(labels)
        colors = (
            px.colors.sample_colorscale("rainbow", list(np.linspace(0, 1, n_colors)))
            if n_colors > 10
            else px.colors.qualitative.Pastel
        )
        for color_idx, category in enumerate(labels):
            indices = np.where(data_copy[hue] == category)[0]
            grouped_data = data_copy[feature].iloc[indices].tolist()
            fig.add_trace(
                go.Box(
                    y=grouped_data,
                    name=str(category),
                    marker=dict(color=colors[color_idx]),
                    showlegend=False,
                ),
                row=1,
                col=3,
            )
        self.add_quantile_annotations(fig, data_copy[feature].values, annotations)
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(
            row=1,
            col=1,
            showticklabels=False,
            tickvals=[],
            ticktext=[],
            title="",
        )
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(17)
    df = pd.DataFrame(
        {
            "feature": np.random.normal(50, 15, 100),
            "target": np.random.normal(100, 20, 100),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )

    plots = ContinuousPlots()

    plots.histogram_plot(
        data=df["feature"],
        plot_title="Feature Distribution",
        width=1000,
        height=600,
        xaxis_title="Feature Value",
        yaxis_title="Frequency",
    )

    plots.histogram_plot(
        data=df["feature"],
        hue=df["category"],
        plot_title="Feature Distribution by Category",
        width=1000,
        height=600,
        xaxis_title="Feature Value",
        yaxis_title="Frequency",
        alpha=0.6,
        show_legend=True,
    )

    plots.compare_distributions_plot(
        data={
            "Train (10k obs)": np.random.normal(50, 15, 10000),
            "Test (100 obs)": np.random.normal(55, 18, 100),
        },
        plot_title="Feature Distribution: Train vs Test",
        xaxis_title="Feature Value",
    )

    plots.scatter_plot(
        data=df[["feature", "target"]].values,
        hue=df["category"],
        plot_title="Scatter Plot of Feature vs Target by Category",
        width=800,
        height=800,
        xaxis_title="Feature",
        yaxis_title="Target",
        marker_size=10,
        show_legend=True,
    )

    plots.histogram_and_box_plot(
        data=df["feature"],
        annotations=True,
        bin_size=None,
        plot_title="Histogram and Box Plot of Feature",
        width=1200,
        height=600,
        xaxis_title="Feature",
        yaxis_title="Count",
    )

    plots.histogram_boxplot_linear_regression(
        data=df,
        feature="feature",
        target="target",
        annotations=True,
        bin_size=None,
        plot_title="Feature Distribution and Linear Regression",
        width=1800,
        height=600,
        xaxis_title="Feature",
        yaxis_title="Target",
        show_correlation=True,
    )

    plots.boxplot_histogram_boxplot_by_hue(
        data=df,
        feature="feature",
        hue="category",
        plot_title="Feature Distribution by Category",
        annotations=True,
        bin_size=None,
        width=1800,
        height=800,
        xaxis_title="Category",
        yaxis_title="Feature",
    )
