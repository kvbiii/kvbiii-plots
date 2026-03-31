
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ..base_plots import BasePlots


class SHAPPlots(BasePlots):
    """Class for creating SHAP (SHapley Additive exPlanations) visualization plots.

    This class inherits from BasePlots and provides specialized methods
    for visualizing SHAP values including feature importance, beeswarm plots,
    categorical analysis, and force plots for model interpretability.
    """

    def _shorten_feature_names(
        self, feature_names: np.ndarray | list[str], max_length: int = 15
    ) -> list[str]:
        """Shortens feature names with collision handling.

        Args:
            feature_names (np.ndarray | list[str]): Original feature names.
            max_length (int, optional): Maximum length for shortened names. Defaults to 15.

        Returns:
            list[str]: List of shortened feature names.
        """
        shortened_names = []
        seen_names = {}

        for name in feature_names:
            name_str = str(name)
            short_name = (
                name_str
                if len(name_str) <= max_length
                else f"{name_str[:max_length-2]}.."
            )

            if short_name in seen_names:
                seen_names[short_name] += 1
                short_name = f"{short_name[:-2]}_{seen_names[short_name]}"
            else:
                seen_names[short_name] = 0

            shortened_names.append(short_name)

        return shortened_names

    def _compute_shap_importance(
        self, shap_values: shap.Explanation, top_n: int = 20, class_id: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes feature importance based on mean absolute SHAP values.

        Args:
            shap_values (shap.Explanation): SHAP explanation object.
            top_n (int, optional): Maximum number of features to display. Defaults to 20.
            class_id (int, optional): Class index for classification tasks. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Top SHAP values,
            feature names, and importance scores.
        """
        values = shap_values.values
        feature_names = shap_values.feature_names or [
            f"feature_{i}" for i in range(values.shape[1])
        ]
        if values.ndim == 3:
            values = values[:, :, class_id]
            shap_values = shap_values[:, :, class_id]
        mean_abs_shap = np.abs(values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
        return (
            shap_values[:, top_idx],
            np.array(feature_names)[top_idx],
            mean_abs_shap[top_idx],
        )

    def _get_dynamic_colors(
        self,
        n_features: int,
        color_scale: str = "rainbow",
        use_qualitative: bool | None = None,
    ) -> list[str]:
        """Generates colors for SHAP visualizations based on feature count.

        Args:
            n_features (int): Number of features to color.
            color_scale (str, optional): Plotly color scale name. Defaults to "rainbow".
            use_qualitative (bool | None, optional): Whether to use qualitative colors.
            Defaults to None.

        Returns:
            list[str]: List of color values.
        """
        if use_qualitative is None:
            use_qualitative = n_features <= 10

        if use_qualitative:
            return px.colors.qualitative.Pastel[:n_features]
        else:
            return px.colors.sample_colorscale(
                color_scale,
                [
                    n / (n_features - 1) if n_features > 1 else 0
                    for n in range(n_features)
                ],
            )

    def _clean_data_for_plotting(
        self, data: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Removes NaN values from data and corresponding SHAP values.

        Args:
            data (np.ndarray): Feature data values.
            values (np.ndarray): SHAP values.

        Returns:
            tuple[np.ndarray, np.ndarray]: Cleaned data and SHAP values.
        """
        mask = ~np.isnan(data.astype(float)) & ~np.isnan(values.astype(float))
        return np.array(data)[mask].astype(float), np.array(values)[mask].astype(float)

    def _setup_matplotlib_styling(
        self,
        xlabel: str = "",
        ylabel: str = "",
        font_size: int = 20,
        font_name: str = "Times New Roman",
        font_color: str = "Black",
    ) -> None:
        """Applies consistent styling to matplotlib plots.

        Args:
            xlabel (str, optional): X-axis label. Defaults to "".
            ylabel (str, optional): Y-axis label. Defaults to "".
            font_size (int, optional): Font size for labels and ticks. Defaults to 20.
            font_name (str, optional): Font family name. Defaults to "Times New Roman".
            font_color (str, optional): Font color. Defaults to "Black".
        """
        if xlabel:
            plt.gca().set_xlabel(
                xlabel,
                fontdict={
                    "family": font_name,
                    "size": font_size + 12,
                    "color": font_color,
                },
            )
        if ylabel:
            plt.gca().set_ylabel(
                ylabel,
                fontdict={
                    "family": font_name,
                    "size": font_size + 12,
                    "color": font_color,
                },
            )
        plt.yticks(fontsize=font_size, fontname=font_name, color=font_color)
        plt.xticks(fontsize=font_size, fontname=font_name, color=font_color)

    def _create_custom_colorbar(
        self,
        ax: Axes,
        colormap: str = "coolwarm",
        label: str = "Feature value",
        high_label: str = "High",
        low_label: str = "Low",
        font_size: int = 20,
        font_name: str = "Times New Roman",
        font_color: str = "Black",
    ) -> None:
        """Creates a styled colorbar for SHAP plots.

        Args:
            ax (Axes): Matplotlib axes object.
            colormap (str, optional): Matplotlib colormap name. Defaults to "coolwarm".
            label (str, optional): Colorbar label. Defaults to "Feature value".
            high_label (str, optional): Label for high values. Defaults to "High".
            low_label (str, optional): Label for low values. Defaults to "Low".
            font_size (int, optional): Font size for colorbar. Defaults to 20.
            font_name (str, optional): Font family name. Defaults to "Times New Roman".
            font_color (str, optional): Font color. Defaults to "Black".
        """
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cbar = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(colormap)),
            orientation="vertical",
            ax=ax,
        )
        cbar.set_label(
            label,
            fontdict={"family": font_name, "size": font_size + 12, "color": font_color},
        )
        cbar.ax.set_yticklabels([])
        cbar.ax.set_yticks([])

        cbar.ax.text(
            0.5,
            1.01,
            high_label,
            transform=cbar.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=font_size,
            fontname=font_name,
            color=font_color,
        )
        cbar.ax.text(
            0.5,
            -0.01,
            low_label,
            transform=cbar.ax.transAxes,
            ha="center",
            va="top",
            fontsize=font_size,
            fontname=font_name,
            color=font_color,
        )
        cbar.ax.tick_params(labelsize=font_size, labelcolor=font_color)

    def plot_shap_bar(
        self,
        shap_values: shap.Explanation,
        top_n: int = 20,
        class_id: int = 0,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Features",
        yaxis_title: str = "Mean(|SHAP value|)",
        color_scale: str = "rainbow",
        use_qualitative_colors: bool | None = None,
        show_values: bool = True,
        value_precision: int = 3,
        font_size: int = 26,
        bar_line_color: str = "black",
        bar_line_width: int = 1,
        max_feature_name_length: int = 15,
    ) -> None:
        """Creates a bar plot showing SHAP feature importance.

        Args:
            shap_values (shap.Explanation): SHAP explanation object
            containing values and feature names.
            top_n (int, optional): Maximum number of features to display. Defaults to 20.
            class_id (int, optional): Class index for classification tasks. Defaults to 0.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot in pixels. Defaults to 1600.
            height (int, optional): Height of the plot in pixels. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "Features".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "Mean(|SHAP value|)".
            color_scale (str, optional): Plotly color scale name for continuous coloring.
            Defaults to "rainbow".
            use_qualitative_colors (bool | None, optional): Whether to use
            qualitative colors. Defaults to None.
            show_values (bool, optional): Whether to display values on bars. Defaults to True.
            value_precision (int, optional): Decimal precision for displayed values. Defaults to 3.
            font_size (int, optional): Font size for text elements. Defaults to 26.
            bar_line_color (str, optional): Color of bar outlines. Defaults to "black".
            bar_line_width (int, optional): Width of bar outlines. Defaults to 1.
            max_feature_name_length (int, optional): Maximum length for feature names.
            Defaults to 15.
        """
        _, top_features, top_shap_mean_values = self._compute_shap_importance(
            shap_values, top_n, class_id=class_id
        )

        shortened_features = self._shorten_feature_names(
            top_features, max_feature_name_length
        )

        colors = self._get_dynamic_colors(
            len(top_features), color_scale, use_qualitative_colors
        )

        if not plot_title:
            plot_title = "SHAP Feature Importance"

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=shortened_features,
                y=top_shap_mean_values,
                marker=dict(
                    line=dict(color=bar_line_color, width=bar_line_width), color=colors
                ),
                showlegend=False,
                text=(
                    [f"{v:.{value_precision}f}" for v in top_shap_mean_values]
                    if show_values
                    else None
                ),
                textposition="auto" if show_values else None,
            )
        )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )

        fig.update_layout(
            font=dict(family="Times New Roman", size=font_size, color="Black"),
        )

        fig.show("png", width=width, height=height)

    def plot_custom_shap_beeswarm(
        self,
        shap_values: shap.Explanation,
        top_n: int = 15,
        class_id: int = 0,
        plot_size: tuple[float, float] = (20, 15),
        xlabel: str = "SHAP value",
        ylabel: str = "",
        plot_title: str = "",
        colormap: str = "coolwarm",
        colorbar_label: str = "Feature value",
        high_label: str = "High",
        low_label: str = "Low",
        font_size: int = 20,
        font_name: str = "Times New Roman",
        font_color: str = "Black",
        show_colorbar: bool = True,
        max_feature_name_length: int = 15,
    ) -> None:
        """Plots a customized SHAP beeswarm plot for top features with styled colorbar and labels.

        Args:
            shap_values (shap.Explanation): SHAP explanation object
            containing values and feature names.
            top_n (int, optional): Number of top features to display. Defaults to 15.
            class_id (int, optional): Class index for classification tasks. Defaults to 0.
            plot_size (tuple[float, float], optional): Figure size as (width, height).
            Defaults to (20, 15).
            xlabel (str, optional): X-axis label. Defaults to "SHAP value".
            ylabel (str, optional): Y-axis label. Defaults to "".
            plot_title (str, optional): Title for the plot. Defaults to "".
            colormap (str, optional): Matplotlib colormap name. Defaults to "coolwarm".
            colorbar_label (str, optional): Label for the colorbar. Defaults to "Feature value".
            high_label (str, optional): Label for high feature values. Defaults to "High".
            low_label (str, optional): Label for low feature values. Defaults to "Low".
            font_size (int, optional): Base font size for text elements. Defaults to 20.
            font_name (str, optional): Font family name. Defaults to "Times New Roman".
            font_color (str, optional): Font color. Defaults to "Black".
            show_colorbar (bool, optional): Whether to show the colorbar. Defaults to True.
            max_feature_name_length (int, optional): Maximum length for feature names.
            Defaults to 15.
        """
        top_shap_values, top_features, _ = self._compute_shap_importance(
            shap_values, top_n, class_id=class_id
        )

        shortened_features = self._shorten_feature_names(
            top_features, max_feature_name_length
        )
        top_shap_values.feature_names = shortened_features

        shap.plots.beeswarm(
            top_shap_values,
            max_display=top_n,
            show=False,
            plot_size=plot_size,
            color_bar=False,
        )

        self._setup_matplotlib_styling(
            xlabel=xlabel,
            ylabel=ylabel,
            font_size=font_size,
            font_name=font_name,
            font_color=font_color,
        )

        if plot_title:
            plt.title(
                plot_title,
                fontdict={
                    "family": font_name,
                    "size": font_size + 12,
                    "color": font_color,
                },
            )

        if show_colorbar:
            ax = plt.gca()
            self._create_custom_colorbar(
                ax=ax,
                colormap=colormap,
                label=colorbar_label,
                high_label=high_label,
                low_label=low_label,
                font_size=font_size,
                font_name=font_name,
                font_color=font_color,
            )

        plt.show()

    def plot_shap_categorical_box(
        self,
        scatter: shap.Explanation,
        feature: str,
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "",
        yaxis_title: str = "SHAP value",
        color_scale: list[str] | None = None,
        font_size: int = 26,
        sort_categories: bool = True,
        sort_by: str = "frequency",
        show_legend: bool = False,
        box_line_color: str = "black",
        box_line_width: int = 1,
        exclude_empty: bool = True,
        max_category_name_length: int = 15,
    ) -> None:
        """Plots a box plot for SHAP values of a categorical feature.

        Args:
            scatter (shap.Explanation): SHAP Explanation object for the feature.
            feature (str): Feature name for axis labeling.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot in pixels. Defaults to 1200.
            height (int, optional): Height of the plot in pixels. Defaults to 1200.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "SHAP value".
            color_scale (list[str] | None, optional): Custom color scale for categories.
            Defaults to None.
            font_size (int, optional): Font size for text elements. Defaults to 26.
            sort_categories (bool, optional): Whether to sort categories. Defaults to True.
            sort_by (str, optional): Sort method - "frequency" or "shap_median".
            Defaults to "frequency".
            show_legend (bool, optional): Whether to show legend. Defaults to False.
            box_line_color (str, optional): Color of box plot outlines. Defaults to "black".
            box_line_width (int, optional): Width of box plot outlines. Defaults to 1.
            exclude_empty (bool, optional): Whether to exclude empty categories. Defaults to True.
            max_category_name_length (int, optional): Maximum length for category names.
            Defaults to 15.
        """
        if color_scale is None:
            color_scale = px.colors.sequential.Rainbow

        category_counts = pd.Series(scatter.data).value_counts()
        labels = np.array(category_counts.index)

        if sort_categories:
            if sort_by == "frequency":
                pass
            elif sort_by == "shap_median":
                medians = []
                for category in labels:
                    indices = np.where(scatter.data == category)[0]
                    grouped_data = scatter.values[indices]
                    grouped_data = grouped_data[~np.isnan(grouped_data)]
                    medians.append(
                        np.median(grouped_data) if len(grouped_data) > 0 else 0
                    )
                sort_indices = np.argsort(medians)[::-1]
                labels = labels[sort_indices]

        shortened_labels = self._shorten_feature_names(labels, max_category_name_length)

        norm = plt.Normalize(0, len(labels))
        colors = [
            color_scale[int(norm(index) * len(color_scale) - 1)]
            for index, _ in enumerate(labels)
        ]

        if not xaxis_title:
            xaxis_title = feature
        if not plot_title:
            plot_title = f"SHAP Values by {feature}"

        fig = go.Figure()

        for color_idx, (category, short_name) in enumerate(
            zip(labels, shortened_labels)
        ):
            indices = np.where(scatter.data == category)[0]
            grouped_data = scatter.values[indices]
            grouped_data = grouped_data[~np.isnan(grouped_data)]

            if exclude_empty and len(grouped_data) == 0:
                continue

            fig.add_trace(
                go.Box(
                    y=grouped_data,
                    name=short_name,
                    marker=dict(
                        color=colors[color_idx],
                        line=dict(color=box_line_color, width=box_line_width),
                    ),
                    showlegend=show_legend,
                )
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )

        fig.update_layout(
            font=dict(family="Times New Roman", size=font_size, color="Black"),
            showlegend=show_legend,
        )

        fig.show("png", width=width, height=height)

    def plot_shap_numerical_scatter(
        self,
        feature_shap_values: shap.Explanation,
        feature: str,
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "",
        yaxis_title: str = "SHAP value",
        colorscale: str = "Rainbow",
        marker_size: int = 5,
        marker_opacity: float = 0.6,
        show_colorbar: bool = True,
        colorbar_thickness: int = 30,
        font_size: int = 26,
        axis_margin_percent: float = 1.0,
        add_trendline: bool = False,
        trendline_color: str = "red",
        trendline_width: int = 2,
    ) -> None:
        """Plots a scatter plot for SHAP values of a numerical feature.

        Args:
            feature_shap_values (shap.Explanation): SHAP Explanation object for the feature.
            feature (str): Feature name for axis labeling.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot in pixels. Defaults to 1200.
            height (int, optional): Height of the plot in pixels. Defaults to 1200.
            xaxis_title (str, optional): Title for the x-axis. Defaults to "".
            yaxis_title (str, optional): Title for the y-axis. Defaults to "SHAP value".
            colorscale (str, optional): Plotly colorscale name for markers. Defaults to "Rainbow".
            marker_size (int, optional): Size of scatter plot markers. Defaults to 5.
            marker_opacity (float, optional): Opacity of markers (0-1). Defaults to 0.6.
            show_colorbar (bool, optional): Whether to show colorbar. Defaults to True.
            colorbar_thickness (int, optional): Thickness of colorbar. Defaults to 30.
            font_size (int, optional): Font size for text elements. Defaults to 26.
            axis_margin_percent (float, optional): Percentage margin for axis ranges.
            Defaults to 1.0.
            add_trendline (bool, optional): Whether to add a trendline. Defaults to False.
            trendline_color (str, optional): Color of trendline. Defaults to "red".
            trendline_width (int, optional): Width of trendline. Defaults to 2.
        """
        x_clean, y_clean = self._clean_data_for_plotting(
            feature_shap_values.data, feature_shap_values.values
        )

        if not xaxis_title:
            xaxis_title = feature
        if not plot_title:
            plot_title = f"SHAP Values vs {feature}"

        fig = go.Figure()

        scatter_trace = go.Scatter(
            x=x_clean,
            y=y_clean,
            mode="markers",
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                color=y_clean,
                colorscale=getattr(
                    px.colors.sequential, colorscale, px.colors.sequential.Rainbow
                ),
                showscale=show_colorbar,
                colorbar=dict(thickness=colorbar_thickness) if show_colorbar else None,
            ),
            showlegend=False,
        )

        fig.add_trace(scatter_trace)

        if add_trendline and len(x_clean) > 1:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)

            fig.add_trace(
                go.Scatter(
                    x=x_clean,
                    y=p(x_clean),
                    mode="lines",
                    line=dict(color=trendline_color, width=trendline_width),
                    name="Trendline",
                    showlegend=False,
                )
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )

        fig.update_layout(
            font=dict(family="Times New Roman", size=font_size, color="Black"),
            showlegend=False,
        )
        if len(x_clean) > 0:
            x_margin = abs(x_clean.max() - x_clean.min()) * axis_margin_percent / 100
            fig.update_xaxes(range=[x_clean.min() - x_margin, x_clean.max() + x_margin])

        if len(y_clean) > 0:
            y_margin = abs(y_clean.max() - y_clean.min()) * axis_margin_percent / 100
            fig.update_yaxes(range=[y_clean.min() - y_margin, y_clean.max() + y_margin])

        fig.show("png", width=width, height=height)

    def plot_shap_force(
        self,
        observation_shap_values: shap.Explanation,
        contribution_threshold: float = 0.07,
        figsize: tuple[float, float] = (25, 5),
        font_size: int = 20,
        font_name: str = "Times New Roman",
        font_color: str = "Black",
        link: str = "identity",
        ordering_keys: list[str] | None = None,
        text_rotation: float = 0,
    ) -> None:
        """Visualizes SHAP force plot for a given sample observation.

        Args:
            observation_shap_values (shap.Explanation): SHAP Explanation
            object for a single observation.
            contribution_threshold (float, optional): Threshold for feature contributions display. 
            Defaults to 0.07.
            figsize (tuple[float, float], optional): Figure size as (width, height).
            Defaults to (25, 5).
            font_size (int, optional): Font size for axis labels and ticks. Defaults to 20.
            font_name (str, optional): Font family name for text elements.
            Defaults to "Times New Roman".
            font_color (str, optional): Color for text elements. Defaults to "Black".
            link (str, optional): Link function for SHAP force plot. Defaults to "identity".
            ordering_keys (list[str] | None, optional): Custom ordering for features.
            Defaults to None.
            text_rotation (float, optional): Rotation angle for text labels. Defaults to 0.
        """
        plt.close("all")

        def safe_round(arr: np.ndarray, decimals: int = 4) -> np.ndarray:
            arr = np.array(arr)
            if np.issubdtype(arr.dtype, np.number):
                return np.round(arr, decimals)
            rounded = []
            for v in arr:
                try:
                    rounded.append(round(float(v), decimals))
                except (TypeError, ValueError):
                    rounded.append(v)
            return np.array(rounded, dtype=object)

        observation_shap_values.values = safe_round(observation_shap_values.values, 4)
        observation_shap_values.data = safe_round(observation_shap_values.data, 4)

        shap.plots.force(
            observation_shap_values,
            matplotlib=True,
            show=False,
            contribution_threshold=contribution_threshold,
            figsize=figsize,
            link=link,
            ordering_keys=ordering_keys,
            text_rotation=text_rotation,
        )
        plt.yticks(fontsize=font_size, fontname=font_name, color=font_color)
        plt.xticks(fontsize=font_size, fontname=font_name, color=font_color)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    np.random.seed(42)
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_informative=7,
        n_redundant=3,
        random_state=42,
    )

    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test[:10])

    shap_plots = SHAPPlots()

    shap_plots.plot_shap_bar(
        shap_values=shap_values,
        top_n=8,
        plot_title="Enhanced SHAP Feature Importance",
        width=1200,
        height=600,
        color_scale="viridis",
        value_precision=4,
        font_size=20,
    )

    shap_plots.plot_custom_shap_beeswarm(
        shap_values=shap_values,
        top_n=8,
        plot_size=(15, 10),
        xlabel="SHAP Value Impact",
        colormap="plasma",
        colorbar_label="Feature Value Magnitude",
        font_size=16,
    )

    shap_plots.plot_shap_numerical_scatter(
        feature_shap_values=shap_values[:, :, 0],
        feature=feature_names[0],
        plot_title="Enhanced SHAP Scatter Analysis",
        width=900,
        height=700,
        colorscale="Viridis",
        marker_size=6,
        add_trendline=True,
        font_size=18,
    )

    shap_plots.plot_shap_force(
        observation_shap_values=shap_values[0, :, 0],
        contribution_threshold=0.05,
        figsize=(20, 4),
        font_size=16,
    )
