import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..base_plots import BasePlots


class CategoricalPlots(BasePlots):
    """Class for creating plots specifically for categorical variables.

    This class inherits from BasePlots and provides specialized methods
    for visualizing categorical data including box plots by category,
    pie charts, and combined categorical-continuous analysis.
    """

    def _apply_top_n_categories(
        self,
        labels: np.ndarray,
        frequency: np.ndarray,
        top_n: int = 10,
        other_category: str = "Other",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Helper method to limit categories to top_n and aggregate the rest as 'Other'.

        Args:
            labels (np.ndarray): Array of category labels.
            frequency (np.ndarray): Array of category frequencies.
            top_n (int): Number of top categories to keep.
            other_category (str): Label for aggregated categories.

        Returns:
            tuple: (labels, frequency, other_labels)
        """
        other_labels = None
        if top_n is not None and top_n < len(labels):
            other_count = frequency[top_n:].sum()
            other_labels = labels[top_n:]
            labels = np.append(labels[:top_n], other_category)
            frequency = np.append(frequency[:top_n], other_count)
        return labels, frequency, other_labels

    def _validate_categorical_groups(
        self,
        data: dict[str, pd.Series | np.ndarray | list[object]],
    ) -> dict[str, pd.Series]:
        """
        Validates and cleans grouped categorical data for distribution comparison.

        Args:
            data (dict[str, pd.Series | np.ndarray | list]): Mapping of group labels
            to their raw categorical values.

        Returns:
            dict[str, pd.Series]: Mapping of group labels to their normalized value
            counts (proportions summing to 1.0 per group).

        Raises:
            TypeError: If data is not a dict.
            ValueError: If fewer than two non-empty groups remain after cleaning.
        """
        if not isinstance(data, dict):
            raise TypeError(
                "data must be a dict mapping group labels to their categorical values."
            )
        group_proportions = {
            str(group_label): pd.Series(group_values)
            .dropna()
            .value_counts(normalize=True)
            for group_label, group_values in data.items()
        }
        group_proportions = {
            group_label: proportions
            for group_label, proportions in group_proportions.items()
            if not proportions.empty
        }
        if len(group_proportions) < 2:
            raise ValueError(
                "compare_categorical_distributions_plot requires at least two "
                "non-empty groups."
            )
        return group_proportions

    def _shared_category_order(
        self,
        group_proportions: dict[str, pd.Series],
        top_n: int | None,
        other_category: str,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Determines shared category order across groups, ranked by summed proportion.

        Args:
            group_proportions (dict[str, pd.Series]): Mapping of group labels to
            their normalized value counts.
            top_n (int | None): Number of top categories to keep. None keeps all.
            other_category (str): Label used for the aggregated category.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: Ordered category labels (with the
            aggregated label appended when truncated) and the original category
            labels collapsed into that bucket, or None when no truncation occurred.
        """
        pooled_share = (
            pd.concat(group_proportions.values(), axis=1).fillna(0.0).sum(axis=1)
        )
        pooled_share = pooled_share.sort_values(ascending=False)
        labels, _, other_labels = self._apply_top_n_categories(
            pooled_share.index.to_numpy(),
            pooled_share.to_numpy(),
            top_n,
            other_category,
        )
        return labels, other_labels

    def _group_values_for_labels(
        self,
        proportions: pd.Series,
        labels: np.ndarray,
        other_labels: np.ndarray | None,
        other_category: str,
    ) -> np.ndarray:
        """
        Builds a per-category proportion array for one group, aligned to shared labels.

        Args:
            proportions (pd.Series): Normalized value counts for a single group.
            labels (np.ndarray): Shared, ordered category labels to align to.
            other_labels (np.ndarray | None): Original categories collapsed into the
            aggregated bucket, if any.
            other_category (str): Label used for the aggregated bucket.

        Returns:
            np.ndarray: Proportions aligned to `labels`, in the same order.
        """
        values = []
        for label in labels:
            if label == other_category and other_labels is not None:
                values.append(proportions.reindex(other_labels, fill_value=0.0).sum())
            else:
                values.append(proportions.get(label, 0.0))
        return np.array(values)

    def compare_categorical_distributions_plot(
        self,
        data: dict[str, pd.Series | np.ndarray | list[object]],
        top_n: int | None = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Categories",
        yaxis_title: str = "Proportion",
        other_category: str = "Other",
        show_counts: bool = True,
        show_other: bool = True,
        tickangle: int = 0,
    ) -> None:
        """
        Creates a grouped bar chart comparing categorical (discrete) distributions
        across named groups of any size.

        Each group's categories are normalized to proportions that sum to 1.0 within
        that group, so groups with very different sample counts (e.g. 10 000 vs 100
        observations) remain comparable by their actual class balance rather than
        being dominated by the larger group's raw counts. Category ordering and
        top-n selection are based on proportions summed across groups rather than
        pooled raw counts, so a category common in a small group is not hidden by a
        large group's volume.

        Args:
            data (dict[str, pd.Series | np.ndarray | list]): Mapping of group labels
            to their raw categorical values (e.g. class labels) to compare.
            top_n (int | None, optional): Number of top categories to show
            separately. Defaults to 10.
            plot_title (str, optional): Title for the plot. Defaults to "".
            width (int, optional): Width of the plot. Defaults to 1600.
            height (int, optional): Height of the plot. Defaults to 800.
            xaxis_title (str, optional): Title for the x-axis. Defaults to
            "Categories".
            yaxis_title (str, optional): Title for the y-axis. Defaults to
            "Proportion".
            other_category (str, optional): Label for aggregated categories.
            Defaults to "Other".
            show_counts (bool, optional): Whether to show percentage labels on bars.
            Defaults to True.
            show_other (bool, optional): Whether to include the aggregated 'Other'
            category. Defaults to True.
            tickangle (int, optional): Angle of the x-axis tick labels. Defaults
            to 0.

        Returns:
            None

        Raises:
            TypeError: If data is not a dict.
            ValueError: If fewer than two non-empty groups are provided.
        """
        group_proportions = self._validate_categorical_groups(data)
        labels, other_labels = self._shared_category_order(
            group_proportions, top_n, other_category
        )
        if not show_other and other_labels is not None:
            labels = labels[labels != other_category]

        colors = self._get_colors(len(group_proportions))

        fig = go.Figure()
        for idx, (group_label, proportions) in enumerate(group_proportions.items()):
            group_values = self._group_values_for_labels(
                proportions, labels, other_labels, other_category
            )
            text_values = (
                [f"{value * 100:.1f}%" for value in group_values]
                if show_counts
                else None
            )
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=group_values,
                    name=group_label,
                    marker=dict(
                        color=colors[idx % len(colors)],
                        line=dict(color="black", width=1),
                    ),
                    text=text_values,
                    textposition="auto" if show_counts else None,
                )
            )
        fig.update_layout(barmode="group")
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(tickangle=tickangle)
        fig.update_layout(
            legend=dict(
                font=dict(size=20),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.99,
                y=0.99,
                xanchor="right",
                yanchor="top",
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            )
        )
        fig.show("png", width=width, height=height)

    def barplot(
        self,
        data: dict[str, int] | pd.Series,
        top_n: int = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Categories",
        yaxis_title: str = "Frequency",
        other_category: str = "Other",
        show_counts: bool = True,
        percentage: bool = True,
        show_other: bool = True,
        tickangle: int = 0,
        return_fig: bool = False,
    ) -> go.Figure | None:
        """Creates a bar plot for categorical data.

        Args:
            data (dict[str, int] | pd.Series): Input data as a dictionary or pandas Series.
            top_n (int): Number of top categories to show separately. Defaults to 10.
            plot_title (str): Title for the plot. Defaults to "".
            width (int): Width of the plot. Defaults to 1600.
            height (int): Height of the plot. Defaults to 800.
            xaxis_title (str): Title for the x-axis. Defaults to "Categories".
            yaxis_title (str): Title for the y-axis. Defaults to "Frequency".
            other_category (str): Label for aggregated categories. Defaults to "Other".
            show_counts (bool): Whether to show counts/percentages on bars. Defaults to True.
            percentage (bool): Whether to show percentages instead of counts. Defaults to True.
            show_other (bool): Whether to include the aggregated 'Other' category. Defaults to True.
            tickangle (int): Angle of the x-axis tick labels. Defaults to 0.
            return_fig (bool): Whether to return the figure object. Defaults to False.
        """
        if isinstance(data, dict):
            labels = np.array(list(data.keys()))
            frequency = np.array(list(data.values()))
        elif isinstance(data, pd.Series):
            labels = np.array(data.index)
            frequency = np.array(data.values)
        else:
            raise ValueError("data must be a dict or pandas Series")

        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        frequency = frequency[sorted_indices]

        labels, frequency, _ = self._apply_top_n_categories(
            labels, frequency, top_n, other_category
        )

        if not show_other and len(labels) > top_n:
            labels = labels[:top_n]
            frequency = frequency[:top_n]

        n_colors = len(labels)
        colors = self._get_colors(n_colors)

        text_values = None
        if show_counts:
            if percentage:
                total = np.sum(frequency)
                if total == 0:
                    text_values = ["0.0%" for _ in frequency]
                else:
                    text_values = [f"{f/total*100:.1f}%" for f in frequency]
            else:
                text_values = [str(c) for c in frequency]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=frequency,
                marker=dict(line=dict(color="black", width=1), color=colors),
                showlegend=False,
                text=text_values,
                textposition="auto" if show_counts else None,
            )
        )
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_xaxes(tickangle=tickangle)
        if return_fig:
            return fig
        fig.show("png", width=width, height=height)
        return None

    def pie_barplot(
        self,
        data: pd.DataFrame,
        feature: str,
        top_n: int = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "Count",
        hole_size: float = 0.3,
        other_category: str = "Other",
        show_counts: bool = True,
        percentage: bool = True,
    ) -> None:
        """Creates a combined pie chart and bar plot for categorical feature distribution.

        Args:
            data (pd.DataFrame): Input DataFrame containing the categorical feature.
            feature (str): Name of the categorical column to analyze.
            top_n (int): Number of top categories to show separately. Defaults to 10.
            plot_title (str): Title for the plot. Defaults to "".
            width (int): Width of the plot. Defaults to 1600.
            height (int): Height of the plot. Defaults to 800.
            xaxis_title (str): Title for the x-axis. Defaults to feature name.
            yaxis_title (str): Title for the y-axis. Defaults to "Count".
            hole_size (float): Size of the hole in the pie chart (0-1). Defaults to 0.3.
            other_category (str): Label for aggregated categories. Defaults to "Other".
            show_counts (bool): Whether to show counts/percentages on bars. Defaults to True.
            percentage (bool): Whether to show percentages instead of counts. Defaults to True.
        """
        data_copy = data[feature].copy()
        data_copy.dropna(inplace=True)
        data_copy.reset_index(drop=True, inplace=True)
        labels, frequency = np.array(data_copy.value_counts().index), np.array(
            data_copy.value_counts().values
        )

        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        frequency = frequency[sorted_indices]

        labels, frequency, _ = self._apply_top_n_categories(
            labels, frequency, top_n, other_category
        )

        n_colors = len(labels)
        colors = self._get_colors(n_colors)

        if not xaxis_title:
            xaxis_title = feature

        bar_text = None
        if show_counts:
            if percentage:
                total = np.sum(frequency)
                bar_text = [f"{f/total*100:.1f}%" for f in frequency]
            else:
                bar_text = [str(f) for f in frequency]

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
        fig.add_trace(
            go.Pie(
                values=frequency,
                labels=labels,
                showlegend=False,
                textinfo="value+percent",
                hole=hole_size,
                marker=dict(line=dict(color="black", width=2), colors=colors),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=frequency,
                text=bar_text,
                textposition="auto" if show_counts else None,
                marker=dict(color=colors, line=dict(color="black", width=2)),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text=yaxis_title, row=1, col=2)
        fig.update_xaxes(title_text=xaxis_title, row=1, col=2)
        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.update_layout(
            barmode="stack",
            legend=dict(
                font=dict(size=20),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.9,
                y=1.1,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )
        fig.show("png", width=width, height=height)

    def boxplot_by_categorical(
        self,
        data: pd.DataFrame,
        categorical: str,
        target: str,
        top_n: int = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "",
        other_category: str = "Other",
        show_counts: bool = True,
        percentage: bool = True,
    ) -> None:
        """Creates box plots grouped by categorical variable.

        Args:
            data (pd.DataFrame): Input DataFrame containing categorical and target variables.
            categorical (str): Name of the categorical column for grouping.
            target (str): Name of the continuous target variable.
            top_n (int): Number of top categories to show separately. Defaults to 10.
            plot_title (str): Title for the plot. Defaults to "".
            width (int): Width of the plot. Defaults to 1600.
            height (int): Height of the plot. Defaults to 800.
            xaxis_title (str): Title for the x-axis. Defaults to categorical column name.
            yaxis_title (str): Title for the y-axis. Defaults to target column name.
            other_category (str): Label for aggregated categories. Defaults to "Other".
            show_counts (bool): Whether to show counts/percentages on boxes. Defaults to True.
            percentage (bool): Whether to show percentages instead of counts. Defaults to True.
        """
        data_copy = data.copy()
        data_copy.reset_index(drop=True, inplace=True)

        value_counts = data_copy[categorical].value_counts()
        labels = np.array(value_counts.index)
        frequency = np.array(value_counts.values)
        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        frequency = frequency[sorted_indices]

        labels, frequency, other_labels = self._apply_top_n_categories(
            labels, frequency, top_n, other_category
        )

        n_colors = len(labels)
        colors = self._get_colors(n_colors)

        fig = go.Figure()

        total_count = np.sum(frequency) if percentage and show_counts else None

        for color_idx, category in enumerate(labels):
            if category == other_category and other_labels is not None:
                indices = data_copy[categorical].isin(other_labels)
            else:
                indices = data_copy[categorical] == category
            grouped_data = list(data_copy[target][indices])

            box_name = str(category)
            if show_counts:
                count = len(grouped_data)
                if percentage and total_count is not None:
                    box_name = f"{category} ({count/total_count*100:.1f}%)"
                else:
                    box_name = f"{category} (n={count})"

            fig.add_trace(
                go.Box(
                    y=grouped_data,
                    name=box_name,
                    marker=dict(color=colors[color_idx]),
                    showlegend=False,
                )
            )

        if not xaxis_title:
            xaxis_title = categorical
        if not yaxis_title:
            yaxis_title = target

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)

    def pie_boxplot_by_categorical(
        self,
        data: pd.DataFrame,
        categorical: str,
        target: str,
        top_n: int = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "",
        hole_size: float = 0.3,
        other_category: str = "Other",
        show_counts: bool = True,
        percentage: bool = True,
    ) -> None:
        """Creates a combined pie chart and box plot analysis by categorical variable.

        Args:
            data (pd.DataFrame): Input DataFrame containing categorical and target variables.
            categorical (str): Name of the categorical column for analysis.
            target (str): Name of the continuous target variable.
            top_n (int): Number of top categories to show separately. Defaults to 10.
            plot_title (str): Title for the plot. Defaults to "".
            width (int): Width of the plot. Defaults to 1600.
            height (int): Height of the plot. Defaults to 800.
            xaxis_title (str): Title for the x-axis. Defaults to categorical column name.
            yaxis_title (str): Title for the y-axis. Defaults to target column name.
            hole_size (float): Size of the hole in the pie chart (0-1). Defaults to 0.3.
            other_category (str): Label for aggregated categories. Defaults to "Other".
            show_counts (bool): Whether to show counts/percentages on boxes. Defaults to True.
            percentage (bool): Whether to show percentages instead of counts. Defaults to True.
        """
        data_copy = data.copy()
        data_copy.reset_index(drop=True, inplace=True)

        value_counts = data_copy[categorical].value_counts()
        labels = np.array(value_counts.index)
        frequency = np.array(value_counts.values)
        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        frequency = frequency[sorted_indices]

        labels, frequency, other_labels = self._apply_top_n_categories(
            labels, frequency, top_n, other_category
        )

        n_colors = len(labels)
        colors = self._get_colors(n_colors)

        if not xaxis_title:
            xaxis_title = categorical
        if not yaxis_title:
            yaxis_title = target

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "pie"}, {"type": "box"}]],
            subplot_titles=("Category Distribution", f"{yaxis_title} by Category"),
        )

        fig.add_trace(
            go.Pie(
                values=frequency,
                labels=labels,
                showlegend=True,
                textinfo="value+percent",
                hole=hole_size,
                marker=dict(line=dict(color="black", width=2), colors=colors),
            ),
            row=1,
            col=1,
        )

        total_count = np.sum(frequency) if percentage and show_counts else None

        for color_idx, category in enumerate(labels):
            if category == other_category and other_labels is not None:
                indices = data_copy[categorical].isin(other_labels)
            else:
                indices = data_copy[categorical] == category
            grouped_data = list(data_copy[target][indices])

            box_name = str(category)
            if show_counts:
                count = len(grouped_data)
                if percentage and total_count is not None:
                    box_name = f"{category} ({count/total_count*100:.1f}%)"
                else:
                    box_name = f"{category} (n={count})"

            fig.add_trace(
                go.Box(
                    y=grouped_data,
                    name=box_name,
                    marker=dict(color=colors[color_idx]),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        self.apply_default_layout(
            fig, plot_title, width, height, xaxis_title, yaxis_title
        )
        fig.show("png", width=width, height=height)

    def pie_stacked_barplot_by_hue(
        self,
        data: pd.DataFrame,
        feature: str,
        hue: str,
        top_n: int = 10,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "Count",
        hole_size: float = 0.3,
        other_category: str = "Other",
        show_counts: bool = True,
        percentage: bool = True,
    ) -> None:
        """Creates a combined pie chart and stacked bar plot by hue variable.

        Args:
            data (pd.DataFrame): Input DataFrame containing the feature and hue columns.
            feature (str): Name of the main categorical column to analyze.
            hue (str): Name of the secondary categorical column for grouping.
            top_n (int): Number of top categories to show separately. Defaults to 10.
            plot_title (str): Title for the plot. Defaults to "".
            width (int): Width of the plot. Defaults to 1600.
            height (int): Height of the plot. Defaults to 800.
            xaxis_title (str): Title for the x-axis. Defaults to hue column name.
            yaxis_title (str): Title for the y-axis. Defaults to "Count".
            hole_size (float): Size of the hole in the pie chart (0-1). Defaults to 0.3.
            other_category (str): Label for aggregated categories. Defaults to "Other".
            show_counts (bool): Whether to show counts/percentages on bars. Defaults to True.
            percentage (bool): Whether to show percentages instead of counts. Defaults to True.
        """
        data_copy = data[[feature, hue]].copy()
        data_copy.dropna(inplace=True)
        data_copy.reset_index(drop=True, inplace=True)
        feature_counts = data_copy[feature].value_counts()
        labels = np.array(feature_counts.index)
        frequency = np.array(feature_counts.values)
        sorted_indices = np.argsort(frequency)[::-1]
        labels = labels[sorted_indices]
        frequency = frequency[sorted_indices]

        labels, frequency, other_labels = self._apply_top_n_categories(
            labels, frequency, top_n, other_category
        )

        if other_labels is not None:
            top_labels = set(labels[:-1])
            data_copy[feature] = data_copy[feature].apply(
                lambda x: x if x in top_labels else other_category
            )

        crosstab = pd.crosstab(data_copy[hue], data_copy[feature])
        crosstab = crosstab[labels]
        n_colors = len(crosstab.columns)
        colors = self._get_colors(n_colors)

        if not xaxis_title:
            xaxis_title = hue

        fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])
        fig.add_trace(
            go.Pie(
                values=frequency,
                labels=labels,
                showlegend=True,
                textinfo="value+percent",
                hole=hole_size,
                marker=dict(line=dict(color="black", width=2), colors=colors),
            ),
            row=1,
            col=1,
        )
        for color_idx, category in enumerate(crosstab.columns):

            bar_text = None
            if show_counts:
                if percentage:
                    percentages = crosstab[category] / crosstab.sum(axis=1) * 100
                    bar_text = [
                        f"{val:.1f}%" if count > 0 else ""
                        for val, count in zip(percentages, crosstab[category])
                    ]
                else:
                    bar_text = [
                        str(count) if count > 0 else "" for count in crosstab[category]
                    ]

            fig.add_trace(
                go.Bar(
                    x=crosstab.index.astype(str),
                    y=crosstab[category],
                    name=str(category),
                    text=bar_text,
                    textposition="auto" if show_counts else None,
                    marker=dict(
                        color=colors[color_idx], line=dict(color="black", width=2)
                    ),
                    showlegend=False,
                ),
                row=1,
                col=2,
            )
        fig.update_yaxes(title_text=yaxis_title, row=1, col=2)
        fig.update_xaxes(title_text=xaxis_title, row=1, col=2)
        fig.update_layout(
            barmode="stack",
            width=width,
            height=height,
            title=f"<b>{plot_title.title()}<b>",
            template="simple_white",
            title_x=0.5,
            font=dict(family="Times New Roman", size=20),
            legend=dict(
                font=dict(size=20),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.9,
                y=1.1,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )
        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    np.random.seed(17)
    sample_data = pd.DataFrame(
        {
            "category": np.random.choice(["A", "B", "C", "D", "E"], 200),
            "value": np.random.normal(50, 15, 200),
            "group": np.random.choice(["Group1", "Group2", "Group3"], 200),
        }
    )

    cat_plots = CategoricalPlots()

    cat_plots.barplot(
        sample_data["category"].value_counts(),
        plot_title="Category Frequency",
        width=800,
        height=600,
        xaxis_title="Category",
        yaxis_title="Frequency",
    )

    cat_plots.compare_categorical_distributions_plot(
        data={
            "Train (10k obs)": np.random.choice(
                ["A", "B", "C", "D", "E"], 10000, p=[0.4, 0.3, 0.15, 0.1, 0.05]
            ),
            "Test (100 obs)": np.random.choice(
                ["A", "B", "C", "D", "E"], 100, p=[0.35, 0.25, 0.2, 0.1, 0.1]
            ),
        },
        plot_title="Class Distribution: Train vs Test",
    )

    cat_plots.pie_barplot(
        sample_data,
        feature="category",
        plot_title="Pie and Bar Plot of Category",
        width=800,
        height=600,
        xaxis_title="Category",
        yaxis_title="Count",
        hole_size=0.3,
    )

    cat_plots.boxplot_by_categorical(
        sample_data,
        categorical="category",
        target="value",
        plot_title="Boxplot by Category",
        width=800,
        height=600,
        xaxis_title="Category",
        yaxis_title="Value",
    )

    cat_plots.pie_boxplot_by_categorical(
        sample_data,
        categorical="category",
        target="value",
        top_n=3,
        plot_title="Pie and Boxplot by Category",
        width=800,
        height=600,
        xaxis_title="Category",
        yaxis_title="Value",
        hole_size=0.3,
    )

    cat_plots.pie_stacked_barplot_by_hue(
        sample_data,
        feature="category",
        hue="group",
        plot_title="Pie and Stacked Barplot by Group",
        width=800,
        height=600,
        xaxis_title="Group",
        yaxis_title="Count",
        hole_size=0.3,
    )
