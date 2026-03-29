
from collections.abc import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)


class BasePlots:
    """Base class for all plotting functionality.

    This class provides common methods and utilities that can be used across
    different plotting modules. It includes data validation, color management,
    and standard plot configurations.

    Attributes:
        quantiles_dict (dict): Dictionary mapping quantile names to their values.
        default_template (str): Default plotly template for all plots.
        default_font (dict): Default font configuration for plots.
        default_colors (dict): Default color schemes for different plot types.
    """

    def __init__(self) -> None:
        """Initialize the BasePlots class with default configurations."""
        self.metrics_dict: dict[str, Callable[..., float]] = {
            "Accuracy": accuracy_score,
            "Balanced Accuracy": balanced_accuracy_score,
            "F1": f1_score,
            "F1 (Micro)": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="micro"
            ),
            "F1 (Macro)": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="macro"
            ),
            "F1 (Weighted)": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="weighted"
            ),
            "Recall": recall_score,
            "Precision": precision_score,
            "Roc AUC": roc_auc_score,
            "MAE": mean_absolute_error,
            "MAPE": mean_absolute_percentage_error,
            "MSE": mean_squared_error,
            "RMSE": root_mean_squared_error,
            "RMSLE": root_mean_squared_log_error,
            "R2": r2_score,
        }
        self.quantiles_dict = {"Min": 0, "Q1": 0.25, "Med": 0.5, "Q3": 0.75, "Max": 1}
        self.default_template = "simple_white"
        self.default_font = dict(family="Times New Roman", size=26, color="Black")
        self.default_colors = {
            "primary": "rgb(48,70,116)",
            "secondary": "lightblue",
            "accent": "green",
            "qualitative": px.colors.qualitative.Dark24,
        }

    def check_data(
        self,
        data: pd.DataFrame | pd.Series | np.ndarray | list[object],
    ) -> np.ndarray:
        """
        Validates and converts input data to a NumPy array.

        Args:
            data (pd.DataFrame | pd.Series | np.ndarray | list): Input data to validate.

        Returns:
            np.ndarray: Converted data as a NumPy array with NaN values removed.

        Raises:
            TypeError: If data is not a pandas DataFrame, Series, numpy array, or list.
        """
        if not isinstance(data, (pd.DataFrame, pd.Series, np.ndarray, list)):
            raise TypeError(
                "Wrong type of data. It should be pandas DataFrame, pandas Series,"
                " numpy array, or list"
            )
        data = np.array(data)
        try:
            data = data[~np.isnan(data)]
        except TypeError:
            pass

        if data.ndim == 2:
            data = data.squeeze()
        return data

    def check_2d_data(
        self,
        data: pd.DataFrame | np.ndarray | list[object],
    ) -> np.ndarray:
        """
        Validates and converts input data to a 2D NumPy array.

        Args:
            data (pd.DataFrame | np.ndarray | list): Input data to validate.

        Returns:
            np.ndarray: Converted data as a 2D NumPy array with rows containing NaN values removed.

        Raises:
            TypeError: If data is not a pandas DataFrame, numpy array, or list.
        """
        if not isinstance(data, (pd.DataFrame, np.ndarray, list)):
            raise TypeError(
                "Wrong type of data. It should be pandas DataFrame, numpy array, or list"
            )
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2:
            raise ValueError("Input data must be 2-dimensional after conversion.")
        if np.issubdtype(data.dtype, np.number):
            data = data[~np.isnan(data).any(axis=1)]
        return data

    def _get_colors(self, n_colors: int) -> list[str]:
        """
        Get colors based on the number of colors.

        Args:
            n_colors (int): Number of colors to determine the color scheme.

        Returns:
            list: A list of colors.
        """
        if n_colors > 10:
            colors = px.colors.sample_colorscale(
                "rainbow",
                [n / (n_colors - 1) if n_colors > 1 else 0 for n in range(n_colors)],
            )
        else:
            colors = px.colors.qualitative.Pastel
        return colors

    def apply_default_layout(
        self,
        fig: go.Figure,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "",
        yaxis_title: str = "",
    ) -> None:
        """
        Apply default layout settings to a plotly figure.

        Args:
            fig (go.Figure): The plotly figure to update.
            plot_title (str, optional): Plot title. Defaults to "".
            width (int, optional): Figure width. Defaults to 1600.
            height (int, optional): Figure height. Defaults to 800.
            xaxis_title (str, optional): X axis title. Defaults to "".
            yaxis_title (str, optional): Y axis title. Defaults to "".
        """
        fig.update_layout(
            template=self.default_template,
            width=width,
            height=height,
            title=f"<b>{plot_title}</b>" if plot_title else "",
            title_x=0.5,
            font=self.default_font,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )

    def filter_nan_indices(self, data: pd.DataFrame, feature: str) -> pd.Index:
        """
        Get indices where the specified feature is not NaN.

        Args:
            data (pd.DataFrame): Input DataFrame.
            feature (str): Column name to check for NaN values.

        Returns:
            pd.Index: Boolean index of non-NaN values.
        """
        return ~data[feature].isna()

    def add_quantile_annotations(
        self,
        fig: go.Figure,
        data: np.ndarray,
        annotations: list[str] | bool | None = None,
        x_position: float = 0.4,
    ) -> None:
        """
        Add selected quantile annotations to a plot.

        Args:
            fig (go.Figure): Target figure that receives the annotation labels.
            data (np.ndarray): Numeric data used to compute quantiles.
            annotations (list[str] | bool | None, optional): Annotation labels to show.
                When None, labels are selected automatically. When True, all labels are
                shown. When False, no labels are added. Defaults to None.
            x_position (float, optional): Horizontal position shared by all labels.
                Defaults to 0.4.

        Returns:
            None: This method updates the figure in place.
        """
        quantiles_dict = {"Min": 0, "Q1": 0.25, "Med": 0.5, "Q3": 0.75, "Max": 1}
        if annotations is None:
            annotations = self._auto_quantile_annotations(data)
        elif annotations is True:
            annotations = list(quantiles_dict.keys())
        elif annotations is False:
            return

        for annotation in annotations:
            if annotation in quantiles_dict:
                quantile_value = np.quantile(data, quantiles_dict[annotation])
                fig.add_annotation(
                    x=x_position,
                    y=quantile_value,
                    text=f"{annotation}: {np.round(quantile_value, 3)}",
                    showarrow=False,
                    yshift=0,
                    font=dict(size=22, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    opacity=0.95,
                )

    def _auto_quantile_annotations(
        self, data: np.ndarray, min_dist: float = 0.05
    ) -> list[str]:
        """
        Decide which quantile labels should be displayed.

        Args:
            data (np.ndarray): Numeric values used to calculate quantiles.
            min_dist (float, optional): Minimum relative distance required to include
                nearby quantile labels. Defaults to 0.05.

        Returns:
            list[str]: Ordered list of quantile labels that should be rendered.
        """
        quantiles_dict = {"Min": 0, "Q1": 0.25, "Med": 0.5, "Q3": 0.75, "Max": 1}
        unique_vals = np.unique(data)
        if len(unique_vals) == 1:
            return ["Med"]
        if len(unique_vals) == 2:
            return ["Min", "Max"]
        quantile_vals = {k: np.quantile(data, v) for k, v in quantiles_dict.items()}
        data_range = quantile_vals["Max"] - quantile_vals["Min"]
        show = ["Min"]
        if (
            data_range == 0
            or abs(quantile_vals["Q1"] - quantile_vals["Med"])
            / (data_range if data_range else 1)
            > min_dist
        ):
            show.append("Q1")
        med_min_dist = abs(quantile_vals["Med"] - quantile_vals["Min"]) / (
            data_range if data_range else 1
        )
        med_max_dist = abs(quantile_vals["Max"] - quantile_vals["Med"]) / (
            data_range if data_range else 1
        )
        if data_range == 0 or (med_min_dist > min_dist and med_max_dist > min_dist):
            show.append("Med")
        if (
            data_range == 0
            or abs(quantile_vals["Q3"] - quantile_vals["Med"])
            / (data_range if data_range else 1)
            > min_dist
        ):
            show.append("Q3")
        show.append("Max")
        show = list(dict.fromkeys(show))
        return show

    def calculate_dynamic_dimensions(
        self,
        n_items: int,
        min_width: int = 1600,
        min_height: int = 800,
        scale_factor: int = 30,
    ) -> tuple[int, int]:
        """
        Calculate dynamic plot dimensions based on number of items.

        Args:
            n_items (int): Number of items to display.
            min_width (int, optional): Minimum width. Defaults to 1600.
            min_height (int, optional): Minimum height. Defaults to 800.
            scale_factor (int, optional): Scaling factor per item. Defaults to 30.

        Returns:
            tuple[int, int]: Width and height values.
        """
        width = max(scale_factor * n_items, min_width)
        height = max(scale_factor * n_items, min_height)
        return width, height

    def create_subplot_layout(
        self, rows: int, cols: int, subplot_types: list[list[str]]
    ) -> go.Figure:
        """
        Create a subplot layout with specified types.

        Args:
            rows (int): Number of rows.
            cols (int): Number of columns.
            subplot_types (list[list[str]]): 2D list of subplot types, where
            each inner list represents a row.

        Returns:
            go.Figure: Configured subplot figure.
        """
        specs = [
            [{"type": subplot_type} for subplot_type in row] for row in subplot_types
        ]
        return make_subplots(rows=rows, cols=cols, specs=specs)

    def _apply_aggregation(self, data: pd.Series, agg_func: str) -> float:
        """Applies aggregation function to time series data.

        Args:
            data (pd.Series): Input time series data.
            agg_func (str): Aggregation function name.

        Returns:
            float: Aggregated scalar value.
        """
        if agg_func == "mean":
            return float(data.mean())
        elif agg_func == "sum":
            return float(data.sum())
        elif agg_func == "max":
            return float(data.max())
        elif agg_func == "min":
            return float(data.min())
        elif agg_func == "std":
            return float(data.std())
        elif agg_func == "median":
            return float(data.median())
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_func}")


if __name__ == "__main__":
    base_plots = BasePlots()
    sample_values = np.array([1.0, 2.5, np.nan, 4.0])
    cleaned_values = base_plots.check_data(sample_values)
    print(cleaned_values.tolist())
