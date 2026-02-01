import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from kvbiii_plots.base_plots import BasePlots


class ClassificationPlots(BasePlots):
    """Class for creating plots specifically for classification model evaluation.

    This class inherits from BasePlots and provides specialized methods
    for visualizing classification results including confusion matrices,
    probability distributions, and ROC curves.
    """

    def _create_confusion_matrix_heatmap(
        self,
        confusion_matrix_data: np.ndarray,
        labels: list[str],
        colorscale: str = "blues",
        show_text: bool = True,
        showscale: bool = True,
    ) -> go.Heatmap:
        """Creates a heatmap trace for confusion matrix visualization.

        Args:
            confusion_matrix_data (np.ndarray): The confusion matrix values.
            labels (list[str]): Class labels for axes.
            colorscale (str, optional): Plotly colorscale name. Defaults to "blues".
            show_text (bool, optional): Whether to show values on heatmap. Defaults to True.
            showscale (bool, optional): Whether to show the colorscale bar. Defaults to True.

        Returns:
            go.Heatmap: Plotly heatmap trace.
        """
        z_text = np.around(confusion_matrix_data, 3).astype(str) if show_text else None

        return go.Heatmap(
            z=confusion_matrix_data,
            x=labels,
            y=labels,
            colorscale=colorscale,
            showscale=showscale,
            text=z_text,
            texttemplate="%{text}" if show_text else None,
        )

    def _add_axis_annotations(
        self,
        fig: go.Figure,
        font_size: int = 22,
    ) -> None:
        """Adds True/Pred axis annotations to confusion matrix subplots.

        Args:
            fig (go.Figure): The plotly figure to annotate.
            font_size (int, optional): Font size for annotations. Defaults to 22.
        """
        if len(fig.data) == 1:
            annotations = [
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=-0.15,
                    y=0.5,
                    showarrow=False,
                    text="True",
                    textangle=-90,
                    xref="paper",
                    yref="paper",
                ),
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=0.5,
                    y=-0.1,
                    showarrow=False,
                    text="Pred",
                    xref="paper",
                    yref="paper",
                ),
            ]
        elif len(fig.data) == 2:
            annotations = [
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=-0.08,
                    y=0.5,
                    showarrow=False,
                    text="True",
                    textangle=-90,
                    xref="paper",
                    yref="paper",
                ),
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=0.21,
                    y=-0.1,
                    showarrow=False,
                    text="Pred",
                    xref="paper",
                    yref="paper",
                ),
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=0.48,
                    y=0.5,
                    showarrow=False,
                    text="True",
                    textangle=-90,
                    xref="paper",
                    yref="paper",
                ),
                dict(
                    font=dict(family="Times New Roman", size=font_size, color="Black"),
                    x=0.81,
                    y=-0.1,
                    showarrow=False,
                    text="Pred",
                    xref="paper",
                    yref="paper",
                ),
            ]
        for ann in annotations:
            fig.add_annotation(ann)

    def _format_threshold_string(self, labels: list[str], cutoffs: np.ndarray) -> str:
        """Formats threshold values for display in subplot titles.

        Args:
            labels (list[str]): Class labels.
            cutoffs (np.ndarray): Threshold values for each class.

        Returns:
            str: Formatted threshold string for display.
        """
        thresholds_str = [
            f"{label}: {threshold:.2f}" for label, threshold in zip(labels, cutoffs)
        ]
        return "<br>- " + "<br>- ".join(thresholds_str)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        cutoffs: float | np.ndarray | None = None,
        normalize: bool = False,
        plot_title: str = "",
        width: int = 1000,
        height: int = 1000,
        colorscale: str = "blues",
        font_size: int = 26,
        show_text: bool = True,
    ) -> None:
        """Creates a confusion matrix with optional threshold-based predictions.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            cutoffs (float | np.ndarray | None, optional): Threshold values for each class. Defaults to None.
            normalize (bool, optional): Whether to normalize confusion matrix. Defaults to False.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1000.
            height (int, optional): Plot height in pixels. Defaults to 1000.
            colorscale (str, optional): Plotly colorscale name. Defaults to "blues".
            font_size (int, optional): Font size for text elements. Defaults to 26.
            show_text (bool, optional): Whether to show values on heatmap. Defaults to True.
        """
        # Convert inputs to numpy arrays for consistent processing
        y_true = np.array(y_true)
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=1)
        n_classes = probabilities.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        # Handle cutoffs
        if cutoffs is None:
            cutoffs = np.full(n_classes, 0.5)
        elif isinstance(cutoffs, float) or isinstance(cutoffs, int):
            cutoffs = np.full(n_classes, float(cutoffs))
        elif isinstance(cutoffs, np.ndarray):
            if cutoffs.shape == ():
                cutoffs = np.full(n_classes, float(cutoffs))
            elif cutoffs.shape[0] != n_classes:
                raise ValueError(
                    f"cutoffs shape {cutoffs.shape} does not match number of classes {n_classes}"
                )
        else:
            raise TypeError("cutoffs must be None, float, int, or np.ndarray")

        if n_classes == 2 and (
            isinstance(cutoffs, np.ndarray) and np.all(cutoffs == cutoffs[0])
        ):
            # Binary classification with single cutoff
            y_pred = (probabilities[:, 1] >= cutoffs[0]).astype(int)
        else:
            # Multiclass: apply per-class cutoffs
            y_pred = np.argmax(probabilities >= cutoffs, axis=1)

        y_pred = [id2label[int(pred)] for pred in y_pred]

        # Format threshold string for display
        labels = list(id2label.values())

        # Compute confusion matrix
        norm_param = "true" if normalize else None

        if y_true.dtype != np.str_ and y_true.dtype != np.object_:
            y_true = [id2label[int(label)] for label in y_true]

        confusion_matrix_values = confusion_matrix(
            y_true, y_pred, normalize=norm_param, labels=labels
        )

        # Generate title
        matrix_type = "Normalized" if normalize else ""
        f1_value = f1_score(y_true, y_pred, average="macro")

        # title = plot_title or f"{matrix_type} Confusion matrix".strip()
        title = (
            f"{plot_title}<br>F1 Score: {f1_value:.4f}".strip()
            if plot_title
            else f"{matrix_type} Confusion matrix<br>F1 Score: {f1_value:.4f}".strip()
        )

        # Create figure
        fig = go.Figure()

        # Add heatmap trace
        labels_str = [str(label) for label in labels]
        fig.add_trace(
            self._create_confusion_matrix_heatmap(
                confusion_matrix_values, labels_str, colorscale, show_text
            )
        )

        # Apply layout and styling
        self.apply_default_layout(fig, title, width, height, "", "")
        self._add_axis_annotations(fig, font_size)

        fig.show("png", width=width, height=height)

    def subplot_conf_matrix(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        cutoffs: float | np.ndarray,
        normalize: bool = False,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        colorscale: str = "blues",
        show_text: bool = True,
        font_size: int = 22,
    ) -> None:
        """Creates side-by-side confusion matrices comparing no threshold vs threshold predictions.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            cutoffs (float | np.ndarray): Threshold values for each class.
            normalize (bool, optional): Whether to normalize confusion matrix. Defaults to False.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1600.
            height (int, optional): Plot height in pixels. Defaults to 800.
            colorscale (str, optional): Plotly colorscale name. Defaults to "blues".
            show_text (bool, optional): Whether to show values on heatmap. Defaults to True.
            font_size (int, optional): Font size for text elements. Defaults to 22.
        """
        # Convert inputs to numpy arrays for consistent processing
        y_true = np.array(y_true)
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=1)
        n_classes = probabilities.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        # Handle cutoffs with proper validation
        if isinstance(cutoffs, float) or isinstance(cutoffs, int):
            cutoffs = np.full(n_classes, float(cutoffs))
        elif isinstance(cutoffs, np.ndarray):
            if cutoffs.shape == ():
                cutoffs = np.full(n_classes, float(cutoffs))
            elif cutoffs.shape[0] != n_classes:
                raise ValueError(
                    f"cutoffs shape {cutoffs.shape} does not match number of classes {n_classes}"
                )
        else:
            raise TypeError("cutoffs must be float, int, or np.ndarray")

        if n_classes == 2 and np.all(cutoffs == cutoffs[0]):
            # Binary classification with single cutoff
            cutoffs = np.array([cutoffs[0]])
            y_pred_no_threshold = (probabilities[:, 1] >= 0.5).astype(int)
            y_pred_threshold = (probabilities[:, 1] >= cutoffs[0]).astype(int)
        else:
            y_pred_no_threshold = np.argmax(probabilities, axis=1)
            y_pred_threshold = np.argmax(probabilities >= cutoffs, axis=1)
        y_pred_no_threshold = [id2label[int(pred)] for pred in y_pred_no_threshold]
        y_pred_threshold = [id2label[int(pred)] for pred in y_pred_threshold]

        # Format threshold string for display
        labels = list(id2label.values())
        thresholds_str = self._format_threshold_string(labels, cutoffs)

        # Compute confusion matrices
        norm_param = "true" if normalize else None

        if y_true.dtype != np.str_ and y_true.dtype != np.object_:
            y_true = [id2label[int(label)] for label in y_true]

        confusion_matrix_values_no_threshold = confusion_matrix(
            y_true, y_pred_no_threshold, normalize=norm_param, labels=labels
        )
        confusion_matrix_values_threshold = confusion_matrix(
            y_true, y_pred_threshold, normalize=norm_param, labels=labels
        )

        # Generate title
        matrix_type = "Normalized" if normalize else ""
        title = plot_title or f"{matrix_type} Confusion matrix".strip()

        # Calculate F1 scores
        f1_no_threshold = f1_score(y_true, y_pred_no_threshold, average="macro")
        f1_threshold = f1_score(y_true, y_pred_threshold, average="macro")

        # Create subplot figure
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"<b>No Threshold<br>F1 Score: {f1_no_threshold:.4f}<b>",
                f"<b>{thresholds_str}<br>F1 Score: {f1_threshold:.4f}<b>",
            ),
        )

        labels_str = [str(label) for label in labels]

        fig.add_trace(
            self._create_confusion_matrix_heatmap(
                confusion_matrix_values_no_threshold,
                labels_str,
                colorscale,
                show_text,
                showscale=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            self._create_confusion_matrix_heatmap(
                confusion_matrix_values_threshold,
                labels_str,
                colorscale,
                show_text,
                showscale=False,
            ),
            row=1,
            col=2,
        )

        # Apply layout and styling
        self.apply_default_layout(fig, title, width, height, "", "")

        # Add custom layout adjustments
        fig.update_layout(
            showlegend=False,
            title_y=0.97,
            font=dict(family="Times New Roman", size=font_size, color="Black"),
        )

        self._add_axis_annotations(fig, font_size)

        fig.show("png", width=width, height=height)

    def _create_threshold_line(
        self,
        fig: go.Figure,
        threshold: float,
        line_color: str = "red",
        line_width: int = 4,
        annotation_text: str = "",
        annotation_position: str = "top right",
        font_size: int = 22,
    ) -> None:
        """Adds a vertical threshold line with annotation to a figure.

        Args:
            fig (go.Figure): The plotly figure to add the line to.
            threshold (float): X-coordinate for the vertical line.
            line_color (str, optional): Color of the threshold line. Defaults to "red".
            line_width (int, optional): Width of the threshold line. Defaults to 4.
            annotation_text (str, optional): Text for the annotation. Defaults to "".
            annotation_position (str, optional): Position of annotation. Defaults to "top right".
            font_size (int, optional): Font size for annotation. Defaults to 22.
        """
        if not annotation_text:
            annotation_text = f"Threshold: {threshold:.2f}"

        fig.add_vline(
            x=threshold, line_width=line_width, line_dash="dash", line_color=line_color
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=line_color,
            annotation_text=annotation_text,
            annotation_position=annotation_position,
            annotation_font_size=font_size,
            annotation_font_color=line_color,
        )

    def _prepare_probability_data(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        labels: list[str],
        class_name: str,
    ) -> pd.DataFrame:
        """Prepares data for probability distribution plotting.

        Args:
            y_true (np.ndarray): True labels.
            probabilities (np.ndarray): Predicted probabilities.
            labels (list[str]): All class labels.
            class_name (str): Current class being processed.

        Returns:
            pd.DataFrame: Prepared data for plotting.
        """
        true_class_indices = np.where(y_true == class_name)[0]
        wide_x_list = [
            str(class_name)
            for _ in range(len(true_class_indices))
            for class_name in labels
        ]
        wide_prob_list = probabilities[true_class_indices].flatten()

        return pd.DataFrame({"Y": wide_x_list, "Pred_Proba": wide_prob_list})

    def plot_probabilities_per_class(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        cutoffs: float | np.ndarray | None = None,
        plot_title: str = "",
        width: int = 1600,
        height: int = 800,
        xaxis_title: str = "Probability",
        yaxis_title: str = "Class",
        threshold_color: str = "red",
        threshold_line_width: int = 4,
        marker_size: int = 5,
        jitter: float = 0.9,
        font_size: int = 22,
    ) -> None:
        """Creates probability distribution plots for each class with threshold lines.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            cutoffs (float | np.ndarray | None, optional): Threshold values for each class. Defaults to None.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1600.
            height (int, optional): Plot height in pixels. Defaults to 800.
            xaxis_title (str, optional): X-axis title. Defaults to "Probability".
            yaxis_title (str, optional): Y-axis title. Defaults to "Class".
            threshold_color (str, optional): Color for threshold lines. Defaults to "red".
            threshold_line_width (int, optional): Width of threshold lines. Defaults to 4.
            marker_size (int, optional): Size of strip plot markers. Defaults to 5.
            jitter (float, optional): Jitter amount for strip plot. Defaults to 0.9.
            font_size (int, optional): Font size for text elements. Defaults to 22.
        """
        # Convert inputs to numpy arrays for consistent processing
        y_true = np.array(y_true)
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=1)
        n_classes = probabilities.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        # Handle cutoffs
        if cutoffs is None:
            cutoffs = np.full(n_classes, 0.5)
        elif isinstance(cutoffs, float) or isinstance(cutoffs, int):
            cutoffs = np.full(n_classes, float(cutoffs))
        elif isinstance(cutoffs, np.ndarray):
            if cutoffs.shape == ():
                cutoffs = np.full(n_classes, float(cutoffs))
            elif cutoffs.shape[0] != n_classes:
                raise ValueError(
                    f"cutoffs shape {cutoffs.shape} does not match number of classes {n_classes}"
                )
        else:
            raise TypeError("cutoffs must be None, float, int, or np.ndarray")

        if y_true.dtype != np.str_ and y_true.dtype != np.object_:
            y_true = np.array([id2label[int(label)] for label in y_true])
        for class_idx, class_name in enumerate(id2label.values()):
            df_temp = self._prepare_probability_data(
                y_true, probabilities, id2label.values(), class_name
            )

            # Create strip plot
            fig = px.strip(
                df_temp,
                y="Y",
                x="Pred_Proba",
                color="Y",
                stripmode="overlay",
                orientation="h",
                title="Beeswarm Plot",
                labels={"Y": yaxis_title, "Pred_Proba": xaxis_title},
            )

            # Generate title
            title = f"Probability distribution for true class: {class_name}"

            # Apply layout and styling
            self.apply_default_layout(
                fig, title, width, height, xaxis_title, yaxis_title
            )

            # Update layout specifics
            fig.update_layout(
                showlegend=False,
                boxgap=0,
            )

            # Update traces and axes
            fig.update_xaxes(range=[-0.01, 1.01])
            fig.update_traces(jitter=jitter, marker={"size": marker_size})

            # Add threshold line
            if cutoffs is not None:
                threshold = cutoffs[class_idx]
                self._create_threshold_line(
                    fig,
                    threshold,
                    threshold_color,
                    threshold_line_width,
                    font_size=font_size,
                )

            fig.show("png", width=width, height=height)

    def _compute_binary_roc(
        self, y_true: np.ndarray, probabilities: np.ndarray, ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Computes ROC curve for binary classification.

        Args:
            y_true (np.ndarray): True binary labels.
            probabilities (np.ndarray): Predicted probabilities.
            ids (list[str]): Class labels.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: FPR, TPR, and AUC score.
        """
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            y_score = probabilities[:, 1]
        else:
            y_score = probabilities.ravel()
        y_true_bin = label_binarize(y_true, classes=ids).ravel()
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc

    def _compute_multiclass_roc(
        self, y_true: np.ndarray, probabilities: np.ndarray, ids: list[str]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Computes ROC curves for multiclass classification.

        Args:
            y_true (np.ndarray): True multiclass labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            ids (list[str]): Class labels.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[float]]: Lists of FPR, TPR, and AUC scores for each class.
        """
        y_true_bin = label_binarize(y_true, classes=ids)
        n_classes = len(ids)

        fpr_list, tpr_list, roc_auc_list = [], [], []

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], probabilities[:, i])
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)

        return fpr_list, tpr_list, roc_auc_list

    def _compute_binary_pr(
        self, y_true: np.ndarray, probabilities: np.ndarray, ids: list[str]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Computes Precision-Recall curve for binary classification.

        Args:
            y_true (np.ndarray): True binary labels.
            probabilities (np.ndarray): Predicted probabilities.
            ids (list[str]): Class labels.

        Returns:
            tuple[np.ndarray, np.ndarray, float]: Precision, Recall, and Average Precision score.
        """
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            y_score = probabilities[:, 1]
        else:
            y_score = probabilities.ravel()
        y_true_bin = label_binarize(y_true, classes=ids).ravel()
        precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
        avg_precision = average_precision_score(y_true_bin, y_score)

        return precision, recall, avg_precision

    def _compute_multiclass_pr(
        self, y_true: np.ndarray, probabilities: np.ndarray, ids: list[str]
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
        """Computes Precision-Recall curves for multiclass classification.

        Args:
            y_true (np.ndarray): True multiclass labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            ids (list[str]): Class labels.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[float]]: Lists of Precision, Recall, and Average Precision scores for each class.
        """
        y_true_bin = label_binarize(y_true, classes=ids)
        n_classes = len(ids)

        precision_list, recall_list, avg_precision_list = [], [], []

        for i in range(n_classes):
            precision, recall, _ = precision_recall_curve(
                y_true_bin[:, i], probabilities[:, i]
            )
            avg_precision = average_precision_score(
                y_true_bin[:, i], probabilities[:, i]
            )
            precision_list.append(precision)
            recall_list.append(recall)
            avg_precision_list.append(avg_precision)

        return precision_list, recall_list, avg_precision_list

    def _add_random_classifier_line(
        self,
        fig: go.Figure,
        line_color: str = "black",
        line_width: int = 1,
        show_legend: bool = False,
    ) -> None:
        """Adds a diagonal line representing random classifier performance.

        Args:
            fig (go.Figure): The plotly figure to add the line to.
            line_color (str, optional): Color of the diagonal line. Defaults to "black".
            line_width (int, optional): Width of the diagonal line. Defaults to 1.
            show_legend (bool, optional): Whether to show in legend. Defaults to False.
        """
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=line_color, dash="dash", width=line_width),
                name="Random classifier",
                showlegend=show_legend,
            )
        )

    def plot_roc_auc(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "False Positive Rate",
        yaxis_title: str = "True Positive Rate",
        random_line_color: str = "black",
        random_line_width: int = 1,
        font_size: int = 22,
        line_width: int = 5,
    ) -> None:
        """Plots ROC curves for multiclass or binary classification.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities. For binary: (n_samples,) or (n_samples, 2).
                For multiclass: (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1200.
            height (int, optional): Plot height in pixels. Defaults to 1200.
            xaxis_title (str, optional): X-axis title. Defaults to "False Positive Rate".
            yaxis_title (str, optional): Y-axis title. Defaults to "True Positive Rate".
            random_line_color (str, optional): Color for random classifier line. Defaults to "black".
            random_line_width (int, optional): Width for random classifier line. Defaults to 1.
            font_size (int, optional): Font size for text elements. Defaults to 22.
            line_width (int, optional): Width of ROC curve lines. Defaults to 2.
        """
        # Convert inputs to numpy arrays for consistent processing
        y_true = np.array(y_true)
        ids = list(id2label.keys())
        labels = list(id2label.values())
        n_classes = len(ids)
        fig = go.Figure()

        # Compute ROC curves based on number of classes
        if n_classes == 2:
            fpr, tpr, roc_auc = self._compute_binary_roc(y_true, probabilities, ids)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"Class {labels[1]} (AUC = {roc_auc:.3f})",
                    line=dict(width=line_width),
                )
            )
            avg_roc_auc = roc_auc
        else:
            fpr_list, tpr_list, roc_auc_list = self._compute_multiclass_roc(
                y_true, probabilities, ids
            )

            for i, (fpr, tpr, roc_auc) in enumerate(
                zip(fpr_list, tpr_list, roc_auc_list)
            ):
                fig.add_trace(
                    go.Scatter(
                        x=fpr,
                        y=tpr,
                        mode="lines",
                        name=f"Class {labels[i]} (AUC = {roc_auc:.3f})",
                        line=dict(width=line_width),
                    )
                )

            avg_roc_auc = np.mean(roc_auc_list)

        self._add_random_classifier_line(
            fig, random_line_color, random_line_width, show_legend=False
        )

        # Generate title
        curve_text = "ROC Curves" if n_classes > 2 else "ROC Curve"
        title = (
            f"{plot_title}<br>(AUC = {avg_roc_auc:.3f})"
            if plot_title
            else f"{curve_text} (AUC = {avg_roc_auc:.3f})"
        )

        # Apply layout and styling
        self.apply_default_layout(fig, title, width, height, xaxis_title, yaxis_title)

        # Update layout specifics
        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=font_size - 2),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.8,
                y=0.01,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )

        # Set axis ranges
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        fig.show("png", width=width, height=height)

    def plot_precision_recall(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "Recall",
        yaxis_title: str = "Precision",
        font_size: int = 22,
        line_width: int = 5,
        cutoffs: float | np.ndarray | None = None,
        cutoff_line_color: str = "red",
        cutoff_line_width: int = 3,
    ) -> None:
        """Plots Precision-Recall curves for multiclass or binary classification, with optional cutoff vlines.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities. For binary: (n_samples,) or (n_samples, 2).
                For multiclass: (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1200.
            height (int, optional): Plot height in pixels. Defaults to 1200.
            xaxis_title (str, optional): X-axis title. Defaults to "Recall".
            yaxis_title (str, optional): Y-axis title. Defaults to "Precision".
            font_size (int, optional): Font size for text elements. Defaults to 22.
            line_width (int, optional): Width of PR curve lines. Defaults to 2.
            cutoffs (float | np.ndarray | None, optional): Threshold values for each class. If provided, draws vlines.
            cutoff_line_color (str, optional): Color for cutoff vlines. Defaults to "red".
            cutoff_line_width (int, optional): Width for cutoff vlines. Defaults to 3.
        """
        # Convert inputs to numpy arrays for consistent processing
        y_true = np.array(y_true)
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=1)
        n_classes = probabilities.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        # Handle cutoffs
        if cutoffs is not None:
            if isinstance(cutoffs, float) or isinstance(cutoffs, int):
                cutoffs = np.full(n_classes, float(cutoffs))
            elif isinstance(cutoffs, np.ndarray):
                if cutoffs.shape == ():
                    cutoffs = np.full(n_classes, float(cutoffs))
                elif cutoffs.shape[0] != n_classes:
                    raise ValueError(
                        f"cutoffs shape {cutoffs.shape} does not match number of classes {n_classes}"
                    )
            else:
                raise TypeError("cutoffs must be float, int, or np.ndarray")

        ids = list(id2label.keys())
        labels = list(id2label.values())
        fig = go.Figure()

        # Compute PR curves based on number of classes
        if n_classes == 2:
            precision, recall, avg_precision = self._compute_binary_pr(
                y_true, probabilities, ids
            )
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"Class {labels[1]} (AP = {avg_precision:.3f})",
                    line=dict(width=line_width),
                )
            )
            avg_ap = avg_precision

            # Add cutoff vline if provided
            if cutoffs is not None:
                cutoff = cutoffs[1]
                fig.add_vline(
                    x=cutoff,
                    line_dash="dash",
                    line_color=cutoff_line_color,
                    line_width=cutoff_line_width,
                    annotation_text=f"Cutoff: {cutoff:.2f}",
                    annotation_position="top right",
                    annotation_font_size=font_size,
                    annotation_font_color=cutoff_line_color,
                )
        else:
            precision_list, recall_list, avg_precision_list = (
                self._compute_multiclass_pr(y_true, probabilities, ids)
            )

            for i, (precision, recall, avg_precision) in enumerate(
                zip(precision_list, recall_list, avg_precision_list)
            ):
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=f"Class {labels[i]} (AP = {avg_precision:.3f})",
                        line=dict(
                            width=line_width,
                            color=px.colors.qualitative.Plotly[
                                i % len(px.colors.qualitative.Plotly)
                            ],
                            dash="solid",
                        ),
                    )
                )
                # Add cutoff vline for each class if provided
                if cutoffs is not None:
                    cutoff = cutoffs[i]
                    fig.add_vline(
                        x=cutoff,
                        line_dash="dash",
                        line_color=cutoff_line_color,
                        line_width=cutoff_line_width,
                        annotation_text=f"Cutoff {labels[i]}: {cutoff:.2f}",
                        annotation_position="top right",
                        annotation_font_size=font_size,
                        annotation_font_color=cutoff_line_color,
                    )

            avg_ap = np.mean(avg_precision_list)

        # Generate title
        curve_text = (
            "Precision-Recall Curves" if n_classes > 2 else "Precision-Recall Curve"
        )
        title = (
            f"{plot_title}<br>(Average Precision = {avg_ap:.3f})"
            if plot_title
            else f"{curve_text} (Average Precision = {avg_ap:.3f})"
        )

        # Apply layout and styling
        self.apply_default_layout(fig, title, width, height, xaxis_title, yaxis_title)

        # Update layout specifics
        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=font_size - 2),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.8,
                y=0.99,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )

        # Set axis ranges
        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        fig.show("png", width=width, height=height)

    def plot_precision_vs_recall_curve(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        plot_title: str = "",
        width: int = 1200,
        height: int = 1200,
        xaxis_title: str = "Recall",
        yaxis_title: str = "Precision",
        font_size: int = 22,
        line_width: int = 5,
        cutoffs: float | np.ndarray | None = None,
        cutoff_marker_size: int = 15,
        cutoff_marker_color: str = "red",
    ) -> None:
        """Plots Precision vs Recall curves at different probability thresholds.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities (np.ndarray): Predicted probabilities. For binary: (n_samples,) or (n_samples, 2).
                For multiclass: (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1200.
            height (int, optional): Plot height in pixels. Defaults to 1200.
            xaxis_title (str, optional): X-axis title. Defaults to "Recall".
            yaxis_title (str, optional): Y-axis title. Defaults to "Precision".
            font_size (int, optional): Font size for text elements. Defaults to 22.
            line_width (int, optional): Width of curve lines. Defaults to 5.
            cutoffs (float | np.ndarray | None, optional): Threshold values to mark on curves. Defaults to None.
            cutoff_marker_size (int, optional): Size of cutoff markers. Defaults to 15.
            cutoff_marker_color (str, optional): Color for cutoff markers. Defaults to "red".
        """
        y_true = np.array(y_true)
        probabilities = np.array(probabilities)
        if probabilities.ndim == 1:
            probabilities = np.stack([1 - probabilities, probabilities], axis=1)
        n_classes = probabilities.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        if cutoffs is not None:
            if isinstance(cutoffs, float) or isinstance(cutoffs, int):
                cutoffs = np.full(n_classes, float(cutoffs))
            elif isinstance(cutoffs, np.ndarray):
                if cutoffs.shape == ():
                    cutoffs = np.full(n_classes, float(cutoffs))
                elif cutoffs.shape[0] != n_classes:
                    raise ValueError(
                        f"cutoffs shape {cutoffs.shape} does not match number of classes {n_classes}"
                    )
            else:
                raise TypeError("cutoffs must be float, int, or np.ndarray")

        ids = list(id2label.keys())
        labels = list(id2label.values())
        fig = go.Figure()

        if n_classes == 2:
            precision, recall, avg_precision = self._compute_binary_pr(
                y_true, probabilities, ids
            )
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"Class {labels[1]} (AP = {avg_precision:.3f})",
                    line=dict(width=line_width),
                )
            )
            avg_ap = avg_precision

            if cutoffs is not None:
                cutoff = cutoffs[1]
                precision_cutoff, recall_cutoff, _ = self._compute_binary_pr(
                    y_true, probabilities, ids
                )
                cutoff_idx = np.argmin(np.abs(recall_cutoff - cutoff))
                fig.add_trace(
                    go.Scatter(
                        x=[recall_cutoff[cutoff_idx]],
                        y=[precision_cutoff[cutoff_idx]],
                        mode="markers",
                        marker=dict(
                            size=cutoff_marker_size,
                            color=cutoff_marker_color,
                            symbol="diamond",
                        ),
                        name=f"Cutoff: {cutoff:.2f}",
                        showlegend=True,
                    )
                )
        else:
            precision_list, recall_list, avg_precision_list = (
                self._compute_multiclass_pr(y_true, probabilities, ids)
            )

            for i, (precision, recall, avg_precision) in enumerate(
                zip(precision_list, recall_list, avg_precision_list)
            ):
                color = px.colors.qualitative.Plotly[
                    i % len(px.colors.qualitative.Plotly)
                ]
                fig.add_trace(
                    go.Scatter(
                        x=recall,
                        y=precision,
                        mode="lines",
                        name=f"Class {labels[i]} (AP = {avg_precision:.3f})",
                        line=dict(width=line_width, color=color),
                    )
                )

                if cutoffs is not None:
                    cutoff = cutoffs[i]
                    cutoff_idx = np.argmin(np.abs(recall - cutoff))
                    fig.add_trace(
                        go.Scatter(
                            x=[recall[cutoff_idx]],
                            y=[precision[cutoff_idx]],
                            mode="markers",
                            marker=dict(
                                size=cutoff_marker_size,
                                color=cutoff_marker_color,
                                symbol="diamond",
                            ),
                            name=f"Cutoff {labels[i]}: {cutoff:.2f}",
                            showlegend=False,
                            legendgroup=f"class_{i}",
                        )
                    )

            avg_ap = np.mean(avg_precision_list)

        curve_text = (
            "Precision-Recall Curves" if n_classes > 2 else "Precision-Recall Curve"
        )
        title = (
            f"{plot_title}<br>(Average Precision = {avg_ap:.3f})"
            if plot_title
            else f"{curve_text} (Average Precision = {avg_ap:.3f})"
        )

        self.apply_default_layout(fig, title, width, height, xaxis_title, yaxis_title)

        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=font_size - 2),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=0.8,
                y=0.01,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )

        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        fig.show("png", width=width, height=height)

    def _compute_f1_at_thresholds(
        self,
        y_true: np.ndarray,
        probabilities: np.ndarray,
        id2label: dict[int, str],
        thresholds: np.ndarray = np.linspace(0, 1, 101),
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Computes F1 scores at different probability thresholds for each class.

        Args:
            y_true (np.ndarray): True labels.
            probabilities (np.ndarray): Predicted probabilities (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            thresholds (np.ndarray, optional): Threshold values to evaluate. Defaults to np.linspace(0, 1, 101).

        Returns:
            dict[str, tuple[np.ndarray, np.ndarray]]: Dictionary mapping class names to (thresholds, f1_scores) tuples.
        """
        n_classes = probabilities.shape[1]
        ids = list(id2label.keys())
        labels = list(id2label.values())

        if y_true.dtype != np.str_ and y_true.dtype != np.object_:
            y_true_labels = np.array([id2label[int(label)] for label in y_true])
        else:
            y_true_labels = y_true

        results = {}

        if n_classes == 2:
            f1_scores = []
            for threshold in thresholds:
                y_pred = (probabilities[:, 1] >= threshold).astype(int)
                y_pred_labels = [id2label[int(pred)] for pred in y_pred]
                f1 = f1_score(
                    y_true_labels,
                    y_pred_labels,
                    average="binary",
                    pos_label=labels[1],
                    zero_division=0,
                )
                f1_scores.append(f1)
            results[labels[1]] = (thresholds, np.array(f1_scores))
        else:
            for class_idx, label in enumerate(labels):
                f1_scores = []
                for threshold in thresholds:
                    y_pred = np.argmax(probabilities >= threshold, axis=1)
                    y_pred_labels = [id2label[int(pred)] for pred in y_pred]
                    f1 = f1_score(
                        y_true_labels,
                        y_pred_labels,
                        labels=labels,
                        average=None,
                        zero_division=0,
                    )[class_idx]
                    f1_scores.append(f1)
                results[label] = (thresholds, np.array(f1_scores))

        return results

    def plot_f1_score_vs_threshold(
        self,
        y_true: np.ndarray | pd.Series | list,
        probabilities_1: np.ndarray,
        id2label: dict[int, str],
        probabilities_2: np.ndarray | None = None,
        subset_1_name: str = "Subset 1",
        subset_2_name: str = "Subset 2",
        thresholds: np.ndarray = np.linspace(0, 1, 101),
        plot_title: str = "",
        width: int = 1200,
        height: int = 800,
        xaxis_title: str = "Threshold",
        yaxis_title: str = "F1 Score",
        font_size: int = 22,
        line_width: int = 3,
        optimal_threshold_marker: bool = True,
        marker_size: int = 12,
    ) -> None:
        """Plots F1 scores vs probability thresholds for one or two prediction subsets.

        Args:
            y_true (np.ndarray | pd.Series | list): True labels.
            probabilities_1 (np.ndarray): First set of predicted probabilities (n_samples, n_classes).
            id2label (dict[int, str]): Mapping from class indices to labels.
            probabilities_2 (np.ndarray | None, optional): Second set of predicted probabilities. Defaults to None.
            subset_1_name (str, optional): Name for first subset. Defaults to "Subset 1".
            subset_2_name (str, optional): Name for second subset. Defaults to "Subset 2".
            thresholds (np.ndarray, optional): Threshold values to evaluate. Defaults to np.linspace(0, 1, 101).
            plot_title (str, optional): Custom plot title. Defaults to "".
            width (int, optional): Plot width in pixels. Defaults to 1200.
            height (int, optional): Plot height in pixels. Defaults to 800.
            xaxis_title (str, optional): X-axis title. Defaults to "Threshold".
            yaxis_title (str, optional): Y-axis title. Defaults to "F1 Score".
            font_size (int, optional): Font size for text elements. Defaults to 22.
            line_width (int, optional): Width of curve lines. Defaults to 3.
            optimal_threshold_marker (bool, optional): Whether to mark optimal thresholds. Defaults to True.
            marker_size (int, optional): Size of optimal threshold markers. Defaults to 12.
        """
        y_true = np.array(y_true)
        probabilities_1 = np.array(probabilities_1)
        if probabilities_1.ndim == 1:
            probabilities_1 = np.stack([1 - probabilities_1, probabilities_1], axis=1)

        if probabilities_2 is not None:
            probabilities_2 = np.array(probabilities_2)
            if probabilities_2.ndim == 1:
                probabilities_2 = np.stack(
                    [1 - probabilities_2, probabilities_2], axis=1
                )

        n_classes = probabilities_1.shape[1]

        if n_classes != len(id2label):
            raise ValueError(
                f"Number of classes in probabilities ({n_classes}) does not match length of id2label ({len(id2label)})"
            )

        labels = list(id2label.values())
        fig = go.Figure()

        results_1 = self._compute_f1_at_thresholds(
            y_true, probabilities_1, id2label, thresholds
        )

        color_palette_1 = px.colors.qualitative.Plotly
        color_palette_2 = px.colors.qualitative.Set2

        for idx, (label, (thresh_vals, f1_vals)) in enumerate(results_1.items()):
            color = color_palette_1[idx % len(color_palette_1)]
            fig.add_trace(
                go.Scatter(
                    x=thresh_vals,
                    y=f1_vals,
                    mode="lines",
                    name=f"{subset_1_name} - {label}",
                    line=dict(width=line_width, color=color, dash="solid"),
                    legendgroup=f"subset1_{label}",
                )
            )

            if optimal_threshold_marker:
                optimal_idx = np.argmax(f1_vals)
                optimal_threshold = thresh_vals[optimal_idx]
                optimal_f1 = f1_vals[optimal_idx]
                fig.add_trace(
                    go.Scatter(
                        x=[optimal_threshold],
                        y=[optimal_f1],
                        mode="markers",
                        marker=dict(
                            size=marker_size,
                            color=color,
                            symbol="diamond",
                            line=dict(width=2, color="white"),
                        ),
                        name=f"Optimal {label}: {optimal_threshold:.2f}",
                        legendgroup=f"subset1_{label}",
                        showlegend=True,
                    )
                )

        if probabilities_2 is not None:
            results_2 = self._compute_f1_at_thresholds(
                y_true, probabilities_2, id2label, thresholds
            )

            for idx, (label, (thresh_vals, f1_vals)) in enumerate(results_2.items()):
                color = color_palette_2[idx % len(color_palette_2)]
                fig.add_trace(
                    go.Scatter(
                        x=thresh_vals,
                        y=f1_vals,
                        mode="lines",
                        name=f"{subset_2_name} - {label}",
                        line=dict(width=line_width, color=color, dash="dash"),
                        legendgroup=f"subset2_{label}",
                    )
                )

                if optimal_threshold_marker:
                    optimal_idx = np.argmax(f1_vals)
                    optimal_threshold = thresh_vals[optimal_idx]
                    optimal_f1 = f1_vals[optimal_idx]
                    fig.add_trace(
                        go.Scatter(
                            x=[optimal_threshold],
                            y=[optimal_f1],
                            mode="markers",
                            marker=dict(
                                size=marker_size,
                                color=color,
                                symbol="square",
                                line=dict(width=2, color="white"),
                            ),
                            name=f"Optimal {label}: {optimal_threshold:.2f}",
                            legendgroup=f"subset2_{label}",
                            showlegend=True,
                        )
                    )

        title = plot_title or "F1 Score vs Threshold"

        self.apply_default_layout(fig, title, width, height, xaxis_title, yaxis_title)

        fig.update_layout(
            showlegend=True,
            legend=dict(
                font=dict(size=font_size - 4),
                bordercolor="black",
                borderwidth=1,
                bgcolor="white",
                traceorder="normal",
                x=1.02,
                y=1,
                orientation="v",
                itemclick="toggleothers",
                itemdoubleclick="toggle",
                tracegroupgap=5,
            ),
        )

        fig.update_xaxes(range=[-0.01, 1.01])
        fig.update_yaxes(range=[-0.01, 1.01])

        fig.show("png", width=width, height=height)


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train a classifier
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    probabilities = clf.predict_proba(X_test)

    # Create label mapping
    id2label = {0: "Class_0", 1: "Class_1", 2: "Class_2"}

    # Initialize ClassificationPlots
    plotter = ClassificationPlots()

    # Create confusion matrix
    plotter.plot_confusion_matrix(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        plot_title="Sample Confusion Matrix",
    )

    # Create subplot confusion matrices
    cutoffs = np.array([0.5, 0.5, 0.5])  # Example cutoffs for each class
    plotter.subplot_multilabel_conf_matrix(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        cutoffs=cutoffs,
        plot_title="Sample Subplot Confusion Matrices",
    )

    # Create probability distributions per class
    plotter.plot_probabilities_per_class(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        cutoffs=cutoffs,
        plot_title="Sample Probability Distributions per Class",
    )

    # Create ROC curve
    plotter.plot_roc_auc(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        plot_title="Sample ROC Curves",
    )

    # Create Precision-Recall curve
    plotter.plot_precision_recall(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        plot_title="Sample Precision-Recall Curves",
    )

    # Create Precision vs Recall curve
    plotter.plot_precision_vs_recall_curve(
        y_true=y_test,
        probabilities=probabilities,
        id2label=id2label,
        plot_title="Sample Precision vs Recall Curves",
    )
