
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

from ..base_plots import BasePlots


class OtherPlots(BasePlots):
    """
    Class for miscellaneous plotting functionality.
    """

    def feature_importance_plot(
        self,
        importances: dict[str, float] | pd.DataFrame,
        top_n: int | None = None,
        plot_title: str = "Feature Importances",
        width: int = 1200,
        height: int = 800,
        color_scale: str = "Viridis",
        show_values: bool = True,
        max_feature_name_length: int = 50,
    ) -> None:
        """
        Plots feature importances as a horizontal bar chart with gradient colors.

        Args:
            importances (dict | pd.DataFrame): Feature importances as dict or DataFrame.
            top_n (int | None, optional): Show only top N features. Defaults to None.
            plot_title (str, optional): Plot title. Defaults to "Feature Importances".
            width (int, optional): Figure width. Defaults to 1200.
            height (int, optional): Figure height. Defaults to 800.
            color_scale (str, optional): Plotly color scale. Defaults to "Viridis".
            show_values (bool, optional): Show values on bars. Defaults to True.
            max_feature_name_length (int, optional): Max feature name length. Defaults to 50.
        """
        if isinstance(importances, dict):
            df = pd.DataFrame(
                list(importances.items()), columns=["Feature", "Importance"]
            )
        elif isinstance(importances, pd.DataFrame):
            if set(["feature", "importance"]).issubset(importances.columns):
                df = importances.rename(
                    columns={"feature": "Feature", "importance": "Importance"}
                )
            elif set(["Feature", "Importance"]).issubset(importances.columns):
                df = importances.copy()
            else:
                raise ValueError(
                    "DataFrame must have columns ['feature', 'importance'] or "
                    "['Feature', 'Importance']."
                )
        else:
            raise TypeError("importances must be a dict or a pandas DataFrame.")

        df = df.sort_values("Importance", ascending=False)
        if top_n is not None:
            df = df.head(top_n)

        df = df.iloc[::-1]

        max_name_len = max(len(name) for name in df["Feature"])
        dynamic_length = min(max_feature_name_length, max(30, max_name_len))

        truncated_features = [
            (name[:dynamic_length] + "...") if len(name) > dynamic_length else name
            for name in df["Feature"]
        ]

        importance_values = df["Importance"].values
        normalized_values = (importance_values - importance_values.min()) / (
            importance_values.max() - importance_values.min() + 1e-10
        )

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df["Importance"],
                y=truncated_features,
                orientation="h",
                marker=dict(
                    color=normalized_values,
                    colorscale=color_scale,
                    line=dict(color="rgba(50,50,50,0.8)", width=2),
                    colorbar=dict(
                        title=dict(
                            text="Normalized<br>Importance",
                            font=dict(size=12, family="Times New Roman"),
                        ),
                        thickness=20,
                        len=0.75,
                        x=1.02,
                        tickfont=dict(size=10),
                        outlinecolor="rgba(0,0,0,0.3)",
                        outlinewidth=1,
                    ),
                ),
                text=(
                    [f"{val:.2f}" for val in df["Importance"]] if show_values else None
                ),
                textposition="outside",
                textfont=dict(size=11, color="black", family="Times New Roman"),
                hovertemplate=(
                    "<b style='font-size:13px'>%{customdata}</b><br>"
                    "<b>Importance:</b> %{x:.2f}<br>"
                    "<b>Normalized:</b> %{marker.color:.2f}<br>"
                    "<extra></extra>"
                ),
                customdata=df["Feature"].tolist(),
                showlegend=False,
            )
        )

        font_size = max(7, min(13, int(500 / len(df))))

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Importance",
            yaxis_title="Feature",
        )

        fig.update_layout(
            yaxis=dict(
                tickmode="linear",
                tickfont=dict(
                    size=font_size,
                    family="Times New Roman, monospace",
                    color="rgba(0,0,0,0.85)",
                ),
                automargin=True,
                showline=True,
                linewidth=2,
                linecolor="rgba(0,0,0,0.3)",
                gridcolor="rgba(200,200,200,0.3)",
                gridwidth=1,
            ),
            xaxis=dict(
                showgrid=True,
                gridwidth=1.5,
                gridcolor="rgba(150,150,150,0.25)",
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="rgba(0,0,0,0.4)",
                showline=True,
                linewidth=2,
                linecolor="rgba(0,0,0,0.3)",
                tickfont=dict(size=11, family="Times New Roman"),
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hoverlabel=dict(
                bgcolor="white",
                font_size=13,
                font_family="Times New Roman",
                bordercolor="rgba(0,0,0,0.3)",
                align="left",
            ),
            title=dict(
                font=dict(size=18, family="Times New Roman", color="rgba(0,0,0,0.9)"),
                x=0.5,
                xanchor="center",
            ),
            margin=dict(l=20, r=120, t=80, b=60),
        )
        fig.show("png", width=width, height=height)

    def cross_validation_split_plot(
        self,
        x_data: pd.DataFrame | np.ndarray,
        y_data: pd.Series | np.ndarray,
        cv_splitter: Any,
        plot_title: str = "Cross Validation Split",
        width: int = 1200,
        height: int = 800,
    ) -> None:
        """Visualizes cross-validation splits showing train/test indices.

        Args:
            x_data (pd.DataFrame | np.ndarray): Feature data.
            y_data (pd.Series | np.ndarray): Target data.
            cv_splitter: Cross-validation splitter object with split method.
            plot_title (str, optional): Title of the plot. Defaults to "Cross Validation Split".
            width (int, optional): Figure width. Defaults to 1200.
            height (int, optional): Figure height. Defaults to 800.
        """
        x_data = self.check_2d_data(x_data)
        y_data = self.check_data(y_data)

        fig = go.Figure()
        n_samples = len(x_data)

        if n_samples > 100:
            marker_symbol = "line-ns"
            marker_size = min(100 / 10, 10)
            marker_width = min(50 / 10, 2)
        else:
            marker_symbol = "hexagon"
            marker_width = 2
            marker_size = 8

        split_count = 0
        for split_idx, (train_idx, test_idx) in enumerate(
            cv_splitter.split(x_data, y_data)
        ):
            showlegend = split_idx == 0

            fig.add_trace(
                go.Scatter(
                    x=train_idx,
                    y=[split_idx + 1] * len(train_idx),
                    mode="markers",
                    marker_symbol=marker_symbol,
                    marker_color="blue",
                    marker_line_color="blue",
                    marker_line_width=marker_width,
                    marker_size=marker_size,
                    showlegend=showlegend,
                    name="Train",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_idx,
                    y=[split_idx + 1] * len(test_idx),
                    mode="markers",
                    marker_symbol=marker_symbol,
                    marker_color="red",
                    marker_line_color="red",
                    marker_line_width=marker_width,
                    marker_size=marker_size,
                    showlegend=showlegend,
                    name="Test",
                )
            )
            split_count = split_idx + 1

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Sample Indices",
            yaxis_title="Fold Number",
        )
        fig.update_yaxes(
            tickvals=list(range(1, split_count + 1)),
            ticktext=[f"Fold {i}" for i in range(1, split_count + 1)],
        )
        fig.show("png", width=width, height=height)

    def anova_comparison_plot(
        self,
        x_groups: pd.Series | np.ndarray,
        y_values: pd.Series | np.ndarray,
        feature_name: str = "Feature",
        plot_title: str | None = None,
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Creates box plots for comparing feature values across categories.

        Args:
            x_groups (pd.Series | np.ndarray): Categorical feature data.
            y_values (pd.Series | np.ndarray): Continuous target values.
            feature_name (str, optional): Name of the feature. Defaults to "Feature".
            plot_title (str, optional): Title of the plot. Auto-generated if None.
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        x_groups = self.check_data(x_groups)
        y_values = self.check_data(y_values)

        if plot_title is None:
            plot_title = f"Comparison of values for {feature_name} categories"

        fig = go.Figure()
        for category in np.unique(x_groups):
            y_subset = y_values[x_groups == category]
            fig.add_trace(
                go.Box(
                    y=y_subset,
                    name=str(category),
                    marker=dict(line=dict(color="black", width=1)),
                )
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title=feature_name,
            yaxis_title="Values",
        )
        fig.update_layout(showlegend=False)
        fig.show("png", width=width, height=height)

    def feature_ranking_scatter_plot(
        self,
        spearman_ranks: dict[str, float],
        hoeffding_ranks: dict[str, float],
        quantile: float = 0.75,
        plot_title: str = "Feature Ranking Comparison",
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Creates a scatter plot comparing different feature ranking methods.

        Args:
            spearman_ranks (dict): Dictionary with feature names as keys and
                Spearman ranks as values.
            hoeffding_ranks (dict): Dictionary with feature names as keys and
                Hoeffding ranks as values.
            quantile (float, optional): Quantile threshold for highlighting features.
            Defaults to 0.75.
            plot_title (str, optional): Title of the plot. Defaults to "Feature Ranking Comparison".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        spearman_features = set(spearman_ranks)
        common_features = [
            feature for feature in hoeffding_ranks if feature in spearman_features
        ]

        x = np.array([hoeffding_ranks[feature] for feature in common_features])
        y = np.array([spearman_ranks[feature] for feature in common_features])
        x_threshold = np.quantile(x, q=quantile)
        y_threshold = np.quantile(y, q=quantile)

        low_ranks_features = [
            feature
            for feature in common_features
            if hoeffding_ranks[feature] >= x_threshold
            and spearman_ranks[feature] >= y_threshold
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(color="white", size=10, line=dict(color="black", width=1)),
                showlegend=False,
            )
        )

        fig.add_vline(x=x_threshold, line_dash="dash", line_color="red", line_width=2)
        fig.add_hline(y=y_threshold, line_dash="dash", line_color="red", line_width=2)

        for feature in low_ranks_features:
            fig.add_annotation(
                x=hoeffding_ranks[feature],
                y=spearman_ranks[feature],
                text=feature,
                showarrow=False,
                yshift=15,
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Hoeffding Ranks",
            yaxis_title="Spearman Ranks",
        )
        fig.show("png", width=width, height=height)

    def plot_metric_change_vs_features_removed(
        self,
        history: pd.DataFrame,
        optimal_n_features: int | None = None,
        plot_title: str = "Metric value vs Number of Features Removed",
        width: int = 1600,
        height: int = 800,
    ) -> None:
        """Plot the metric trajectory versus number of features removed.

        Args:
            history (pd.DataFrame): History DataFrame produced by run().
            optimal_n_features (int | None, optional): Optional vertical guide for the selected
            count.
            plot_title (str, optional): Title of the plot.
            Defaults to "Metric value vs Number of Features Removed".
            width (int, optional): Figure width. Defaults to 1600.
            height (int, optional): Figure height. Defaults to 800.

        Returns:
            None: This function displays a Plotly figure.
        """
        history = history.sort_values(by=["n_features_removed"]).reset_index(drop=True)
        step_change_indices = (
            history["step"].ne(history["step"].shift())
            if "step" in history.columns
            else [False] * len(history)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["n_features_removed"],
                y=history["metric_value"],
                mode="lines+markers",
                marker=dict(size=10, color="blue", opacity=0.6),
                line=dict(width=2, color="blue"),
                name="Metric Change",
            )
        )
        if step_change_indices is not None and hasattr(history, "loc"):
            fig.add_trace(
                go.Scatter(
                    x=history.loc[step_change_indices, "n_features_removed"],
                    y=history.loc[step_change_indices, "metric_value"],
                    mode="markers",
                    marker=dict(size=18, color="red", opacity=0.9, symbol="circle"),
                    name="Step Change",
                )
            )
        if optimal_n_features is not None:
            fig.add_vline(
                x=history["n_features_removed"].max() - optimal_n_features + 1,
                line_dash="dash",
                line_color="green",
                line_width=3,
                annotation_text=f"Optimal: {optimal_n_features}",
                annotation_position="top right",
                annotation_font_size=20,
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Number of Features Removed",
            yaxis_title="Metric value",
        )
        fig.show("png", width=width, height=height)

    def lda_separation_plot(
        self,
        x_transformed: np.ndarray,
        y_data: pd.Series | np.ndarray,
        plot_title: str = "LDA Separation Plot",
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Visualizes LDA-transformed data showing class separation.

        Args:
            x_transformed (np.ndarray): LDA-transformed feature data.
            y_data (pd.Series | np.ndarray): Class labels.
            plot_title (str, optional): Title of the plot. Defaults to "LDA Separation Plot".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        x_transformed = self.check_2d_data(x_transformed)
        y_data = self.check_data(y_data)

        fig = go.Figure()
        if x_transformed.shape[1] == 1:
            fig.add_trace(
                go.Scatter(
                    x=x_transformed[:, 0],
                    y=np.zeros_like(x_transformed[:, 0]),
                    mode="markers",
                    marker=dict(color=y_data, colorscale="Viridis"),
                    showlegend=False,
                )
            )
            xaxis_title = "LDA Component 1"
            yaxis_title = ""
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_transformed[:, 0],
                    y=x_transformed[:, 1],
                    mode="markers",
                    marker=dict(color=y_data, colorscale="Viridis"),
                    showlegend=False,
                )
            )
            xaxis_title = "LDA Component 1"
            yaxis_title = "LDA Component 2"

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
        fig.show("png", width=width, height=height)

    def pca_explained_variance_plot(
        self,
        fitted_pca: Any,
        plot_title: str = "PCA Explained Variance",
        width: int = 1200,
        height: int = 800,
        variance_threshold: float = 0.8,
    ) -> None:
        """Creates a plot showing individual and cumulative explained variance for PCA components.

        Args:
            fitted_pca: Fitted PCA object with explained_variance_ratio_ attribute.
            plot_title (str, optional): Title of the plot. Defaults to "PCA Explained Variance".
            width (int, optional): Figure width. Auto-calculated if None.
            height (int, optional): Figure height. Auto-calculated if None.
            variance_threshold (float, optional): Threshold line for cumulative variance.
            Defaults to 0.8.
        """
        n_components = len(fitted_pca.explained_variance_ratio_)
        labels = list(range(1, n_components + 1))

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=labels,
                y=fitted_pca.explained_variance_ratio_,
                marker=dict(line=dict(color="black", width=1)),
                name="Individual",
                showlegend=True,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=np.cumsum(fitted_pca.explained_variance_ratio_),
                mode="lines+markers",
                line=dict(color="red", width=2),
                marker=dict(color="red", size=6),
                name="Cumulative",
                showlegend=True,
            )
        )
        fig.add_hline(
            y=variance_threshold,
            line_dash="dash",
            line_color="green",
            line_width=2,
            annotation_text=f"{variance_threshold*100}% variance",
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Principal Component",
            yaxis_title="Explained Variance Ratio",
        )
        fig.update_xaxes(type="category")
        fig.show("png", width=width, height=height)

    def pca_eigenvectors_plot(
        self,
        fitted_pca: Any,
        feature1: pd.Series | np.ndarray,
        feature2: pd.Series | np.ndarray,
        plot_title: str = "PCA Eigenvectors",
        width: int = 800,
        height: int = 800,
        vector_scale: float = 3.0,
    ) -> None:
        """Visualizes PCA eigenvectors overlaid on original 2D data.

        Args:
            fitted_pca: Fitted PCA object.
            feature1 (pd.Series | np.ndarray): First feature values.
            feature2 (pd.Series | np.ndarray): Second feature values.
            plot_title (str, optional): Title of the plot. Defaults to "PCA Eigenvectors".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            vector_scale (float, optional): Scale factor for eigenvector visualization.
            Defaults to 3.0.
        """
        feature1 = self.check_data(feature1)
        feature2 = self.check_data(feature2)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=feature1,
                y=feature2,
                mode="markers",
                marker=dict(color="lightblue", line=dict(color="black", width=1)),
                name="Data Points",
            )
        )
        for i in range(fitted_pca.components_.shape[0]):
            vector_x = (
                fitted_pca.components_[i, 0]
                * np.sqrt(fitted_pca.explained_variance_[i])
                * vector_scale
            )
            vector_y = (
                fitted_pca.components_[i, 1]
                * np.sqrt(fitted_pca.explained_variance_[i])
                * vector_scale
            )

            fig.add_annotation(
                ax=fitted_pca.mean_[0],
                ay=fitted_pca.mean_[1],
                axref="x",
                ayref="y",
                x=fitted_pca.mean_[0] + vector_x,
                y=fitted_pca.mean_[1] + vector_y,
                showarrow=True,
                arrowsize=1,
                arrowhead=2,
                arrowwidth=3,
                arrowcolor="red",
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def scree_plot(
        self,
        fitted_fa: Any,
        plot_title: str = "Scree Plot",
        width: int = 800,
        height: int = 800,
        eigenvalue_threshold: float = 1.0,
    ) -> None:
        """Creates a scree plot for factor analysis showing eigenvalues.

        Args:
            fitted_fa: Fitted factor analysis object with eigenvalues_ attribute.
            plot_title (str, optional): Title of the plot. Defaults to "Scree Plot".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            eigenvalue_threshold (float, optional): Threshold line for eigenvalues. Defaults to 1.0.
        """

        cov_matrix = fitted_fa.get_covariance()
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]

        fig = go.Figure()
        component_numbers = np.arange(1, len(eigenvalues) + 1)

        fig.add_trace(
            go.Scatter(
                x=component_numbers,
                y=eigenvalues,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
                showlegend=False,
            )
        )

        fig.add_hline(
            y=eigenvalue_threshold,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Eigenvalue = {eigenvalue_threshold}",
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Component Number",
            yaxis_title="Eigenvalue",
        )
        fig.show("png", width=width, height=height)

    def pca_inverse_transform_plot(
        self,
        fitted_pca: Any,
        feature1: pd.Series | np.ndarray,
        feature2: pd.Series | np.ndarray,
        feature1_reconstructed: pd.Series | np.ndarray,
        feature2_reconstructed: pd.Series | np.ndarray,
        plot_title: str | None = None,
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Compares original data with PCA-reconstructed data.

        Args:
            fitted_pca: Fitted PCA object.
            feature1 (pd.Series | np.ndarray): Original first feature values.
            feature2 (pd.Series | np.ndarray): Original second feature values.
            feature1_reconstructed (pd.Series | np.ndarray): Reconstructed first feature values.
            feature2_reconstructed (pd.Series | np.ndarray): Reconstructed second feature values.
            plot_title (str, optional): Title of the plot. Auto-generated if None.
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        feature1 = self.check_data(feature1)
        feature2 = self.check_data(feature2)
        feature1_reconstructed = self.check_data(feature1_reconstructed)
        feature2_reconstructed = self.check_data(feature2_reconstructed)

        if plot_title is None:
            explained_var = np.round(fitted_pca.explained_variance_ratio_[0], 4)
            plot_title = (
                f"PCA Inverse Transform<br>Explained variance ratio: {explained_var}"
            )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=feature1,
                y=feature2,
                mode="markers",
                marker=dict(color="lightblue", line=dict(color="black", width=1)),
                name="Original",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=feature1_reconstructed,
                y=feature2_reconstructed,
                mode="markers",
                marker=dict(color="red", line=dict(color="black", width=1)),
                name="Reconstructed",
            )
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def cross_validation_components_plot(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        categorical_features: list[str],
        reduction_algorithm: Any,
        predictive_algorithm: Any,
        cv_splitter: Any,
        random_state: int = 42,
        plot_title: str = "Cross Validation Components Analysis",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        """Analyzes optimal number of components using cross-validation.

        Args:
            x_data (pd.DataFrame): Feature data.
            y_data (pd.Series): Target data.
            categorical_features (list): List of categorical feature names.
            reduction_algorithm: Dimensionality reduction algorithm with n_components parameter.
            predictive_algorithm: Predictive model with fit/predict_proba methods.
            cv_splitter: Cross-validation splitter.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
            plot_title (str, optional): Title of the plot.
            Defaults to "Cross Validation Components Analysis".
            width (int, optional): Figure width. Auto-calculated if None.
            height (int, optional): Figure height. Auto-calculated if None.
        """
        continuous_features = [
            col for col in x_data.columns if col not in categorical_features
        ]
        n_continuous = len(continuous_features)

        if width is None:
            width = max(30 * n_continuous, 800)
        if height is None:
            height = max(30 * n_continuous, 600)

        fig = go.Figure()

        for n_components in range(1, n_continuous + 1):
            reduction_algorithm.set_params(
                n_components=n_components, random_state=random_state
            )
            valid_scores = []

            for train_idx, valid_idx in cv_splitter.split(x_data, y_data):
                x_train, x_valid = x_data.iloc[train_idx, :], x_data.iloc[valid_idx, :]
                y_train, y_valid = y_data.iloc[train_idx], y_data.iloc[valid_idx]

                x_train_continuous = x_train[continuous_features]
                x_valid_continuous = x_valid[continuous_features]

                x_train_transformed = reduction_algorithm.fit_transform(
                    x_train_continuous
                )
                x_valid_transformed = reduction_algorithm.transform(x_valid_continuous)

                if categorical_features:
                    x_train_final = np.concatenate(
                        [x_train_transformed, x_train[categorical_features].values],
                        axis=1,
                    )
                    x_valid_final = np.concatenate(
                        [x_valid_transformed, x_valid[categorical_features].values],
                        axis=1,
                    )
                else:
                    x_train_final = x_train_transformed
                    x_valid_final = x_valid_transformed

                predictive_algorithm.fit(x_train_final, y_train)
                y_valid_prob = predictive_algorithm.predict_proba(x_valid_final)[:, 1]
                valid_scores.append(roc_auc_score(y_valid, y_valid_prob))

            fig.add_trace(
                go.Box(
                    y=valid_scores,
                    name=str(n_components),
                    marker=dict(line=dict(color="black", width=1)),
                    showlegend=False,
                )
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Number of Components",
            yaxis_title="Validation Scores",
        )
        fig.show("png", width=width, height=height)

    def dendrogram_plot(
        self,
        linkage_matrix: np.ndarray,
        labels: list[str] | None = None,
        plot_title: str = "Hierarchical Clustering Dendrogram",
        truncate_mode: str = "level",
        p: int = 5,
        hline_level: float | None = None,
        figsize: tuple[int, int] = (12, 8),
    ) -> None:
        """Creates a dendrogram plot for hierarchical clustering.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
            labels (list, optional): Labels for the data points. Defaults to None.
            plot_title (str, optional): Title of the plot.
            Defaults to "Hierarchical Clustering Dendrogram".
            truncate_mode (str, optional): Truncation mode for large dendrograms.
            Defaults to "level".
            p (int, optional): Truncation parameter. Defaults to 5.
            hline_level (float, optional): Horizontal line level for cluster cutoff.
            Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (12, 8).
        """
        plt.figure(figsize=figsize)
        plt.title(plot_title, fontsize=14)
        plt.ylabel("Distance", fontsize=12)

        dendrogram(
            linkage_matrix,
            truncate_mode=truncate_mode,
            p=p,
            labels=labels,
            leaf_rotation=80,
            leaf_font_size=12,
            show_contracted=True,
        )

        if hline_level is not None:
            plt.axhline(y=hline_level, color="r", linestyle="--", linewidth=2)

        plt.tight_layout()
        plt.show()

    def elbow_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        clustering_algorithm: Any,
        max_clusters: int = 15,
        plot_title: str = "Elbow Method",
        width: int = 800,
        height: int = 800,
        show_optimal: bool = True,
    ) -> None:
        """Creates an elbow plot for determining optimal number of clusters.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            clustering_algorithm: Clustering algorithm with n_clusters parameter.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 15.
            plot_title (str, optional): Title of the plot. Defaults to "Elbow Method".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            show_optimal (bool, optional): Whether to show optimal k line. Defaults to True.
        """
        data = self.check_2d_data(data)
        sse_values = []
        k_range = range(1, max_clusters + 1)

        for k in k_range:
            clustering_algorithm.set_params(n_clusters=k)
            clustering_algorithm.fit(data)

            sse = 0
            for cluster_id in np.unique(clustering_algorithm.labels_):
                cluster_data = data[clustering_algorithm.labels_ == cluster_id]
                cluster_center = np.mean(cluster_data, axis=0)
                sse += np.sum((cluster_data - cluster_center) ** 2)

            sse_values.append(sse)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=sse_values,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
                showlegend=False,
            )
        )

        if show_optimal:

            optimal_k = None
            total_sse = np.sum(sse_values)
            for i in range(1, len(sse_values) - 1):
                if sse_values[i - 1] - sse_values[i + 1] < 0.05 * total_sse:
                    optimal_k = i + 1
                    break

            if optimal_k:
                fig.add_vline(
                    x=optimal_k,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Optimal k = {optimal_k}",
                )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Sum of Squared Errors (SSE)",
        )
        fig.update_yaxes(rangemode="tozero")
        fig.show("png", width=width, height=height)

    def silhouette_analysis_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        clustering_algorithm: Any,
        max_clusters: int = 10,
        plot_title: str = "Silhouette Analysis",
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Creates a silhouette analysis plot for determining optimal number of clusters.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            clustering_algorithm: Clustering algorithm with n_clusters parameter.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 10.
            plot_title (str, optional): Title of the plot. Defaults to "Silhouette Analysis".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        data = self.check_2d_data(data)
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        optimal_k = None
        best_score = -1

        for k in k_range:
            clustering_algorithm.set_params(n_clusters=k)
            clustering_algorithm.fit(data)

            score = np.mean(
                self._calculate_silhouette_coefficient(data, clustering_algorithm)
            )
            silhouette_scores.append(score)

            if score > best_score:
                best_score = score
                optimal_k = k

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=silhouette_scores,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
                showlegend=False,
            )
        )

        if optimal_k:
            fig.add_vline(
                x=optimal_k,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Optimal k = {optimal_k}<br>Score = {best_score:.3f}",
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Average Silhouette Score",
        )
        fig.show("png", width=width, height=height)

    def cluster_visualization_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        fitted_model: Any,
        plot_title: str = "Cluster Visualization",
        width: int = 800,
        height: int = 800,
        show_centers: bool = True,
    ) -> None:
        """Visualizes clustered 2D data with cluster assignments.

        Args:
            data (pd.DataFrame | np.ndarray): 2D input data.
            fitted_model: Fitted clustering model with labels_ attribute.
            plot_title (str, optional): Title of the plot. Defaults to "Cluster Visualization".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            show_centers (bool, optional): Whether to show cluster centers if available.
            Defaults to True.
        """
        data = self.check_2d_data(data)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode="markers",
                marker=dict(
                    color=fitted_model.labels_,
                    colorscale="Viridis",
                    line=dict(color="black", width=1),
                    size=8,
                ),
                name="Data Points",
                showlegend=False,
            )
        )

        if show_centers and hasattr(fitted_model, "cluster_centers_"):
            fig.add_trace(
                go.Scatter(
                    x=fitted_model.cluster_centers_[:, 0],
                    y=fitted_model.cluster_centers_[:, 1],
                    mode="markers",
                    marker=dict(
                        color="red",
                        symbol="x",
                        size=15,
                        line=dict(color="black", width=2),
                    ),
                    name="Cluster Centers",
                )
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def k_distance_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        min_points: int,
        plot_title: str = "K-Distance Plot",
        width: int = 800,
        height: int = 800,
        normalized: bool = False,
    ) -> None:
        """Creates a k-distance plot for DBSCAN parameter selection.

        Args:
            data (pd.DataFrame | np.ndarray): Input data.
            min_points (int): Number of neighbors to consider.
            plot_title (str, optional): Title of the plot. Defaults to "K-Distance Plot".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            normalized (bool, optional): Whether to normalize the plot. Defaults to False.
        """
        data = self.check_2d_data(data)

        model = NearestNeighbors(n_neighbors=min_points, metric="euclidean")
        model.fit(data)
        distances, _ = model.kneighbors(data)
        distances = distances[:, 1:]

        mean_distances = np.mean(distances, axis=1)
        sorted_distances = np.sort(mean_distances)[::-1]

        if normalized:

            sorted_distances = (sorted_distances - np.min(sorted_distances)) / (
                np.max(sorted_distances) - np.min(sorted_distances)
            )
            data_range = np.arange(len(sorted_distances))
            data_range = (data_range - np.min(data_range)) / (
                np.max(data_range) - np.min(data_range)
            )
            x_data = data_range
            y_data = sorted_distances
            xaxis_title = "Normalized Sorted Indices"
            yaxis_title = "Normalized Mean Distance"
        else:
            x_data = np.arange(len(sorted_distances))
            y_data = sorted_distances
            xaxis_title = "Sorted Indices"
            yaxis_title = "Mean Distance to Neighbors"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines",
                line=dict(color="blue", width=2),
                name="K-Distance Curve",
            )
        )

        if normalized:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[1, 0],
                    mode="lines",
                    line=dict(color="green", dash="dash", width=2),
                    name="Diagonal Reference",
                )
            )
            fig.update_xaxes(range=[0, 1.05])
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0, len(sorted_distances)],
                    y=[np.max(sorted_distances), np.min(sorted_distances)],
                    mode="lines",
                    line=dict(color="green", dash="dash", width=2),
                    name="Diagonal Reference",
                )
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
        )
        fig.update_yaxes(rangemode="tozero")
        fig.show("png", width=width, height=height)

    def mahalanobis_anomaly_plot(
        self,
        x_data: pd.DataFrame | np.ndarray,
        n_outliers: int = 0,
        alpha: float = 0.05,
        plot_title: str = "Mahalanobis Distance Anomaly Detection",
        width: int = 1000,
        height: int = 1000,
    ) -> None:
        """Creates a Mahalanobis distance-based anomaly detection plot.

        Args:
            x_data (pd.DataFrame | np.ndarray): Input 2D data.
            n_outliers (int, optional): Number of artificial outliers to add. Defaults to 0.
            alpha (float, optional): Significance level for threshold. Defaults to 0.05.
            plot_title (str, optional): Title of the plot.
            Defaults to "Mahalanobis Distance Anomaly Detection".
            width (int, optional): Figure width. Defaults to 1000.
            height (int, optional): Figure height. Defaults to 1000.
        """
        x_data = self.check_2d_data(x_data)

        if n_outliers > 0:
            outliers = np.random.uniform(
                low=np.min(x_data) - 3 * np.std(x_data),
                high=np.max(x_data) + 3 * np.std(x_data),
                size=(n_outliers, x_data.shape[1]),
            )
            x_with_outliers = np.concatenate([x_data, outliers], axis=0)
        else:
            x_with_outliers = x_data

        empirical_cov = EmpiricalCovariance().fit(x_with_outliers)

        x_min, x_max = np.min(x_with_outliers), np.max(x_with_outliers)
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100)
        )
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        mahal_distances = empirical_cov.mahalanobis(mesh_points) ** 0.5
        mahal_distances = mahal_distances.reshape(xx.shape)

        threshold = np.sqrt(chi2.ppf(1 - alpha, df=x_data.shape[1]))

        fig = go.Figure()

        if n_outliers > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Inliers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_with_outliers[-n_outliers:, 0],
                    y=x_with_outliers[-n_outliers:, 1],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Outliers",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Data Points",
                )
            )

        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=mahal_distances,
                contours_coloring="lines",
                showscale=True,
                opacity=0.5,
                line=dict(width=2, dash="dash"),
                colorscale="Jet",
                colorbar=dict(title="Mahalanobis<br>Distance", x=1.02),
            )
        )

        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=mahal_distances,
                contours_coloring="lines",
                showscale=False,
                opacity=1,
                line=dict(width=3, color="black"),
                contours=dict(start=threshold, end=threshold, size=1),
                name=f"Threshold (α={alpha})",
            )
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def isolation_forest_plot(
        self,
        x_data: pd.DataFrame | np.ndarray,
        n_outliers: int = 0,
        contamination: float = 0.05,
        plot_title: str = "Isolation Forest Anomaly Detection",
        width: int = 1000,
        height: int = 1000,
        random_state: int = 42,
    ) -> None:
        """Creates an Isolation Forest anomaly detection visualization.

        Args:
            x_data (pd.DataFrame | np.ndarray): Input 2D data.
            n_outliers (int, optional): Number of artificial outliers to add. Defaults to 0.
            contamination (float, optional): Expected proportion of outliers. Defaults to 0.05.
            plot_title (str, optional): Title of the plot.
            Defaults to "Isolation Forest Anomaly Detection".
            width (int, optional): Figure width. Defaults to 1000.
            height (int, optional): Figure height. Defaults to 1000.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        x_data = self.check_2d_data(x_data)

        if n_outliers > 0:
            outliers = np.random.uniform(
                low=np.min(x_data) - 3 * np.std(x_data),
                high=np.max(x_data) + 3 * np.std(x_data),
                size=(n_outliers, x_data.shape[1]),
            )
            x_with_outliers = np.concatenate([x_data, outliers], axis=0)
        else:
            x_with_outliers = x_data

        isolation_forest = IsolationForest(
            random_state=random_state, contamination=contamination
        )
        isolation_forest.fit(x_with_outliers)

        x_min, x_max = np.min(x_with_outliers), np.max(x_with_outliers)
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100)
        )
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        anomaly_scores = isolation_forest.decision_function(mesh_points)
        predictions = isolation_forest.predict(mesh_points)

        fig = go.Figure()

        if n_outliers > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Inliers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_with_outliers[-n_outliers:, 0],
                    y=x_with_outliers[-n_outliers:, 1],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Outliers",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Data Points",
                )
            )

        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=anomaly_scores.reshape(xx.shape),
                contours_coloring="lines",
                showscale=False,
                opacity=0.5,
                line=dict(width=2, dash="dash"),
                colorscale="RdYlBu",
            )
        )

        fig.add_trace(
            go.Contour(
                x=xx[0],
                y=yy[:, 0],
                z=predictions.reshape(xx.shape),
                contours_coloring="lines",
                showscale=False,
                opacity=1,
                line=dict(width=3, color="black"),
                name=f"Decision Boundary",
            )
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def local_outlier_factor_plot(
        self,
        x_data: pd.DataFrame | np.ndarray,
        n_outliers: int = 0,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        plot_title: str = "Local Outlier Factor Anomaly Detection",
        width: int = 1000,
        height: int = 1000,
    ) -> None:
        """Creates a Local Outlier Factor anomaly detection visualization.

        Args:
            x_data (pd.DataFrame | np.ndarray): Input 2D data.
            n_outliers (int, optional): Number of artificial outliers to add. Defaults to 0.
            contamination (float, optional): Expected proportion of outliers. Defaults to 0.05.
            n_neighbors (int, optional): Number of neighbors for LOF calculation. Defaults to 20.
            plot_title (str, optional): Title of the plot.
            Defaults to "Local Outlier Factor Anomaly Detection".
            width (int, optional): Figure width. Defaults to 1000.
            height (int, optional): Figure height. Defaults to 1000.
        """
        x_data = self.check_2d_data(x_data)

        if n_outliers > 0:
            outliers = np.random.uniform(
                low=np.min(x_data) - 3 * np.std(x_data),
                high=np.max(x_data) + 3 * np.std(x_data),
                size=(n_outliers, x_data.shape[1]),
            )
            x_with_outliers = np.concatenate([x_data, outliers], axis=0)
        else:
            x_with_outliers = x_data

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        lof.fit(x_with_outliers)
        scores = lof.negative_outlier_factor_

        radius = (scores.max() - scores) / (scores.max() - scores.min())

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=x_with_outliers[:, 0],
                y=x_with_outliers[:, 1],
                mode="markers",
                marker=dict(
                    color="white",
                    size=radius * 100,
                    line=dict(color="black", width=2),
                ),
                name="LOF Scores",
                showlegend=True,
            )
        )

        if n_outliers > 0:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Inliers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_with_outliers[-n_outliers:, 0],
                    y=x_with_outliers[-n_outliers:, 1],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="Outliers",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x_data[:, 0],
                    y=x_data[:, 1],
                    mode="markers",
                    marker=dict(color="blue", size=8),
                    name="Data Points",
                )
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Feature 1",
            yaxis_title="Feature 2",
        )
        fig.show("png", width=width, height=height)

    def difference_curve(
        self,
        data: pd.DataFrame | np.ndarray,
        algorithm_instance: Any,
        max_clusters: int = 15,
        width: int = 1200,
        height: int = 800,
    ) -> None:
        """
        Creates a difference curve plot for clustering algorithms.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            algorithm_instance: Clustering algorithm instance with n_clusters parameter.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 15.
            width (int, optional): Figure width. Defaults to 1200.
            height (int, optional): Figure height. Defaults to 800.
        """
        data = self.check_2d_data(data=data)
        sse = []
        for k in range(1, max_clusters + 1):
            algorithm_instance.set_params(n_clusters=k)
            algorithm_instance.fit(data)
            sse_data = 0
            for cluster in np.unique(algorithm_instance.labels_):
                x_cluster = data[np.where(algorithm_instance.labels_ == cluster)]
                center_of_cluster = np.mean(x_cluster, axis=0)
                sse_data += np.linalg.norm(x_cluster - center_of_cluster) ** 2
            sse.append(sse_data)
        scaled_sse = (sse - np.min(sse)) / (np.max(sse) - np.min(sse))
        cluster_range = [k for k in range(1, max_clusters + 1)]
        scaled_cluster_range = (cluster_range - np.min(cluster_range)) / (
            np.max(cluster_range) - np.min(cluster_range)
        )
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=scaled_cluster_range,
                y=scaled_sse,
                mode="lines+markers",
                name="Normalized curve",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=scaled_cluster_range,
                y=(1 - scaled_cluster_range) - scaled_sse,
                mode="lines+markers",
                line=dict(color="orange"),
                name="Difference curve",
            )
        )
        maximum_indice = np.argmax((1 - scaled_cluster_range) - scaled_sse)
        optimal_sse = sse[maximum_indice]
        optimal_k = cluster_range[maximum_indice]
        fig.add_vline(
            x=scaled_cluster_range[maximum_indice],
            line_dash="dash",
            line_color="red",
            line_width=2,
        )
        fig.update_yaxes(rangemode="tozero")
        fig.update_xaxes(range=[0, 1.05], constrain="domain", linecolor="black")
        fig.update_layout(
            template="simple_white",
            width=600,
            height=600,
            xaxis_title="Normalized number of clusters (k)",
            yaxis_title="Normalized distortion Score",
            legend=dict(x=0.75, y=0.9),
            showlegend=True,
            title=(
                f"<b>Difference curve</b><br>Optimal SSE: "
                f"{np.round(optimal_sse, 4)} for k={optimal_k}"
            ),
            title_x=0.5,
            font=dict(family="Times New Roman", size=16, color="Black"),
        )
        fig.show("png", width=width, height=height)

    def normalized_elbow_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        clustering_algorithm: Any,
        max_clusters: int = 15,
        plot_title: str = "Normalized Elbow Method",
        width: int = 800,
        height: int = 800,
        show_perpendicular: bool = False,
        show_difference_curve: bool = False,
    ) -> None:
        """Creates a normalized elbow plot with optional analysis curves.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            clustering_algorithm: Clustering algorithm with n_clusters parameter.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 15.
            plot_title (str, optional): Title of the plot. Defaults to "Normalized Elbow Method".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            show_perpendicular (bool, optional): Whether to show perpendicular lines.
            Defaults to False.
            show_difference_curve (bool, optional): Whether to show difference curve.
            Defaults to False.
        """
        data = self.check_2d_data(data)
        sse_values = []
        k_range = range(1, max_clusters + 1)

        for k in k_range:
            clustering_algorithm.set_params(n_clusters=k)
            clustering_algorithm.fit(data)

            sse = 0
            for cluster_id in np.unique(clustering_algorithm.labels_):
                cluster_data = data[clustering_algorithm.labels_ == cluster_id]
                cluster_center = np.mean(cluster_data, axis=0)
                sse += np.sum((cluster_data - cluster_center) ** 2)

            sse_values.append(sse)

        sse_normalized = (sse_values - np.min(sse_values)) / (
            np.max(sse_values) - np.min(sse_values)
        )
        k_normalized = (np.array(list(k_range)) - min(k_range)) / (
            max(k_range) - min(k_range)
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=k_normalized,
                y=sse_normalized,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
                name="Normalized Curve",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[1, 0],
                mode="lines",
                line=dict(color="green", dash="dash", width=2),
                name="Diagonal Line",
            )
        )

        if show_perpendicular:
            for i, (k_value, sse_value) in enumerate(
                zip(k_normalized, sse_normalized)
            ):
                x_perp = (k_value + sse_value - 1) / 2
                y_perp = 1 - x_perp

                fig.add_trace(
                    go.Scatter(
                        x=[k_value, x_perp],
                        y=[sse_value, y_perp],
                        mode="lines",
                        line=dict(color="gray", width=1),
                        showlegend=i == 0,
                        name="Perpendicular Lines" if i == 0 else None,
                    )
                )

        if show_difference_curve:
            difference_curve = (1 - k_normalized) - sse_normalized
            optimal_idx = np.argmax(difference_curve)

            fig.add_trace(
                go.Scatter(
                    x=k_normalized,
                    y=difference_curve,
                    mode="lines+markers",
                    line=dict(color="orange", width=2),
                    marker=dict(color="orange", size=6),
                    name="Difference Curve",
                )
            )

            fig.add_vline(
                x=k_normalized[optimal_idx],
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Optimal k = {list(k_range)[optimal_idx]}",
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Normalized Number of Clusters",
            yaxis_title="Normalized SSE",
        )
        fig.update_xaxes(range=[0, 1.05])
        fig.update_yaxes(rangemode="tozero")
        fig.show("png", width=width, height=height)

    def gap_statistic_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        clustering_algorithm: Any,
        max_clusters: int = 15,
        n_reference_datasets: int = 30,
        plot_title: str = "Gap Statistic",
        width: int = 800,
        height: int = 800,
        random_state: int = 42,
    ) -> None:
        """Creates a gap statistic plot for determining optimal number of clusters.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            clustering_algorithm: Clustering algorithm with n_clusters parameter.
            max_clusters (int, optional): Maximum number of clusters to test. Defaults to 15.
            n_reference_datasets (int, optional): Number of reference datasets to generate.
            Defaults to 30.
            plot_title (str, optional): Title of the plot. Defaults to "Gap Statistic".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
            random_state (int, optional): Random state for reproducibility. Defaults to 42.
        """
        data = self.check_2d_data(data)
        np.random.seed(random_state)

        gaps = []
        stds = []
        k_range = range(1, max_clusters + 1)
        optimal_k = None

        data_ranges = np.array(
            [[np.min(data[:, i]), np.max(data[:, i])] for i in range(data.shape[1])]
        )

        for k in k_range:

            reference_sses = []
            for _ in range(n_reference_datasets):
                reference_data = np.random.uniform(
                    low=data_ranges[:, 0],
                    high=data_ranges[:, 1],
                    size=data.shape,
                )

                clustering_algorithm.set_params(n_clusters=k)
                clustering_algorithm.fit(reference_data)

                sse = 0
                for cluster_id in np.unique(clustering_algorithm.labels_):
                    cluster_data = reference_data[
                        clustering_algorithm.labels_ == cluster_id
                    ]
                    cluster_center = np.mean(cluster_data, axis=0)
                    sse += np.sum((cluster_data - cluster_center) ** 2)

                reference_sses.append(sse)

            clustering_algorithm.set_params(n_clusters=k)
            clustering_algorithm.fit(data)

            data_sse = 0
            for cluster_id in np.unique(clustering_algorithm.labels_):
                cluster_data = data[clustering_algorithm.labels_ == cluster_id]
                cluster_center = np.mean(cluster_data, axis=0)
                data_sse += np.sum((cluster_data - cluster_center) ** 2)

            gap = np.log(np.mean(reference_sses)) - np.log(data_sse)
            gaps.append(gap)

            std = np.std(np.log(reference_sses))
            stds.append(std)

            if k > 1 and optimal_k is None:
                if gaps[k - 2] >= gaps[k - 1] - stds[k - 1]:
                    optimal_k = k - 1

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(k_range),
                y=gaps,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="blue", size=6, line=dict(color="black", width=1)),
                error_y=dict(type="data", array=stds, visible=True),
                name="Gap Statistic",
                showlegend=False,
            )
        )

        if optimal_k:
            fig.add_vline(
                x=optimal_k,
                line_dash="dash",
                line_color="red",
                line_width=2,
                annotation_text=f"Optimal k = {optimal_k}<br>Gap = {gaps[optimal_k-1]:.3f}",
            )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Gap Statistic",
        )
        fig.show("png", width=width, height=height)

    def silhouette_detailed_plot(
        self,
        data: pd.DataFrame | np.ndarray,
        clustering_algorithm: Any,
        n_clusters: int,
        plot_title: str | None = None,
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Creates a detailed silhouette plot for a specific number of clusters.

        Args:
            data (pd.DataFrame | np.ndarray): Input data for clustering.
            clustering_algorithm: Clustering algorithm with n_clusters parameter.
            n_clusters (int): Number of clusters to analyze.
            plot_title (str, optional): Title of the plot. Auto-generated if None.
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        data = self.check_2d_data(data)

        clustering_algorithm.set_params(n_clusters=n_clusters)
        clustering_algorithm.fit(data)

        silhouette_coefficients = self._calculate_silhouette_coefficient(
            data, clustering_algorithm
        )
        average_silhouette = np.mean(silhouette_coefficients)

        if plot_title is None:
            plot_title = (
                f"Silhouette Analysis (k={n_clusters})<br>"
                f"Average Score: {average_silhouette:.3f}"
            )

        fig = go.Figure()

        y_lower = 0
        colors = self._get_colors(n_clusters)

        for i, cluster_id in enumerate(np.unique(clustering_algorithm.labels_)):
            cluster_indices = clustering_algorithm.labels_ == cluster_id
            cluster_silhouettes = np.sort(silhouette_coefficients[cluster_indices])

            y_upper = y_lower + len(cluster_silhouettes)

            fig.add_trace(
                go.Scatter(
                    x=cluster_silhouettes,
                    y=np.arange(y_lower, y_upper),
                    fill="tozerox",
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=0),
                    fillcolor=colors[i % len(colors)],
                    name=f"Cluster {cluster_id}",
                    showlegend=False,
                )
            )

            fig.add_annotation(
                x=0.05,
                y=(y_lower + y_upper) / 2,
                text=str(cluster_id),
                showarrow=False,
                font=dict(size=16),
            )

            y_lower = y_upper + 10

        fig.add_vline(
            x=average_silhouette,
            line_dash="dash",
            line_color="red",
            line_width=2,
            annotation_text=f"Average: {average_silhouette:.3f}",
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Silhouette Coefficient",
            yaxis_title="Cluster",
        )
        fig.update_yaxes(showticklabels=False, visible=False)
        fig.show("png", width=width, height=height)

    def mahalanobis_distance_plot(
        self,
        x_data: pd.DataFrame | np.ndarray,
        alpha: float = 0.05,
        plot_title: str = "Mahalanobis Distance Distribution",
        width: int = 800,
        height: int = 800,
    ) -> None:
        """Creates a plot showing the distribution of Mahalanobis distances.

        Args:
            x_data (pd.DataFrame | np.ndarray): Input data.
            alpha (float, optional): Significance level for threshold. Defaults to 0.05.
            plot_title (str, optional): Title of the plot.
            Defaults to "Mahalanobis Distance Distribution".
            width (int, optional): Figure width. Defaults to 800.
            height (int, optional): Figure height. Defaults to 800.
        """
        x_data = self.check_2d_data(x_data)

        empirical_cov = EmpiricalCovariance().fit(x_data)
        mahal_distances = empirical_cov.mahalanobis(x_data) ** 0.5
        mahal_distances_sorted = np.sort(mahal_distances)

        threshold = np.sqrt(chi2.ppf(1 - alpha, df=x_data.shape[1]))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(mahal_distances_sorted)),
                y=mahal_distances_sorted,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(color="red", size=6),
                showlegend=False,
            )
        )

        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="black",
            line_width=2,
            annotation_text=f"Threshold (α={alpha}): {threshold:.3f}",
        )

        self.apply_default_layout(
            fig=fig,
            plot_title=plot_title,
            width=width,
            height=height,
            xaxis_title="Sample Index (sorted)",
            yaxis_title="Mahalanobis Distance",
        )
        fig.show("png", width=width, height=height)

    def _calculate_silhouette_coefficient(
        self,
        data: np.ndarray,
        clustering_algorithm: Any,
    ) -> np.ndarray:
        """Calculates silhouette coefficients for clustered data.

        Args:
            data (np.ndarray): Input data.
            clustering_algorithm: Fitted clustering algorithm with labels_ attribute.

        Returns:
            np.ndarray: Silhouette coefficients for each data point.
        """
        labels = clustering_algorithm.labels_
        silhouette_coefficients = np.zeros(data.shape[0])

        for cluster_id in np.unique(labels):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_data = data[cluster_indices]

            if len(cluster_data) == 1:
                silhouette_coefficients[cluster_indices] = 0
                continue

            intra_distances = np.sqrt(
                np.sum((cluster_data[:, np.newaxis] - cluster_data) ** 2, axis=2)
            )
            a_i = np.sum(intra_distances, axis=1) / (len(cluster_data) - 1)

            min_b_i = np.full(len(cluster_data), np.inf)

            for other_cluster_id in np.unique(labels):
                if other_cluster_id != cluster_id:
                    other_cluster_data = data[labels == other_cluster_id]
                    inter_distances = np.sqrt(
                        np.sum(
                            (cluster_data[:, np.newaxis] - other_cluster_data) ** 2,
                            axis=2,
                        )
                    )
                    b_i = np.mean(inter_distances, axis=1)
                    min_b_i = np.minimum(min_b_i, b_i)

            max_ab = np.maximum(a_i, min_b_i)
            silhouette_coefficients[cluster_indices] = (min_b_i - a_i) / max_ab

        return silhouette_coefficients


if __name__ == "__main__":
    plotter = OtherPlots()

    np.random.seed(42)

    X_clusters, y_clusters = make_blobs(
        n_samples=300, centers=4, n_features=2, random_state=42, cluster_std=1.5
    )

    X_class, y_class = make_classification(
        n_samples=200,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        random_state=42,
        n_clusters_per_class=1,
    )

    print("=== ALGORITHM PLOTS EXAMPLES ===")

    print("\n2. Creating cross-validation split plot...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    plotter.cross_validation_split_plot(
        X_class, y_class, cv, plot_title="5-Fold Cross Validation"
    )

    print("\n3. Creating PCA explained variance plot...")
    pca = PCA()
    pca.fit(X_class)
    plotter.pca_explained_variance_plot(pca, plot_title="PCA Analysis")

    print("\n4. Creating LDA separation plot...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(X_class, y_class)
    plotter.lda_separation_plot(X_lda, y_class, plot_title="LDA Separation")

    print("\n5. Creating elbow plot...")
    kmeans = KMeans(random_state=42)
    plotter.elbow_plot(
        X_clusters, kmeans, max_clusters=10, plot_title="K-Means Elbow Method"
    )

    print("\n6. Creating normalized elbow plot with difference curve...")
    plotter.normalized_elbow_plot(
        X_clusters,
        kmeans,
        max_clusters=10,
        plot_title="Normalized Elbow with Difference Curve",
        show_difference_curve=True,
    )

    print("\n7. Creating silhouette analysis plot...")
    plotter.silhouette_analysis_plot(
        X_clusters, kmeans, max_clusters=10, plot_title="Silhouette Analysis"
    )

    print("\n8. Creating detailed silhouette plot...")
    plotter.silhouette_detailed_plot(
        X_clusters,
        kmeans,
        n_clusters=4,
        plot_title="Detailed Silhouette Analysis (k=4)",
    )

    print("\n9. Creating cluster visualization...")
    kmeans_fitted = KMeans(n_clusters=4, random_state=42)
    kmeans_fitted.fit(X_clusters)
    plotter.cluster_visualization_plot(
        X_clusters, kmeans_fitted, plot_title="K-Means Clustering Result"
    )

    print("\n10. Creating k-distance plot...")
    plotter.k_distance_plot(
        X_clusters, min_points=5, plot_title="K-Distance Plot for DBSCAN"
    )

    print("\n11. Creating Mahalanobis anomaly detection plot...")
    plotter.mahalanobis_anomaly_plot(
        X_clusters,
        n_outliers=20,
        alpha=0.05,
        plot_title="Mahalanobis Distance Anomaly Detection",
    )

    print("\n12. Creating Isolation Forest plot...")
    plotter.isolation_forest_plot(
        X_clusters,
        n_outliers=20,
        contamination=0.1,
        plot_title="Isolation Forest Anomaly Detection",
    )

    print("\n13. Creating feature ranking comparison...")

    spearman_ranks = {
        "feature_1": 0.8,
        "feature_2": 0.6,
        "feature_3": 0.9,
        "feature_4": 0.4,
    }
    hoeffding_ranks = {
        "feature_1": 0.7,
        "feature_2": 0.8,
        "feature_3": 0.85,
        "feature_4": 0.3,
    }
    plotter.feature_ranking_scatter_plot(
        spearman_ranks, hoeffding_ranks, plot_title="Feature Ranking Comparison"
    )

    print("\n=== All plots created successfully! ===")
