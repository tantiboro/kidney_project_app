import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def plot_feature_distribution(df: pd.DataFrame, feature: str, target: str):
    """
    Plots a box plot for a selected feature against the target.
    """
    fig = px.box(
        df,
        x=target,
        y=feature,
        color=target,
        title=f"Distribution of '{feature}' by '{target}'",
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list):
    """
    Plots a correlation heatmap for the selected features.
    """
    if len(features) < 2:
        return go.Figure().update_layout(
            title="Select at least 2 features for a heatmap"
        )

    corr = df[features].corr()
    fig = px.imshow(
        corr, text_auto=True, aspect="auto", title="Feature Correlation Heatmap"
    )
    return fig


def plot_confusion_matrix(cm: np.ndarray, labels: list = None):
    """
    Plots a confusion matrix heatmap.
    """
    if labels is None:
        labels = ["0", "1"]

    # Format for plotly
    z = cm
    x = [f"Predicted {label}" for label in labels]
    y = [f"Actual {label}" for label in labels]

    fig = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z, colorscale="Blues"
    )
    fig.update_layout(title="Confusion Matrix")
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    """
    Plots the Receiver Operating Characteristic (ROC) curve.
    """
    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC = {roc_auc:.2f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=700,
        height=500,
    )
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    return fig
