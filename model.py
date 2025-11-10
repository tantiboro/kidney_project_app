import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc


def train_model(df: pd.DataFrame, features: list, target: str):
    """
    Trains a logistic regression model and returns all relevant objects
    with consistent keys expected by app.py.
    """
    try:
        X = df[features]
        y = df[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Standardize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the logistic regression model
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # For ROC (only if binary classification)
        labels = np.unique(y)
        is_binary = len(labels) == 2

        roc_data = None
        if is_binary:
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba, pos_label=labels[1])
            roc_auc = auc(fpr, tpr)
            roc_data = (fpr, tpr, roc_auc)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        return {
            "status": "success",
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "roc_curve": roc_data,
            "labels": labels,
            "is_binary": is_binary,
        }

    except Exception as e:
        # Always return a predictable structure
        return {
            "status": "error",
            "message": f"Model training failed: {e}",
        }


def get_prediction(model, scaler, new_data):
    """
    Scales new data and makes a prediction.
    """
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)
    return prediction
