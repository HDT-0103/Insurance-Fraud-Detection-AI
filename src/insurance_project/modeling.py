from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import auc

def balance_data(x_train, y_train):
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(x_train, y_train)
    print("SMOTE applied. Classes are now balanced.")
    return x_res, y_res

def train_baseline_rf(x_train, y_train, x_test, y_test):
    print("--- Training baseline Random Forest ---")
    rf_model = RandomForestClassifier(random_state=42)

    rf_model.fit(x_train, y_train)
    y_pred = rf_model.predict(x_test)

    recall = recall_score(y_test, y_pred)
    print(f"Baseline recall: {recall:.4f}")
    return rf_model, y_pred

def tune_xgboost (x_train, y_train):
    print("--- Tuning XGBoost hyperparameters ---")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss')

    param_grid = {
        'n_estimators' : [200,400],
        'max_depth': [7,9,11],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='recall',
        n_jobs=1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV recall: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def train_xgboost(x_train, y_train, x_test, y_test):
    print("--- Training XGBoost ---")
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    xgb_model.fit(x_train, y_train)

    y_pred = xgb_model.predict(x_test)

    recall = recall_score(y_test, y_pred)
    print(f"XGBoost recall: {recall:.4f}")
    
    return xgb_model, y_pred

def plot_pr_comparison(models, x_test, y_test):
    plt.figure(figsize=(10,7))
    for name, model in models.items():
        y_probs = model.predict_proba(x_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_auc = auc(recall, precision)

        plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall curve comparison")
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.show()

def plot_pr_comparision(models, x_test, y_test):
    # Backwards-compatible alias (typo in old name)
    return plot_pr_comparison(models, x_test, y_test)
    
def save_model(model, filename: str, models_dir: str | Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    models_path = Path(models_dir) if models_dir is not None else (project_root / "models")
    models_path.mkdir(parents=True, exist_ok=True)

    path = models_path / filename
    joblib.dump(model, path)
    print(f"Model saved to {path}")
    return path

def load_model(filename: str, models_dir: str | Path | None = None):
    project_root = Path(__file__).resolve().parents[1]
    models_path = Path(models_dir) if models_dir is not None else (project_root / "models")
    path = models_path / filename
    return joblib.load(path)

#ASSESSMENT
def evaluate_model_performance(model, x_test, y_test):
    print("--- Performance metrics ---")
    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:,1]

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

    roc_auc = roc_auc_score(y_test, y_probs)
    pr_auc = average_precision_score(y_test, y_probs)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (average precision): {pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=['Normal', 'Fraud'],
        cmap='Blues', ax=ax
    )
    plt.title("Confusion matrix — Fraud detection")
    plt.show()
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}

def plot_all_curves(model, x_test, y_test):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))

    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax1, color='darkorange')
    ax1.plot([0,1], [0,1], "k--", label="Random (AUC = 0.5)")
    ax1.set_title("ROC Curve")
    ax1.legend()

    PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax2, color='Blue')
    ax2.set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.show()

def interpret_with_shap(model, x_train, x_test):
    print("--- 🔍 Interpreting Model with SHAP ---")
    try:
        import shap
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SHAP is required for interpretability. Install it with: `pip install shap`."
        ) from exc
    explainer = shap.TreeExplainer(model)

    shap_values = explainer(x_test)
    plt.figure(figsize=(10,8))
    shap.summary_plot(shap_values, x_test, show=False)
    plt.title("SHAP Summary Plot - Feature Importance")
    plt.show()
    return explainer, shap_values


