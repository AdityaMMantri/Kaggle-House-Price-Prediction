"""
predictor.py
-------------
Loads the four saved artefacts and exposes a single `predict_price` function.
"""

import joblib
import numpy as np
import pandas as pd
import os

from src.preprocessing import preprocess

# ─────────────────────────────────────────────────────────────────────────────
# Load saved models
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

def _load(fname: str):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Make sure the Models_scaler/ directory is present next to app.py."
        )
    return joblib.load(path)


scaler     = _load("scaler.pkl")
elasticnet = _load("elasticnet_model.pkl")
xgb        = _load("xgb_model.pkl")
cat        = _load("cat_model.pkl")
meta_model = _load("meta_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction Function
# ─────────────────────────────────────────────────────────────────────────────
def predict_price(raw_input_df: pd.DataFrame) -> np.ndarray:
    """
    Given a raw input DataFrame (one or more rows of house features),
    return predicted sale prices in USD.

    Parameters
    ----------
    raw_input_df : pd.DataFrame
        Raw feature rows; missing columns are filled with defaults automatically.

    Returns
    -------
    np.ndarray  –  Predicted prices (USD).
    """
    # Step 1 – preprocess
    X = preprocess(raw_input_df)

    # Step 2 – scale (only ElasticNet path)
    X_scaled = scaler.transform(X)

    # Step 3 – base model predictions
    pred_enet = elasticnet.predict(X_scaled)
    pred_xgb  = xgb.predict(X)
    pred_cat  = cat.predict(X)

    # Step 4 – stacking meta-model
    X_meta    = np.column_stack([pred_enet, pred_xgb, pred_cat])
    pred_log  = meta_model.predict(X_meta)

    # Step 5 – reverse log-transform
    final_price = np.expm1(pred_log)
    return final_price


def predict_with_breakdown(raw_input_df: pd.DataFrame) -> dict:
    """
    Like `predict_price` but also returns the individual base-model estimates,
    useful for the 'Model Insights' dashboard tab.

    Returns
    -------
    dict with keys:
        "final"      – stacked ensemble price (USD)
        "elasticnet" – ElasticNet estimate (USD)
        "xgboost"    – XGBoost estimate (USD)
        "catboost"   – CatBoost estimate (USD)
    """
    X        = preprocess(raw_input_df)
    X_scaled = scaler.transform(X)

    pred_enet = elasticnet.predict(X_scaled)
    pred_xgb  = xgb.predict(X)
    pred_cat  = cat.predict(X)

    X_meta   = np.column_stack([pred_enet, pred_xgb, pred_cat])
    pred_log = meta_model.predict(X_meta)

    return {
        "final":      float(np.expm1(pred_log[0])),
        "elasticnet": float(np.expm1(pred_enet[0])),
        "xgboost":    float(np.expm1(pred_xgb[0])),
        "catboost":   float(np.expm1(pred_cat[0])),
    }