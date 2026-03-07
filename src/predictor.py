"""
predictor.py
-------------
Loads the trained models and exposes prediction functions.
Works correctly regardless of whether the app is run from root,
Streamlit, Docker, or Jenkins.
"""

import joblib
import numpy as np
import pandas as pd
import os

from src.preprocessing import preprocess


# ─────────────────────────────────────────────────────────────
# Resolve absolute path to models directory
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ─────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────
def _load(fname: str):
    path = os.path.join(MODEL_DIR, fname)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Ensure the models directory exists at:\n{MODEL_DIR}"
        )

    return joblib.load(path)


# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
scaler     = _load("scaler.pkl")
elasticnet = _load("elasticnet_model.pkl")
xgb        = _load("xgb_model.pkl")
cat        = _load("cat_model.pkl")
meta_model = _load("meta_model.pkl")


# ─────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────
def predict_price(raw_input_df: pd.DataFrame) -> np.ndarray:
    """
    Predict house price (USD) for given raw input features.
    """

    # Step 1 — preprocessing
    X = preprocess(raw_input_df)

    # Step 2 — scaling (ElasticNet only)
    X_scaled = scaler.transform(X)

    # Step 3 — base models
    pred_enet = elasticnet.predict(X_scaled)
    pred_xgb  = xgb.predict(X)
    pred_cat  = cat.predict(X)

    # Step 4 — stacking
    X_meta   = np.column_stack([pred_enet, pred_xgb, pred_cat])
    pred_log = meta_model.predict(X_meta)

    # Step 5 — reverse log transform
    final_price = np.expm1(pred_log)

    return final_price


# ─────────────────────────────────────────────────────────────
# Prediction with breakdown
# ─────────────────────────────────────────────────────────────
def predict_with_breakdown(raw_input_df: pd.DataFrame) -> dict:
    """
    Predict price and return breakdown of each model.
    """

    X = preprocess(raw_input_df)
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