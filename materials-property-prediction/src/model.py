"""Model training and evaluation for band gap prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .utils import MODELS_DIR


DEFAULT_MODEL_PATH = MODELS_DIR / "model.pkl"


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_path: Optional[Path] = None,
    random_state: int = 42,
) -> Tuple[XGBRegressor, Dict[str, float], np.ndarray, np.ndarray]:
    """Train XGBoost regressor and return metrics plus holdout predictions."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=random_state,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    joblib.dump(model, model_path)
    return model, metrics, y_test.to_numpy(), y_pred


def load_model(model_path: Optional[Path] = None) -> XGBRegressor:
    """Load trained model from disk."""
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    return joblib.load(model_path)
