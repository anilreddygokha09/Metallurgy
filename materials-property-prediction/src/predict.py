"""Inference utilities for new materials compositions."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pymatgen.core import Composition

from .featurization import featurize_new_compositions
from .model import load_model


def _is_valid(formula: str) -> bool:
    try:
        Composition(formula)
        return True
    except Exception:
        return False


def predict_band_gap_for_compositions(compositions: Iterable[str]) -> pd.DataFrame:
    """Predict band gap values for new composition strings."""
    provided = list(compositions)
    valid_compositions = [c for c in provided if _is_valid(c)]

    model = load_model()
    X_new = featurize_new_compositions(valid_compositions)
    preds = model.predict(X_new)

    out = pd.DataFrame(
        {
            "composition": valid_compositions,
            "predicted_band_gap": preds,
        }
    )
    return out.sort_values("predicted_band_gap", ascending=False).reset_index(drop=True)
