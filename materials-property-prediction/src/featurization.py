"""Feature engineering for compositions using matminer."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

from .utils import DATA_PROCESSED_DIR


DEFAULT_FEATURES_FILENAME = "features.csv"


def _validate_composition(formula: str) -> bool:
    """Check if formula can be parsed by pymatgen Composition."""
    try:
        Composition(formula)
        return True
    except Exception:
        return False


def featurize_compositions(
    df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Convert composition strings into Magpie elemental descriptors.

    Materials-science note:
    ElementProperty with the Magpie preset captures composition-derived
    statistics (e.g., mean electronegativity, atomic radius mismatch), which
    are physically meaningful proxies for electronic structure trends such as
    band gap behavior.
    """
    if output_path is None:
        output_path = DATA_PROCESSED_DIR / DEFAULT_FEATURES_FILENAME

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data = data[data["composition"].astype(str).map(_validate_composition)].reset_index(drop=True)

    data["composition_obj"] = data["composition"].map(Composition)

    featurizer = ElementProperty.from_preset("magpie")
    feature_df = featurizer.featurize_dataframe(
        data,
        col_id="composition_obj",
        ignore_errors=True,
    )

    feature_cols = featurizer.feature_labels()
    X = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = pd.to_numeric(feature_df["band_gap"], errors="coerce").fillna(0.0)

    processed_df = pd.concat([data[["formula", "composition", "band_gap"]], X], axis=1)
    processed_df.to_csv(output_path, index=False)

    return X, y


def featurize_new_compositions(compositions: list[str]) -> pd.DataFrame:
    """Featurize unseen composition strings for inference."""
    valid = [c for c in compositions if _validate_composition(c)]
    infer_df = pd.DataFrame({"composition": valid})
    infer_df["composition_obj"] = infer_df["composition"].map(Composition)

    featurizer = ElementProperty.from_preset("magpie")
    infer_df = featurizer.featurize_dataframe(infer_df, col_id="composition_obj", ignore_errors=True)
    X_new = infer_df[featurizer.feature_labels()].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X_new
