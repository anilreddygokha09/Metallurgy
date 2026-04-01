"""Data loading module for Materials Project band gap data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from mp_api.client import MPRester

from .utils import DATA_RAW_DIR


DEFAULT_RAW_FILENAME = "materials_project_bandgap.csv"


def fetch_materials_data(
    api_key: str,
    output_path: Optional[Path] = None,
    num_materials: int = 1000,
) -> pd.DataFrame:
    """Fetch formula/composition/band gap data from Materials Project.

    Parameters
    ----------
    api_key
        Materials Project API key.
    output_path
        Optional output CSV path. Defaults to data/raw/materials_project_bandgap.csv.
    num_materials
        Maximum number of materials to fetch.

    Returns
    -------
    pd.DataFrame
        DataFrame with formula_pretty, composition, and band_gap columns.
    """
    if output_path is None:
        output_path = DATA_RAW_DIR / DEFAULT_RAW_FILENAME

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Query only entries with a known band gap to support supervised learning.
    with MPRester(api_key=api_key) as mpr:
        docs = mpr.materials.summary.search(
            fields=["formula_pretty", "composition_reduced", "band_gap"],
            band_gap=(0, 20),
            num_chunks=1,
            chunk_size=num_materials,
        )

    records = []
    for doc in docs:
        records.append(
            {
                "formula": doc.formula_pretty,
                "composition": str(doc.composition_reduced)
                if doc.composition_reduced is not None
                else doc.formula_pretty,
                "band_gap": float(doc.band_gap),
            }
        )

    df = pd.DataFrame(records).dropna(subset=["formula", "composition", "band_gap"])
    df = df.drop_duplicates(subset=["composition"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df


def load_raw_data(path: Optional[Path] = None) -> pd.DataFrame:
    """Load raw CSV data."""
    if path is None:
        path = DATA_RAW_DIR / DEFAULT_RAW_FILENAME
    return pd.read_csv(path)
