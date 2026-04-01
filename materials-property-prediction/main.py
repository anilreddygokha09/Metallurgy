"""End-to-end pipeline for materials property prediction and discovery."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data_loader import fetch_materials_data
from src.featurization import featurize_compositions
from src.model import train_model
from src.predict import predict_band_gap_for_compositions
from src.utils import PLOTS_DIR, RESULTS_DIR, ensure_directories, plot_predictions, save_json


def generate_candidate_compositions() -> list[str]:
    """Generate candidate compositions in Li-Fe-O and Ni-Mn-Co chemical spaces."""
    return [
        "LiFeO2",
        "Li2FeO3",
        "LiFe2O4",
        "Li3FeO4",
        "LiFePO4",
        "LiNi0.3Mn0.3Co0.4O2",
        "LiNi0.5Mn0.3Co0.2O2",
        "LiNi0.6Mn0.2Co0.2O2",
        "LiNi0.8Mn0.1Co0.1O2",
        "LiNi0.7Mn0.15Co0.15O2",
        "NiMnCoO4",
        "Ni0.5Mn1.0Co1.5O4",
        "Ni0.6Mn1.2Co1.2O4",
        "Ni1.0Mn1.0Co1.0O4",
        "Li2NiMn3O8",
        "Li2Ni0.5Mn1.5O4",
        "LiCoO2",
        "LiNiO2",
        "LiMn2O4",
        "Li2MnO3",
    ]


def run_pipeline(api_key: str, num_materials: int = 1000) -> None:
    """Run full workflow: extract, featurize, train, evaluate, discover."""
    ensure_directories()

    raw_df = fetch_materials_data(api_key=api_key, num_materials=num_materials)
    X, y = featurize_compositions(raw_df)

    _, metrics, y_test, y_pred = train_model(X, y)

    save_json(metrics, RESULTS_DIR / "metrics.json")
    plot_predictions(y_true=y_test, y_pred=y_pred, output_path=PLOTS_DIR / "predicted_vs_actual.png")

    candidates = generate_candidate_compositions()
    discovered = predict_band_gap_for_compositions(candidates)
    discovered.to_csv(RESULTS_DIR / "discovered_materials.csv", index=False)

    print("Pipeline completed successfully.")
    print(f"Metrics saved to: {RESULTS_DIR / 'metrics.json'}")
    print(f"Discovered materials saved to: {RESULTS_DIR / 'discovered_materials.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materials property prediction and discovery pipeline")
    parser.add_argument(
        "--api-key",
        type=str,
        default="YOUR_MATERIALS_PROJECT_API_KEY",
        help="Materials Project API key",
    )
    parser.add_argument("--num-materials", type=int, default=1000, help="Number of materials to fetch")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.api_key == "YOUR_MATERIALS_PROJECT_API_KEY":
        raise ValueError(
            "Please provide a valid Materials Project API key via --api-key or replace placeholder in main.py"
        )
    run_pipeline(api_key=args.api_key, num_materials=args.num_materials)
