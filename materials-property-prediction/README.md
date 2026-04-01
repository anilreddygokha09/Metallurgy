# Materials Property Prediction and New Material Discovery using Machine Learning

A production-ready Python project for predicting **band gap** from material composition and discovering promising new candidate materials.

## Project Overview

This project builds an end-to-end materials informatics pipeline:
1. Fetches composition and band gap data from the **Materials Project API**.
2. Converts compositions into physically meaningful descriptors using **matminer** (Magpie preset).
3. Trains an **XGBoost regressor** to predict band gap.
4. Evaluates model quality using **R²** and **RMSE**.
5. Generates and ranks new candidate compositions (e.g., Li-Fe-O and Ni-Mn-Co systems).

## Dataset

Data is sourced from the [Materials Project](https://materialsproject.org/) using the `mp-api` client.

Fetched columns:
- `formula`
- `composition`
- `band_gap` (target, in eV)

Raw data is saved in:
- `data/raw/materials_project_bandgap.csv`

## Method

### Feature Engineering (matminer)
- Composition strings are parsed using `pymatgen.Composition`.
- `matminer.featurizers.composition.ElementProperty.from_preset("magpie")` converts each composition into elemental-statistics descriptors.
- The resulting numeric matrix is saved to:
  - `data/processed/features.csv`

### Model (XGBoost)
- Model: `xgboost.XGBRegressor`
- Train/test split: 80/20
- Evaluation metrics:
  - **R²**
  - **RMSE**
- Trained model saved to:
  - `models/model.pkl`

## Discovery Output

The project includes candidate generation in relevant chemistry spaces (Li-Fe-O and Ni-Mn-Co).

Outputs:
- `results/discovered_materials.csv`: Ranked candidate compositions with predicted band gaps.
- `results/metrics.json`: R² and RMSE on held-out test data.
- `results/plots/predicted_vs_actual.png`: parity plot (predicted vs actual).

## Project Structure

```text
materials-property-prediction/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_material_discovery.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── featurization.py
│   ├── model.py
│   ├── predict.py
│   ├── utils.py
│
├── models/
│   └── model.pkl
│
├── results/
│   ├── plots/
│   ├── metrics.json
│   ├── discovered_materials.csv
│
├── requirements.txt
├── README.md
└── main.py
```

## How to Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Run the full pipeline

```bash
python main.py --api-key "YOUR_MATERIALS_PROJECT_API_KEY" --num-materials 1000
```

### 3) Check outputs

- Metrics: `results/metrics.json`
- Plot: `results/plots/predicted_vs_actual.png`
- Discoveries: `results/discovered_materials.csv`

## Notes

- Replace the API key placeholder with your valid Materials Project API key.
- The exact model performance (R², RMSE) depends on fetched data size and composition distribution.
- Candidate generation is configurable via `generate_candidate_compositions()` in `main.py`.
