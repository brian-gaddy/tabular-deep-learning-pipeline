# End-to-End Tabular Deep Learning Pipeline (Production-Ready)

## Overview

This repository contains a production-oriented machine learning pipeline for structured/tabular data using TensorFlow/Keras. The project demonstrates data cleaning, preprocessing, feature handling, model training, evaluation, inference, artifact persistence, and test coverage.

The original notebook-style workflow has been refactored into a modular codebase that is easier to maintain, test, reuse, and present in a professional data science portfolio.

> Recommended repository name: `production-ready-tabular-deep-learning-pipeline` or `tabular-deep-learning-pipeline`.

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── features/
│   │   └── preprocessing.py
│   ├── models/
│   │   └── model.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── inference/
│   │   └── predict.py
│   └── utils/
│       └── config.py
├── tests/
├── models/
├── figures/
├── results/
├── PROJECT_REPORT.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Key Improvements

- Removes `SK_ID_CURR` identifier leakage before modeling.
- Drops constant features learned from the training data.
- Handles the `DAYS_EMPLOYED` sentinel value.
- Supports mixed numeric and categorical tabular data.
- Uses median imputation and scaling for numeric features.
- Uses most-frequent imputation and one-hot encoding for categorical features.
- Uses class weights instead of undersampling.
- Saves the trained Keras model and fitted preprocessing pipeline.
- Adds a batch inference script for scoring new data.
- Adds smoke tests for preprocessing, model creation, and training data splitting.

---

## Evaluation Metrics

- ROC-AUC
- PR-AUC
- Recall
- F1-score
- Confusion Matrix
- Classification Report

---

## Model Training

Training uses:

- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- Fixed random seeds
- Centralized configuration
- Saved model and preprocessor artifacts

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python src/training/train.py --data data/processed/train.csv
```

Generate predictions:

```bash
python src/inference/predict.py --input data/processed/scoring.csv --output results/predictions.csv
```

Run tests:

```bash
pytest
```

---

## Expected Artifacts

Training produces:

```text
models/final_model.keras
models/best_model.keras
models/preprocessor.joblib
results/metrics.json
```

These artifacts are ignored by Git so the repository stays lightweight.

---

## Portfolio Relevance

This project demonstrates production-style data science practices: modular Python code, reproducible training, saved preprocessing, model evaluation, batch inference, configuration management, and basic automated tests. It is suitable for demonstrating applied machine learning engineering skills for Data Analyst, Data Scientist, ML Engineer, and Analytics Engineer roles.
