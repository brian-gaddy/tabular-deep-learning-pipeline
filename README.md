# End-to-End Tabular Deep Learning Pipeline (Production-Ready)

## Overview
This repository contains a production-quality machine learning pipeline for structured/tabular data using TensorFlow/Keras. The project demonstrates best practices in data preprocessing, feature engineering, model training, evaluation, and deployment readiness.

It has been refactored from a notebook-based workflow into a modular, maintainable, and reproducible codebase suitable for portfolio and production use.

---

## Project Structure

```
.
├── data/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
├── models/
├── figures/
├── results/
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Key Improvements

- Removed SK_ID_CURR identifier
- Dropped constant features
- Fixed DAYS_EMPLOYED sentinel handling
- Removed unsafe transformations
- Added class weights instead of undersampling
- Added threshold tuning

## Evaluation Metrics
- ROC-AUC
- PR-AUC
- Recall
- F1-score
- Confusion Matrix

## Model Training
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint

## Reproducibility
- Fixed seeds
- Centralized config
- Saved preprocessing pipeline

## How to Run
```bash
pip install -r requirements.txt
python src/training/train.py
```