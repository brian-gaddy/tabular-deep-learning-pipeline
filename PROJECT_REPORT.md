# Project Report: End-to-End Tabular Deep Learning Pipeline

## Executive Summary

This project converts a notebook-based tabular deep learning workflow into a modular, production-oriented Python codebase. The pipeline supports structured data preprocessing, neural network training, binary classification evaluation, artifact persistence, and batch inference.

The project is designed to demonstrate practical machine learning engineering habits: separating data preparation from modeling, saving the fitted preprocessing pipeline, preserving reproducibility through configuration, and enabling repeatable scoring through an inference script.

## Problem Context

Many tabular machine learning projects remain trapped in notebooks, making them difficult to reproduce, test, deploy, or reuse. This project addresses that gap by creating a reusable pipeline for binary classification on structured data.

The assumed target column is `TARGET`, and the model is designed for imbalanced classification problems where ROC-AUC, PR-AUC, recall, F1-score, and confusion matrix performance are important.

## Pipeline Architecture

The project is organized into clear modules:

- `src/features/preprocessing.py` handles feature cleaning and preprocessing.
- `src/models/model.py` defines the TensorFlow/Keras neural network.
- `src/training/train.py` orchestrates training, evaluation, and artifact persistence.
- `src/evaluation/metrics.py` calculates classification metrics.
- `src/inference/predict.py` generates batch predictions from saved artifacts.
- `src/utils/config.py` centralizes project settings.
- `tests/` provides smoke tests for preprocessing, modeling, and training data splitting.

## Preprocessing Design

The preprocessing layer removes the `SK_ID_CURR` identifier to reduce leakage risk, drops constant columns, replaces the Home Credit `DAYS_EMPLOYED` sentinel value with missing values, imputes numeric and categorical features, scales numeric values, and one-hot encodes categorical features.

The fitted preprocessing pipeline is saved with `joblib` so future inference uses the same transformation logic used during training.

## Model Design

The model is a feedforward neural network built with TensorFlow/Keras. It uses dense layers, ReLU activations, batch normalization, dropout regularization, and a sigmoid output layer for binary classification.

Training uses class weights to address target imbalance without undersampling the data. It also uses EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint callbacks.

## Evaluation Strategy

The evaluation module reports:

- ROC-AUC
- PR-AUC
- Recall
- F1-score
- Confusion Matrix
- Classification Report

These metrics are appropriate for imbalanced binary classification because accuracy alone can be misleading when the positive class is rare.

## Production-Readiness Assessment

This project now includes several production-style components:

- Modular source-code layout
- Centralized configuration
- Saved model artifact
- Saved preprocessing artifact
- Batch inference script
- Test coverage for core components
- Git-friendly artifact exclusions
- Clear run commands in the README

## Remaining Improvements

The next upgrades would make the project stronger:

1. Add GitHub Actions for automated tests.
2. Add logging instead of relying on print statements.
3. Add command-line arguments for target column, threshold, and output paths.
4. Add model card documentation.
5. Add example synthetic data for quick local testing.
6. Add Docker support for environment reproducibility.
7. Add MLflow or experiment tracking.

## Portfolio Positioning

This project is best positioned as a machine learning engineering and production-readiness portfolio piece. It shows that the author can move beyond notebook-only analysis and structure a machine learning project in a way that is easier to maintain, test, and operationalize.
