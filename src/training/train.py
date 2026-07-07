import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.evaluation.metrics import evaluate_model
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.model import build_model
from src.utils.config import Config


def set_seed(seed=Config.SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path, target_column=Config.TARGET_COLUMN):
    df = pd.read_csv(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {path}.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.SEED,
        stratify=y,
    )


def ensure_output_dirs():
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("results").mkdir(parents=True, exist_ok=True)


def train(data_path):
    set_seed()
    ensure_output_dirs()

    X_train, X_test, y_train, y_test = load_data(data_path)

    preprocessor = build_preprocessing_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, Config.PREPROCESSOR_PATH)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = dict(zip(np.unique(y_train), class_weights))

    model = build_model(input_dim=X_train_processed.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor="val_auc",
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            patience=5,
            factor=0.5,
            monitor="val_auc",
            mode="max",
        ),
        tf.keras.callbacks.ModelCheckpoint(
            Config.BEST_MODEL_PATH,
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        ),
    ]

    model.fit(
        X_train_processed,
        y_train,
        validation_split=Config.VALIDATION_SIZE,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    metrics = evaluate_model(model, X_test_processed, y_test, threshold=Config.THRESHOLD)

    model.save(Config.MODEL_PATH)
    with open(Config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a production-ready tabular deep learning model.")
    parser.add_argument("--data", type=str, required=True, help="Path to a CSV file containing the target column.")
    args = parser.parse_args()

    train(args.data)
