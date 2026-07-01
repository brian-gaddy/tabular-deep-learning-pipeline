import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.models.model import build_model
from src.evaluation.metrics import evaluate_model
from src.utils.config import Config


def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train(data_path):
    set_seed()

    X_train, X_test, y_train, y_test = load_data(data_path)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    model = build_model(input_dim=X_train.shape[1])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        tf.keras.callbacks.ModelCheckpoint("models/best_model.h5", save_best_only=True)
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=256,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    evaluate_model(model, X_test, y_test)

    model.save("models/final_model.h5")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    args = parser.parse_args()

    train(args.data)
