import argparse
from pathlib import Path

import joblib
import pandas as pd
import tensorflow as tf

from src.utils.config import Config


def load_artifacts(model_path=Config.MODEL_PATH, preprocessor_path=Config.PREPROCESSOR_PATH):
    model_file = Path(model_path)
    preprocessor_file = Path(preprocessor_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_file}")
    if not preprocessor_file.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_file}")

    model = tf.keras.models.load_model(model_file)
    preprocessor = joblib.load(preprocessor_file)
    return model, preprocessor


def predict(input_path, output_path, threshold=Config.THRESHOLD):
    model, preprocessor = load_artifacts()
    df = pd.read_csv(input_path)

    if Config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[Config.TARGET_COLUMN])

    X_processed = preprocessor.transform(df)
    probabilities = model.predict(X_processed).ravel()
    predictions = (probabilities >= threshold).astype(int)

    output = df.copy()
    output["prediction_probability"] = probabilities
    output["prediction"] = predictions

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions from a trained tabular deep learning model.")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV for scoring.")
    parser.add_argument("--output", type=str, default="results/predictions.csv", help="Path to save predictions.")
    parser.add_argument("--threshold", type=float, default=Config.THRESHOLD, help="Classification threshold.")
    args = parser.parse_args()

    predict(args.input, args.output, args.threshold)
