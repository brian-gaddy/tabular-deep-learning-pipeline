import pandas as pd

from src.training.train import load_data


def test_load_data_splits_features_and_target(tmp_path):
    df = pd.DataFrame(
        {
            "feature_a": range(20),
            "feature_b": ["A", "B"] * 10,
            "TARGET": [0, 1] * 10,
        }
    )
    data_path = tmp_path / "sample.csv"
    df.to_csv(data_path, index=False)

    X_train, X_test, y_train, y_test = load_data(data_path)

    assert "TARGET" not in X_train.columns
    assert len(X_train) + len(X_test) == 20
    assert len(y_train) + len(y_test) == 20
