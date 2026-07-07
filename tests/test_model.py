from src.models.model import build_model


def test_build_model_compiles_for_binary_classification():
    model = build_model(input_dim=10)

    assert model.input_shape == (None, 10)
    assert model.output_shape == (None, 1)
    assert model.loss == "binary_crossentropy"
