class Config:
    SEED = 42
    TARGET_COLUMN = "TARGET"
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.2
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    THRESHOLD = 0.5
    MODEL_PATH = "models/final_model.keras"
    BEST_MODEL_PATH = "models/best_model.keras"
    PREPROCESSOR_PATH = "models/preprocessor.joblib"
    METRICS_PATH = "results/metrics.json"
