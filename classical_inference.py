
import time

import joblib
import numpy as np

MODEL_FILE = "classical_model.pkl"

# Load the trained Random Forest once at startup.
saved = joblib.load(MODEL_FILE)

model = saved["model"]
FEATURES = saved["features"]


def predict_traffic(hour, humidity, temperature, clouds_all):
    """Return (predicted traffic volume, inference time in milliseconds)."""

    raw_features = np.array(
        [[hour, humidity, temperature, clouds_all]],
        dtype=float,
    )

    start = time.perf_counter()
    prediction = float(model.predict(raw_features)[0])
    inference_ms = (time.perf_counter() - start) * 1000

    return max(0.0, prediction), inference_ms
