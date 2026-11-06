
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATASET = "TrafficVolumeData.csv"
MODEL_FILE = "classical_model.pkl"

FEATURES = ["hour", "humidity", "temperature", "clouds_all"]
TARGET = "traffic_volume"

RANDOM_SEED = 42
TRAIN_SAMPLES = 128
TEST_SAMPLES = 32

print("Loading traffic dataset...")

df = pd.read_csv(DATASET)
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df["hour"] = df["date_time"].dt.hour

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

# Reproduce the original quantum train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_SEED,
)

# Reproduce the original quantum training sample selection.
rng = np.random.default_rng(RANDOM_SEED)

train_indices = rng.choice(
    len(X_train),
    size=min(TRAIN_SAMPLES, len(X_train)),
    replace=False,
)

test_indices = np.arange(min(TEST_SAMPLES, len(X_test)))

X_train = X_train[train_indices]
y_train = y_train[train_indices]

X_test = X_test[test_indices]
y_test = y_test[test_indices]

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input features: {FEATURES}")

# Train the classical model.
model = RandomForestRegressor(
    n_estimators=100,
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate on the same held-out records selected by the quantum code.
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Classical Test MAE: {mae:.2f} vehicles")
print(f"Classical Test R²: {r2:.4f}")

print("\nSample predictions:")
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual:.0f}, Predicted: {predicted:.0f}")

# Save model and metadata for inference.
joblib.dump(
    {
        "model": model,
        "features": FEATURES,
        "random_seed": RANDOM_SEED,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    },
    MODEL_FILE,
)

print(f"\nClassical model saved to {MODEL_FILE}")
