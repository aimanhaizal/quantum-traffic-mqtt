
import numpy as np
import pandas as pd
import pennylane as qml

from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET = "TrafficVolumeData.csv"
MODEL_FILE = "quantum_model_params.npz"

N_QUBITS = 4
SHOTS = 200
EPOCHS = 5
BATCH_SIZE = 8
RANDOM_SEED = 42

FEATURES = ["hour", "humidity", "temperature", "clouds_all"]
TARGET = "traffic_volume"


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

print("Loading traffic dataset...")

df = pd.read_csv(DATASET)
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df["hour"] = df["date_time"].dt.hour

df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES].to_numpy(dtype=float)
y = df[TARGET].to_numpy(dtype=float)

# Use a small subset for our first training experiment.
# This is a prototype, not the final model evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)

# Scale features into rotation angles.
x_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

# Standardize the target to make optimization easier.
y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()
y_test_scaled = y_scaler.transform(
    y_test.reshape(-1, 1)
).ravel()

# Keep the first experiment small to limit simulation time.
rng = np.random.default_rng(RANDOM_SEED)
train_indices = rng.choice(
    len(X_train), size=min(128, len(X_train)), replace=False
)
test_indices = np.arange(min(32, len(X_test)))

X_train = X_train[train_indices]
y_train_scaled = y_train_scaled[train_indices]

X_test = X_test[test_indices]
y_test = y_test[test_indices]
y_test_actual = y_test.copy()

print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
print("Input features:", FEATURES)


# --------------------------------------------------
# 3. Define the local quantum circuit
# --------------------------------------------------

# This is Amazon Braket's LOCAL simulator.
# No AWS device or remote task is used.
dev = qml.device(
    "braket.local.qubit",
    wires=N_QUBITS,
    shots=SHOTS,
)


@qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
def quantum_circuit(inputs, circuit_weights):
    # Encode the four classical features as rotations.
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # Trainable quantum rotations.
    for i in range(N_QUBITS):
        qml.RY(circuit_weights[i], wires=i)

    # Entangle the qubits.
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return tuple(
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    )


# --------------------------------------------------
# 4. Build a hybrid quantum-classical regressor
# --------------------------------------------------

# Parameters:
# 0-3: trainable quantum rotation angles
# 4-7: classical output weights
# 8:   classical output bias

def predict_one(inputs, params):
    circuit_weights = params[:N_QUBITS]
    output_weights = params[N_QUBITS:2 * N_QUBITS]
    bias = params[2 * N_QUBITS]

    quantum_outputs = pnp.stack(
        quantum_circuit(inputs, circuit_weights)
    )

    return pnp.dot(quantum_outputs, output_weights) + bias


def loss(params, inputs, targets):
    predictions = pnp.stack([
        predict_one(x, params) for x in inputs
    ])
    return pnp.mean((predictions - targets) ** 2)


# Initialize the 9 trainable parameters.
params = pnp.array(
    rng.uniform(-0.1, 0.1, size=2 * N_QUBITS + 1),
    requires_grad=True,
)

optimizer = qml.AdamOptimizer(stepsize=0.03)


# --------------------------------------------------
# 5. Train
# --------------------------------------------------

print("\nTraining hybrid quantum model...")
print("This may take a little while on the local simulator.")

for epoch in range(EPOCHS):
    order = rng.permutation(len(X_train))
    epoch_losses = []

    for start in range(0, len(X_train), BATCH_SIZE):
        batch_indices = order[start:start + BATCH_SIZE]

        X_batch = X_train[batch_indices]
        y_batch = pnp.array(
            y_train_scaled[batch_indices],
            requires_grad=False,
        )

        params, batch_loss = optimizer.step_and_cost(
            lambda p: loss(p, X_batch, y_batch),
            params,
        )
        epoch_losses.append(float(batch_loss))

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- MSE: {np.mean(epoch_losses):.4f}"
    )


# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

print("\nEvaluating model...")

scaled_predictions = np.array([
    float(predict_one(x, params))
    for x in X_test
])

predictions = y_scaler.inverse_transform(
    scaled_predictions.reshape(-1, 1)
).ravel()

mae = mean_absolute_error(y_test_actual, predictions)
r2 = r2_score(y_test_actual, predictions)

print(f"Test MAE: {mae:.2f} vehicles")
print(f"Test R²: {r2:.4f}")

print("\nSample predictions:")
for actual, predicted in zip(y_test_actual[:10], predictions[:10]):
    print(f"Actual: {actual:.0f}, Predicted: {predicted:.0f}")


# --------------------------------------------------
# 7. Save parameters for later inference
# --------------------------------------------------

np.savez(
    MODEL_FILE,
    params=np.array(params, dtype=float),
    x_min=x_scaler.min_,
    x_scale=x_scaler.scale_,
    y_mean=y_scaler.mean_,
    y_scale=y_scaler.scale_,
    features=np.array(FEATURES),
)

print(f"\nModel parameters saved to {MODEL_FILE}")
print("Training experiment completed.")
