
import time

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


MODEL_FILE = "quantum_model_params.npz"
N_QUBITS = 4
SHOTS = 200

FEATURES = ["hour", "humidity", "temperature", "clouds_all"]

# Load trained parameters and scaling information once.
saved = np.load(MODEL_FILE)

params = pnp.array(saved["params"], requires_grad=False)
x_min = saved["x_min"]
x_scale = saved["x_scale"]
y_mean = float(saved["y_mean"][0])
y_scale = float(saved["y_scale"][0])

# Local Amazon Braket simulator only.
dev = qml.device(
    "braket.local.qubit",
    wires=N_QUBITS,
    shots=SHOTS,
)


@qml.qnode(dev, interface="autograd")
def quantum_circuit(inputs, circuit_weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    for i in range(N_QUBITS):
        qml.RY(circuit_weights[i], wires=i)

    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return tuple(
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    )


def predict_traffic(hour, humidity, temperature, clouds_all):
    """Return (predicted traffic volume, inference time in milliseconds)."""

    raw_features = np.array(
        [[hour, humidity, temperature, clouds_all]],
        dtype=float,
    )

    # Apply the same MinMaxScaler transformation used during training.
    scaled_features = raw_features * x_scale + x_min
    inputs = pnp.array(scaled_features[0], requires_grad=False)

    circuit_weights = params[:N_QUBITS]
    output_weights = params[N_QUBITS:2 * N_QUBITS]
    bias = params[2 * N_QUBITS]

    start = time.perf_counter()

    quantum_outputs = pnp.stack(
        quantum_circuit(inputs, circuit_weights)
    )
    scaled_prediction = float(
        pnp.dot(quantum_outputs, output_weights) + bias
    )

    inference_ms = (time.perf_counter() - start) * 1000

    # Reverse the target standardization from training.
    prediction = scaled_prediction * y_scale + y_mean

    return max(0.0, prediction), inference_ms
