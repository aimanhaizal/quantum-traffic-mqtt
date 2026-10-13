
import pennylane as qml
from pennylane import numpy as np

# Amazon Braket LOCAL simulator only.
# No AWS device or remote task is used.
dev = qml.device(
    "braket.local.qubit",
    wires=2,
    shots=1000,
)

@qml.qnode(dev, diff_method="parameter-shift")
def circuit(theta):
    qml.RY(theta, wires=0)
    qml.CNOT(wires=[0, 1])

    return qml.expval(qml.PauliZ(0))


if __name__ == "__main__":
    theta = np.array(0.7, requires_grad=True)

    value = circuit(theta)
    gradient = qml.grad(circuit)(theta)

    print("Local Braket gradient test")
    print("Input angle:", float(theta))
    print("Circuit output:", float(value))
    print("Calculated gradient:", float(gradient))
    print("Gradient calculation completed.")
