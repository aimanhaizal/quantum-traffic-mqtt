
import pennylane as qml

# Use Amazon Braket's LOCAL simulator.
# This does not submit a task to AWS.
dev = qml.device(
    "braket.local.qubit",
    wires=2,
    shots=1000,
)

@qml.qnode(dev)
def circuit(theta):
    # Encode a value using a rotation gate.
    qml.RY(theta, wires=0)

    # Entangle the two qubits.
    qml.CNOT(wires=[0, 1])

    # Measure both qubits.
    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
    )


if __name__ == "__main__":
    result = circuit(0.7)

    print("Local Amazon Braket simulator test")
    print("Input angle:", 0.7)
    print("Qubit 0 expectation:", result[0])
    print("Qubit 1 expectation:", result[1])
    print("Circuit executed successfully.")
