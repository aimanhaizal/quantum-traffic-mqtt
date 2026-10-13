# A Beginner's Guide to Quantum Machine Learning

This guide explains how this project uses quantum computing as part of a machine learning model to predict traffic volume. It is written for readers who are new to quantum computing and want to understand not only what the code does, but also why the circuit is designed this way.

The guide follows the implementation in `quantum_model.py` and the inference workflow used by the MQTT receiver.

## 1. Project overview: what are we trying to do?

The goal of this project is to build a small machine learning pipeline that predicts traffic volume from historical traffic and weather data, then sends those predictions to a dashboard.

The project combines three main ideas:

1. **Machine learning:** Learn relationships between input features and traffic volume from historical examples.
2. **Quantum computing:** Use a parameterized quantum circuit as part of the prediction model.
3. **MQTT and a web dashboard:** Deliver traffic records to the model and display the resulting predictions.

The dataset contains traffic observations alongside information such as humidity, temperature, cloud coverage, and timestamps.

For each incoming record, the model uses four input features:

- Hour of the day
- Humidity
- Temperature
- Cloud coverage

The target is `traffic_volume`, which represents the observed traffic volume in the dataset.

### How the system works

```text
Historical traffic dataset
          |
          v
    quantum_sender.py
          |
          v
      MQTT broker
      traffic/data
          |
          v
   quantum_receiver.py
          |
          v
   quantum_inference.py
          |
          v
 Quantum circuit simulation
          |
          v
  Classical output layer
          |
          v
 Traffic-volume prediction
          |
          +------> Dashboard
          |
          +------> quantum_results.csv
```

The model is trained separately using `quantum_model.py`. Its learned parameters are saved in `quantum_model_params.npz`, which the inference module loads when making predictions.

The sender, MQTT broker, receiver, and dashboard are the delivery and monitoring infrastructure. The quantum machine learning component is the circuit and its trainable parameters.

## 2. Where exactly do we use quantum computing?

Quantum computing is used inside the model's **quantum circuit**.

A conventional machine learning model might use a neural network made entirely of classical numerical operations. In this project, a portion of the computation is represented by a quantum circuit consisting of:

- Four qubits
- Rotation gates that encode the input features
- Trainable rotation gates
- CNOT gates that connect qubits
- Measurements of the Pauli-Z observable

The circuit produces numerical measurements. A classical linear output layer combines these measurements to produce the final traffic-volume prediction.

This makes the model a **hybrid quantum-classical model**: quantum operations and classical calculations work together.

Importantly, the circuit runs on Amazon Braket's local simulator through PennyLane. This project does not require a paid AWS quantum device or remote quantum-computing job.

## 3. Understanding a qubit

A classical computer stores information in bits. A bit has a value of either `0` or `1`.

A qubit, or quantum bit, is the basic unit of quantum information. Its state can be written as a combination of two computational-basis states:

$$
|\psi\rangle=\alpha|0\rangle+\beta|1\rangle
$$

Here, $\alpha$ and $\beta$ are complex probability amplitudes, and their squared magnitudes satisfy:

$$
|\alpha|^2+|\beta|^2=1
$$

When the qubit is measured in the computational basis, the probability of observing `0` is $|\alpha|^2$, and the probability of observing `1` is $|\beta|^2$.

A qubit can therefore be in a superposition of the two basis states before measurement. This is not simply a classical bit that is secretly either zero or one; the amplitudes also determine how quantum operations interfere.

### What does a qubit represent in our project?

A useful first approximation is to think of a qubit as a component of the model's quantum state.

Our model has four input features and four qubits. Each feature is encoded into the corresponding qubit using a rotation gate.

However, a qubit is not permanently assigned a classical value such as "humidity = 70%". Instead, a numerical feature is transformed into a gate parameter, and the gate changes the qubit's quantum state.

The qubit does not store the original feature in a way that can simply be read back. The circuit transforms the state, and measurements extract information from the resulting state.

## 4. Step one: encoding the input data

Before the quantum circuit can process our traffic data, the classical numerical inputs must influence the quantum state.

This process is called **data encoding** or **feature embedding**.

Our implementation uses rotation gates around the Y axis, written as `RY(θ)`.

A rotation gate changes a qubit's state according to an angle, $\theta$. For a qubit initially in state $|0\rangle$, an RY rotation produces:

$$
RY(\theta)|0\rangle =
\cos\left(\frac{\theta}{2}\right)|0\rangle+
\sin\left(\frac{\theta}{2}\right)|1\rangle
$$

The rotation angle changes the amplitudes of the resulting state.

For example, an angle of zero leaves the initial state unchanged. Other angles produce different superpositions.

### How does this encode traffic data?

Suppose the input row contains:

- Hour: 8
- Humidity: 65
- Temperature: 290
- Cloud coverage: 40

These are ordinary classical values. Before using them as rotation angles, the model applies its input preprocessing and scaling.

The resulting feature values are used to determine the angles of the encoding rotations.

Conceptually:

```text
Classical feature
       |
       v
Preprocessing and scaling
       |
       v
Rotation angle
       |
       v
RY gate changes the qubit state
```

The purpose is to make the quantum state depend on the input example. Different input values can produce different quantum states, allowing the model's output to depend on the data.

Why use rotation gates? They provide a simple, continuous way to map numerical values to quantum states. RY is a standard choice for encoding real-valued features.

It is not the only possible encoding method, and using RY does not automatically make the representation superior to classical feature processing.

## 5. Step two: trainable quantum operations

Encoding alone is not the entire learning process.

The circuit also contains **trainable parameters**: adjustable rotation angles that are optimized during training.

These parameters determine how the quantum state is transformed after the input has been encoded.

A useful analogy is a conventional neural network. Its weights are adjusted during training so that its predictions become closer to the correct answers. In a variational quantum circuit, trainable gate parameters play a related role.

The training loop adjusts these parameters to reduce the model's prediction error.

### Why not use only the encoding gates?

If the circuit only encoded the input and immediately measured it, its output would be largely determined by the encoding procedure. There would be little or no learned transformation of the encoded data.

Trainable gates give the model adjustable degrees of freedom. Training searches for parameter values that make the circuit's measured outputs useful for predicting traffic volume.

This type of circuit is commonly called a **variational quantum circuit** or a parameterized quantum circuit.

## 6. Step three: connecting qubits with CNOT gates

So far, the circuit has encoded the features and applied trainable rotations. We also want the qubits to interact.

The circuit uses CNOT gates, which stands for *controlled-NOT*.

A CNOT operates on two qubits:

- The first qubit is the control.
- The second qubit is the target.
- If the control is measured in the computational basis as `1`, the target bit is flipped; if the control is `0`, the target is unchanged.

Its action on computational-basis states is:

| Input | Output |
|---|---|
| $|00\rangle$ | $|00\rangle$ |
| $|01\rangle$ | $|01\rangle$ |
| $|10\rangle$ | $|11\rangle$ |
| $|11\rangle$ | $|10\rangle$ |

These examples describe the gate's action on basis states. For general superpositions, quantum mechanics applies the operation to the amplitudes as well.

### Why do we need CNOT gates?

CNOT gates allow the circuit to perform joint operations on multiple qubits rather than treating each qubit independently.

This is important because the model may benefit from representing relationships between features.

For example, traffic conditions may depend on a combination of humidity and temperature rather than either feature independently. In machine learning, such relationships are often called feature interactions.

The CNOT gate allows quantum correlations to be introduced into the circuit. When combined with appropriate rotations, it can create entanglement.

Our circuit uses CNOT gates in a simple entangling pattern. This is a design choice intended to let information represented across different qubits interact.

## 7. What is entanglement, and why might it matter?

Entanglement is a quantum property in which the joint state of multiple qubits cannot be described as a product of independent states for the individual qubits.

Consider two qubits initially in state $|00\rangle$.

First, apply a Hadamard gate to the first qubit. This creates a superposition:

$$
\frac{|00\rangle+|10\rangle}{\sqrt{2}}
$$

Next, apply a CNOT with the first qubit as the control and the second as the target:

$$
\frac{|00\rangle+|11\rangle}{\sqrt{2}}
$$

This final state is entangled. The two qubits cannot be represented as two independent single-qubit states.

This example uses a Hadamard gate to explain the principle; our traffic model instead uses its own sequence of rotation and CNOT gates.

### Why introduce entanglement into a machine learning circuit?

Entangling gates allow the circuit to represent joint quantum states that cannot be reproduced by simply assigning an independent state to each qubit.

After further circuit operations, these joint states can affect the measurement statistics. This gives the model a way to express relationships involving multiple encoded features.

For example, a model might need to represent a relationship involving both humidity and temperature, rather than treating them as completely separate signals.

There is an important limitation: entanglement does not automatically mean better predictions. The usefulness of entanglement depends on the circuit design, training procedure, data, and task. A small quantum circuit can still underperform a simpler model.

In this project, entanglement is part of the model's design, not proof of a quantum advantage.

## 8. Step four: measuring the circuit with Pauli-Z

After the circuit has applied its gates, we need to extract numerical information from the quantum state.

This is the purpose of measurement.

Our model measures the Pauli-Z observable on each qubit. The Pauli-Z operator is represented by:

$$
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}
$$

For computational-basis measurement:

- Outcome $|0\rangle$ corresponds to $+1$.
- Outcome $|1\rangle$ corresponds to $-1$.

A single measurement produces only one outcome. Repeated measurements allow us to estimate the average value, called the **expectation value**.

For Pauli-Z:

$$
\langle Z\rangle=P(0)-P(1)
$$

Because the two probabilities sum to one, the expectation value lies between $-1$ and $+1$.

### A simple example

Suppose repeated measurements produce `0` about 75% of the time and `1` about 25% of the time.

The expected Pauli-Z value is:

$$
\langle Z\rangle=0.75-0.25=0.50
$$

If the probabilities were reversed, the expectation value would be $-0.50$.

The expectation value therefore gives us a numerical summary of the measurement distribution.

### Why use Pauli-Z?

Pauli-Z is a convenient and standard observable for extracting information from a qubit. It produces a scalar value that can be used as a feature by the classical part of the model.

With four qubits, measuring each qubit's Pauli-Z expectation value provides four numerical outputs.

Those outputs are not traffic volumes by themselves. They are intermediate values generated by the quantum circuit.

Other observables and measurement strategies are possible. Pauli-Z is a straightforward choice for this prototype.

### Why do we use multiple shots?

The simulation is configured to use a finite number of measurement shots. A shot is one circuit execution followed by measurement.

Repeated shots estimate the probabilities and expectation values. With a finite number of shots, the estimates can fluctuate because of sampling noise.

More shots generally improve the statistical precision of the estimates, but they also increase computational work. The trade-off is relevant when deciding how to configure a quantum machine learning experiment.

## 9. Step five: the classical output layer

The quantum circuit produces four Pauli-Z expectation values. The model then passes these measurements to a classical linear output layer.

Conceptually, the final prediction has the form:

$$
\hat{y}=w_1z_1+w_2z_2+w_3z_3+w_4z_4+b
$$

Here:

- $z_1,\ldots,z_4$ are the measured expectation values.
- $w_1,\ldots,w_4$ are classical weights.
- $b$ is a bias term.
- $\hat{y}$ is the predicted traffic volume.

The classical output layer turns the circuit's measurements into a value expressed in the scale of the target variable.

Why not make the entire model quantum? A hybrid design is simpler to implement and train. It also demonstrates how a quantum circuit can be incorporated into a conventional machine learning pipeline without requiring every operation to be quantum.

## 10. How does the model learn?

The model is trained using examples that contain both input features and known traffic volumes.

A simplified training loop looks like this:

```text
For each training example:
    1. Read the input features.
    2. Encode the features into the quantum circuit.
    3. Apply trainable rotations and entangling gates.
    4. Measure the qubits.
    5. Calculate the traffic-volume prediction.
    6. Compare the prediction with the actual value.
    7. Calculate the loss.
    8. Update trainable parameters.
```

The loss function measures prediction error. This project uses mean squared error (MSE) during training.

For a single prediction, the squared error is:

$$
L=(y-\hat{y})^2
$$

where $y$ is the actual traffic volume and $\hat{y}$ is the prediction.

Training aims to reduce the average loss over the training examples.

### How are quantum parameters updated?

The model needs to estimate how changing a trainable rotation angle affects the prediction and loss.

For suitable parameterized quantum gates, the **parameter-shift rule** can estimate derivatives by evaluating the circuit at shifted parameter values. These derivatives can then be used by a classical optimizer to update the parameters.

The optimization is therefore hybrid:

- The quantum circuit generates measurement values.
- Classical calculations determine the loss and parameter updates.
- The updated parameters are used in later circuit executions.

Training repeats this process over multiple examples and epochs.

## 11. How does this relate to the actual code?

The main implementation is in `quantum_model.py`.

| Component | Role in the project |
|---|---|
| Feature preparation | Selects the hour, humidity, temperature, and cloud coverage inputs and prepares them for the circuit. |
| Quantum device | Configures the local Braket simulator and the number of measurement shots. |
| Input encoding | Uses RY rotations to make the quantum state depend on the input features. |
| Trainable gates | Provide adjustable parameters that training can optimize. |
| CNOT gates | Connect qubits and allow entangling operations. |
| Pauli-Z measurements | Convert the final quantum state into numerical expectation values. |
| Classical output layer | Combines measurements into a traffic-volume prediction. |
| Loss and optimizer | Evaluate prediction error and adjust the model parameters. |
| Saved parameters | Store the trained parameters for later inference. |

The inference implementation in `quantum_inference.py` reconstructs the circuit, loads the saved parameters, and uses the trained model to predict traffic volume for incoming MQTT messages.

The receiver then records the prediction and timing information for the dashboard.

## 12. What did the prototype achieve?

In the development run, the model was trained for five epochs on a small subset of the dataset.

| Metric | Recorded result |
|---|---:|
| Training examples | Up to 128 |
| Test examples | 32 |
| Training epochs | 5 |
| Test MAE | 1,031.77 vehicles |
| Test R² | 0.5173 |

These are results from a small experiment, not a comprehensive evaluation. They can vary with the training setup, simulator behavior, and parameter initialization.

The prototype demonstrates that a hybrid quantum-classical model can be integrated into a working MQTT pipeline and used to generate predictions for a dashboard.

It does not establish that the model is more accurate or faster than classical machine learning. It also does not demonstrate a quantum advantage.

## 13. Important limitations

Several limitations should be kept in mind:

- **Local simulation:** The circuit runs on a classical computer using the Braket local simulator. It is not executing on physical quantum hardware.
- **Small training run:** The training subset and number of epochs were deliberately limited for this prototype.
- **Finite-shot measurements:** Measurement estimates can contain sampling noise.
- **Model design:** The number of qubits, gate sequence, and encoding method are design choices, not universal requirements for quantum machine learning.
- **Classical output:** The final prediction is produced by a classical linear layer using quantum measurement results.
- **No demonstrated quantum advantage:** Using quantum gates does not guarantee improved accuracy, efficiency, or scalability.

## 14. The big picture

The main idea is not that every machine learning problem needs a quantum computer.

Instead, this project demonstrates a particular approach to quantum machine learning:

1. Convert classical features into parameters of quantum operations.
2. Transform the encoded state using trainable gates.
3. Use entangling operations to allow qubits to interact.
4. Measure the circuit to obtain numerical expectation values.
5. Use classical calculations to turn those measurements into predictions.
6. Train the model by adjusting its parameters to reduce prediction error.

The quantum circuit acts as a trainable component inside a larger classical system. MQTT transports the data, the model generates predictions, and the dashboard makes the experiment observable.

That combination provides a practical starting point for understanding how quantum circuits can be integrated into machine learning applications.
