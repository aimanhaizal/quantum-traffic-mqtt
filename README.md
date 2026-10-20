# Quantum Traffic Prediction with MQTT

A prototype demonstrating how a small **hybrid quantum-classical machine learning model** can be integrated into an MQTT data pipeline and monitored through a live web dashboard.

The project replays historical traffic data, publishes each record through MQTT, runs a quantum circuit on Amazon Braket's **local simulator**, and displays predicted versus actual traffic volume alongside inference timings.

> **No paid AWS quantum device is used.** Model training and inference run locally using the Braket local simulator.

## Dashboard

![Quantum Traffic MQTT Dashboard](screenshots/quantum-dashboard.png)

*The dashboard displays recent traffic predictions, actual traffic volume, and quantum inference runtime.*

## Features

- **Quantum machine learning:** A small hybrid quantum-classical regressor implemented with PennyLane.
- **Amazon Braket local simulator:** Executes the quantum circuit locally without submitting jobs to a paid quantum device.
- **MQTT data pipeline:** Publishes historical traffic records to the `traffic/data` topic.
- **Live Flask dashboard:** Displays recent actual and predicted traffic values.
- **Inference monitoring:** Records quantum inference time and total receiver processing time.
- **Experiment logging:** Saves processed prediction records to `quantum_results.csv`.
- **Dataset replay:** Replays valid rows from the traffic dataset at a configurable interval.

## Architecture

```text
TrafficVolumeData.csv
        |
        v
quantum_sender.py
        |
        v
   MQTT Broker
   traffic/data
        |
        v
quantum_receiver.py
        |
        +----------------------+
        |                      |
        v                      v
quantum_inference.py     quantum_results.csv
        |
        v
PennyLane + Braket
Local quantum simulator
        |
        v
Hybrid classical output layer
        |
        v
Flask Dashboard
Traffic predictions + timings
```

The model is trained separately before the MQTT demonstration. Its learned parameters are saved locally and loaded by the inference module when the receiver starts.

## Turning it on

Hosted in machine 10.1.32.1

Turning on mosquitto
```bash
docker start mosquitto
```

Open **TWO** separate terminal windows and run the following commands:

### Terminal 1. Start the Receiver
```bash
cd /home/work/quantum-traffic-mqtt
source .venv/bin/activate
python quantum_receiver.py
```
### Terminal 2. Start the Sender

```bash
cd /home/work/quantum-traffic-mqtt
source .venv/bin/activate
python quantum_sender.py
```

Open Dashboard at http://localhost:5119/ or http://10.1.32.1:5119/ if on ssh

## Technology stack

- Python
- PennyLane
- Amazon Braket SDK and PennyLane plugin
- NumPy, pandas, and scikit-learn utilities
- Paho MQTT
- Flask
- Chart.js

## Dataset

This project uses the [TrafficVolumeDataCSV dataset on Kaggle](https://www.kaggle.com/datasets/bobaaayoung/trafficvolumedatacsv/data).

The dataset contains historical traffic volume observations alongside weather and environmental information.

Download the CSV and place it in the repository root as:

```text
TrafficVolumeData.csv
```

The dataset is not included in this repository. Please consult its source page for applicable terms.

## Model overview

The prototype predicts `traffic_volume` using four input features:

- Hour of the day, derived from `date_time`
- Humidity
- Temperature
- Cloud coverage (`clouds_all`)

The quantum circuit uses four qubits, rotation gates to encode inputs and trainable parameters, and CNOT gates to introduce entanglement. Pauli-Z expectation values from the circuit feed a classical linear output layer to produce a traffic-volume prediction.

This is a small experimental model intended to demonstrate a quantum machine learning workflow. Its results should not be interpreted as evidence that quantum machine learning outperforms classical methods or is ready for operational traffic forecasting.

## Training results

In the development run, the model was trained for five epochs using a limited subset of the dataset.

| Metric | Development result |
|---|---:|
| Training epochs | 5 |
| Training examples | Up to 128 |
| Test examples | 32 |
| Test MAE | 1,031.77 vehicles |
| Test R² | 0.5173 |

These figures describe one prototype run and are not a comprehensive performance evaluation. Results may vary with changes to the data, parameters, or software environment.

## Requirements

- Python 3.10
- An MQTT broker reachable at `127.0.0.1:1884`
- The dataset described above
- Internet access in the browser to load the Chart.js CDN

The Python dependencies are listed in `requirements.txt`.

## Setup

Clone the repository:

```bash
git clone https://github.com/aimanhaizal/quantum-traffic-mqtt.git
cd quantum-traffic-mqtt
```

Create and activate a virtual environment on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Download `TrafficVolumeData.csv` from the dataset source and place it in the repository root.

Ensure that an MQTT broker is running and reachable at `127.0.0.1:1884`. The sender and receiver use the `traffic/data` topic.

## Run the project

Run the following steps in order, using separate terminals where indicated.

### 1. Train the quantum model

With the virtual environment active and dataset in place:

```bash
python quantum_model.py
```

This trains the hybrid quantum-classical regressor, evaluates it on a small test set, and saves the learned parameters to `quantum_model_params.npz`.

The generated model file is local and is not included in the repository. Run training again when setting up a fresh copy.

### 2. Start the MQTT receiver and dashboard

In a second terminal, activate the same virtual environment and run:

```bash
python quantum_receiver.py
```

The receiver subscribes to `traffic/data`, performs predictions, logs results to `quantum_results.csv`, and serves the Flask dashboard.

Open the dashboard at:

[http://localhost:5119/](http://localhost:5119/)

If accessing it from another machine, use the host's reachable IP address and port `5119`.

### 3. Start the dataset sender

In a third terminal, activate the virtual environment and run:

```bash
python quantum_sender.py
```

The sender publishes each valid row from the CSV at approximately one-second intervals. It reports publishing progress in the terminal.

Use **Ctrl+C** to stop the replay when desired. The receiver and dashboard can be stopped separately.

## Dashboard and result logging

The dashboard presents:

- Total messages received during the current receiver session
- Average, median, and maximum inference time over the retained recent records
- Actual versus predicted traffic volume
- Inference runtime for individual messages
- A table of the latest records

The dashboard keeps a rolling window of recent messages in memory, while `quantum_results.csv` stores processed results on disk. The CSV is append-only in the current prototype, so previous experiment records remain unless the file is manually cleared.

Inference timing measures the prediction path, while processing timing includes additional receiver-side work. The first inference may take longer than subsequent calls due to initialization.

## Project files

```text
quantum-traffic-mqtt/
├── quantum_model.py
├── quantum_inference.py
├── quantum_receiver.py
├── quantum_sender.py
├── sender.py
├── quantum_circuit_test.py
├── gradient_test.py
├── test_inference.py
├── requirements.txt
├── .gitignore
├── README.md
├── templates/
│   └── quantum_index.html
└── screenshots/
    └── quantum-dashboard.png
```

The exact file list may include additional configuration or test files.

## Scope and limitations

This is a prototype for demonstrating the integration of quantum machine learning with an MQTT data pipeline.

- Quantum circuits run on the local Braket simulator; no remote quantum hardware or cloud quantum jobs are required.
- The model is trained on a small subset for demonstration rather than exhaustive optimization.
- The dashboard compares predictions with known historical traffic values in the replayed dataset; it is not a live traffic sensor.
- Inference times depend on the local machine, simulator, and execution conditions.
- The dashboard's Chart.js graphs require browser access to the external CDN.

## License

This repository's code is released under the **MIT License**. See [`LICENSE`](LICENSE). The Kaggle dataset is a separate work; refer to its source page for applicable terms.
