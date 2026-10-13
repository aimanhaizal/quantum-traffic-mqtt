
import csv
import os
import json
import threading
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from flask import Flask, jsonify, render_template

from quantum_inference import predict_traffic


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1884
TOPIC = "traffic/data"
WEB_HOST = "0.0.0.0"
WEB_PORT = 5119
MAX_RECORDS = 50

app = Flask(__name__)

# records = []
# records_lock = threading.Lock()
records = []
records_lock = threading.Lock()
total_messages_received = 0

RESULTS_FILE = "quantum_results.csv"

CSV_FIELDS = [
    "date_time",
    "actual",
    "predicted",
    "inference_ms",
    "processing_ms",
]

# Create the CSV with a header if it doesn't already exist.
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

# --------------------------------------------------
# MQTT callbacks
# --------------------------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
        print(f"Subscribed to {TOPIC}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, msg):
    global total_messages_received
    received_at = time.perf_counter()

    try:
        data = json.loads(msg.payload.decode("utf-8"))

        date_time = data.get("date_time", "Unknown")
        hour = data.get("hour")

        if hour is None:
            hour = datetime.fromisoformat(date_time).hour

        prediction, inference_ms = predict_traffic(
            hour=float(hour),
            humidity=float(data["humidity"]),
            temperature=float(data["temperature"]),
            clouds_all=float(data["clouds_all"]),
        )

        processing_ms = (time.perf_counter() - received_at) * 1000
        actual = data.get("traffic_volume")

        record = {
            "date_time": date_time,
            "actual": float(actual) if actual is not None else None,
            "predicted": round(float(prediction), 2),
            "inference_ms": round(float(inference_ms), 2),
            "processing_ms": round(float(processing_ms), 2),
        }

        # Persist each processed prediction for later review.
        with open(RESULTS_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writerow(record)

        # with records_lock:
        #     records.append(record)
        #     if len(records) > MAX_RECORDS:
        #         records.pop(0)
        
        with records_lock:
            records.append(record)
            total_messages_received += 1

            if len(records) > MAX_RECORDS:
                records.pop(0)

        print(
            f"{date_time} | predicted={prediction:.0f} | "
            f"inference={inference_ms:.2f} ms"
        )

    except Exception as exc:
        print(f"Could not process MQTT message: {exc}")


# --------------------------------------------------
# Web dashboard
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("quantum_index.html")


@app.route("/api/data")
def get_data():
    with records_lock:
        snapshot = list(records)
        total_received = total_messages_received

    runtimes = [r["inference_ms"] for r in snapshot]

    if runtimes:
        sorted_runtimes = sorted(runtimes)
        middle = len(sorted_runtimes) // 2

        if len(sorted_runtimes) % 2:
            median_runtime = sorted_runtimes[middle]
        else:
            median_runtime = (
                sorted_runtimes[middle - 1] + sorted_runtimes[middle]
            ) / 2

        average_runtime = sum(runtimes) / len(runtimes)
        max_runtime = max(runtimes)
    else:
        average_runtime = 0
        median_runtime = 0
        max_runtime = 0

    return jsonify({
        "records": snapshot,
        # "count": len(snapshot),
        "count": total_received,
        "window_count": len(snapshot),
        "average_runtime_ms": round(average_runtime, 2),
        "median_runtime_ms": round(median_runtime, 2),
        "max_runtime_ms": round(max_runtime, 2),
    })


# --------------------------------------------------
# Start MQTT in background, then start Flask
# --------------------------------------------------

if __name__ == "__main__":
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2
    )
    client.on_connect = on_connect
    client.on_message = on_message

    print("Starting quantum MQTT receiver...")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()

    print(f"Quantum dashboard will be available on port {WEB_PORT}")

    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            debug=False,
            use_reloader=False,
        )
    finally:
        client.loop_stop()
        client.disconnect()
