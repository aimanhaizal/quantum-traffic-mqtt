
import json
import time

import paho.mqtt.client as mqtt

from quantum_inference import predict_traffic


BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1884
TOPIC = "traffic/data"


def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"Connected to MQTT broker at {BROKER_HOST}:{BROKER_PORT}")
        client.subscribe(TOPIC)
        print(f"Subscribed to {TOPIC}")
    else:
        print(f"MQTT connection failed: {reason_code}")


def on_message(client, userdata, msg):
    received_at = time.perf_counter()

    try:
        data = json.loads(msg.payload.decode("utf-8"))

        date_time = data.get("date_time", "Unknown")
        hour = data.get("hour")

        if hour is None:
            # Derive the hour from the CSV timestamp.
            from datetime import datetime
            hour = datetime.fromisoformat(date_time).hour

        prediction, inference_ms = predict_traffic(
            hour=float(hour),
            humidity=float(data["humidity"]),
            temperature=float(data["temperature"]),
            clouds_all=float(data["clouds_all"]),
        )

        callback_ms = (time.perf_counter() - received_at) * 1000
        actual = data.get("traffic_volume")

        print("\n--- Quantum Traffic Prediction ---")
        print(f"Time: {date_time}")

        if actual is not None:
            print(f"Actual traffic: {float(actual):.0f} vehicles")

        print(f"Predicted traffic: {prediction:.0f} vehicles")
        print(f"Quantum inference: {inference_ms:.2f} ms")
        print(f"Receiver processing: {callback_ms:.2f} ms")

    except Exception as exc:
        print(f"Could not process MQTT message: {exc}")


client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)
client.on_connect = on_connect
client.on_message = on_message

print("Starting quantum traffic receiver...")
client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nStopping receiver...")
finally:
    client.disconnect()
