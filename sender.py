
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt


DATASET = "TrafficVolumeData.csv"
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1884
TOPIC = "traffic/data"

INTERVAL_SECONDS = 1.0
MAX_ROWS = 50


df = pd.read_csv(DATASET)
df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
df = df.dropna(
    subset=[
        "date_time",
        "humidity",
        "temperature",
        "clouds_all",
        "traffic_volume",
    ]
)

client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)

print(f"Connecting to MQTT broker at {BROKER_HOST}:{BROKER_PORT}...")
client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
client.loop_start()

try:
    # Wait until the MQTT connection is ready.
    deadline = time.monotonic() + 5
    while not client.is_connected() and time.monotonic() < deadline:
        time.sleep(0.1)

    if not client.is_connected():
        raise RuntimeError("Could not connect to MQTT broker")

    print(f"Publishing to {TOPIC} every {INTERVAL_SECONDS} second(s)")

    for index, row in df.head(MAX_ROWS).iterrows():
        message = {
            "date_time": row["date_time"].isoformat(sep=" "),
            "hour": int(row["date_time"].hour),
            "humidity": float(row["humidity"]),
            "temperature": float(row["temperature"]),
            "clouds_all": float(row["clouds_all"]),
            "traffic_volume": int(row["traffic_volume"]),
        }

        info = client.publish(
            TOPIC,
            json.dumps(message),
            qos=1,
        )
        info.wait_for_publish()

        print(
            f"Published {index}: {message['date_time']} "
            f"| actual traffic = {message['traffic_volume']}"
        )

        time.sleep(INTERVAL_SECONDS)

    print("Finished sending dataset rows.")

except KeyboardInterrupt:
    print("\nSender stopped.")

finally:
    client.loop_stop()
    client.disconnect()
