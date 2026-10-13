
import json
import time

import pandas as pd
import paho.mqtt.client as mqtt


# Configuration
DATASET = "TrafficVolumeData.csv"
BROKER_HOST = "127.0.0.1"
BROKER_PORT = 1884
TOPIC = "traffic/data"

# Publish one message per second.
INTERVAL_SECONDS = 1.0


# Load and prepare the full dataset
print("Loading traffic dataset...")

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
).reset_index(drop=True)

total_rows = len(df)

if total_rows == 0:
    raise RuntimeError("No valid rows found in the dataset.")

print(f"Valid rows to publish: {total_rows:,}")
print(f"Publishing interval: {INTERVAL_SECONDS} seconds")


# Connect to MQTT
client = mqtt.Client(
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)

print(f"Connecting to {BROKER_HOST}:{BROKER_PORT}...")
client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
client.loop_start()

try:
    deadline = time.monotonic() + 5

    while not client.is_connected() and time.monotonic() < deadline:
        time.sleep(0.1)

    if not client.is_connected():
        raise RuntimeError("Could not connect to the MQTT broker.")

    print(f"Publishing the full dataset to {TOPIC}...")
    start_time = time.monotonic()

    # No row limit: iterate over every valid row.
    for position, row in enumerate(df.itertuples(index=False), start=1):
        message = {
            "date_time": row.date_time.isoformat(sep=" "),
            "hour": int(row.date_time.hour),
            "humidity": float(row.humidity),
            "temperature": float(row.temperature),
            "clouds_all": float(row.clouds_all),
            "traffic_volume": int(row.traffic_volume),
        }

        info = client.publish(
            TOPIC,
            json.dumps(message),
            qos=1,
        )
        info.wait_for_publish()

        if position % 100 == 0 or position == total_rows:
            elapsed = time.monotonic() - start_time
            rate = position / elapsed if elapsed > 0 else 0
            percent = position / total_rows * 100

            print(
                f"Progress: {position:,}/{total_rows:,} "
                f"({percent:.1f}%) | "
                f"Rate: {rate:.1f} messages/s | "
                f"Latest timestamp: {message['date_time']}"
            )

        time.sleep(INTERVAL_SECONDS)

    elapsed = time.monotonic() - start_time

    print("\nFull dataset replay completed.")
    print(f"Rows published: {total_rows:,}")
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Average publishing rate: {total_rows / elapsed:.2f} messages/s")

except KeyboardInterrupt:
    print("\nReplay interrupted by user.")

finally:
    client.loop_stop()
    client.disconnect()
    print("MQTT sender disconnected.")
