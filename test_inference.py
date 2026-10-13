
from quantum_inference import predict_traffic

prediction, elapsed_ms = predict_traffic(
    hour=9,
    humidity=89,
    temperature=288.28,
    clouds_all=40,
)

print(f"Predicted traffic volume: {prediction:.0f} vehicles")
print(f"Quantum inference runtime: {elapsed_ms:.2f} ms")
