import pandas as pd
import numpy as np
import requests
import datetime
import json

API_URL = "http://8.231.86.107/predict"  # replace with your LoadBalancer IP

# 1. Generate random dataset
np.random.seed(42)
data = pd.DataFrame({
    "sno": np.arange(1, 101),
    "age": np.random.randint(30, 80, 100),
    "gender": np.random.randint(0, 2, 100),  # 0=female, 1=male
    "cp": np.random.randint(0, 4, 100),
    "trestbps": np.random.randint(100, 180, 100),
    "chol": np.random.randint(150, 300, 100),
    "fbs": np.random.randint(0, 2, 100),
    "restecg": np.random.randint(0, 2, 100),
    "thalach": np.random.randint(100, 200, 100),
    "exang": np.random.randint(0, 2, 100),
    "oldpeak": np.round(np.random.uniform(0, 5, 100), 1),
    "slope": np.random.randint(0, 3, 100),
    "ca": np.random.randint(0, 4, 100),
    "thal": np.random.randint(1, 4, 100)
})

# Save input dataset (Deliverable 5 proof)
data.to_csv("deliverable5_dataset.csv", index=False)

# 2. Loop through rows and call API
results = []
for _, row in data.iterrows():
    payload = row.to_dict()
    try:
        response = requests.post(API_URL, json=payload)
        result = response.json()
    except Exception as e:
        result = {"error": str(e)}

    # 3. Log entry
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": payload,
        "prediction": result.get("prediction"),
        "probability": result.get("probability")
    }
    results.append(log_entry)
    print(log_entry, flush=True)

# 4. Save predictions as CSV (flattened)
output_df = pd.DataFrame([
    {**entry["input"], "timestamp": entry["timestamp"],
     "prediction": entry["prediction"], "probability": entry["probability"]}
    for entry in results
])
output_df.to_csv("predictions_output.csv", index=False)

# 5. Save predictions as JSON (nested structure)
with open("predictions_output.json", "w") as f:
    json.dump(results, f, indent=2)
