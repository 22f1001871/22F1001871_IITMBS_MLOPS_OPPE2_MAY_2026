# OPPE AI Usage Documentation 

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam  
> Provide **public share links** to AI chats or attach conversation files if links are not available  

- CHATGPT
    - Purpose : Intial coding generation
    - Shared Chat Link : https://chatgpt.com/share/6a9d488a-05e0-83e8-bdfb-2fe80d09412f
    - Notes (optional) :

- MICROSOT COPILOT
    - Purpose : For error solving and creation of Last three deliverables.
    - Shared Chat Link : https://copilot.microsoft.com/shares/a25GjZJCNb9fpdCPA8G3r
    - Notes (optional) :


*If a tool does not support link sharing, please specify and paste prompt-response logs in next Section*

---

## Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks

> Include **all responses** in case of public share links are not available to share

### Tool Name #1:  MICROSOT COPILOT
- Example Prompt 1:

Per-Sample Prediction with Logging & Observability
Generate a 100-row random dataset and run per-sample predictions through your deployed API.

Log each prediction request individually with its input features, predicted output, and timestamp.
Demonstrate observability using GCP Cloud Loggin

    - Response Log:

    Perfect — you’ve got your cluster ready, image pushed, and `kubectl` working. For **Deliverable 5**, here’s how you can implement **per‑sample prediction with logging & observability**:

---

## 🧩 Steps to Implement

### 1. Generate a random dataset
Use Python to create 100 rows of synthetic patient data with the same schema as your model:

```python
import pandas as pd
import numpy as np

FEATURES = [
    "sno", "age", "gender", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

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
```

---

### 2. Send per‑sample predictions to your API
Loop through each row and call your deployed FastAPI endpoint:

```python
import requests
import datetime

API_URL = "http://<EXTERNAL-IP>/predict"  # replace with LoadBalancer IP

for _, row in data.iterrows():
    payload = row.to_dict()
    response = requests.post(API_URL, json=payload)
    result = response.json()

    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "input": payload,
        "prediction": result.get("prediction"),
        "probability": result.get("probability")
    }
    print(log_entry)
```

---

### 3. Logging to GCP Cloud Logging
Your FastAPI app already prints logs to stdout:

```python
print({
    "timestamp": timestamp,
    "input": data.dict(),
    "prediction": str(prediction),
    "probability": probability
}, flush=True)
```

When deployed on GKE, **stdout/stderr are automatically collected by GCP Cloud Logging**.  
So every per‑sample prediction request will appear in Cloud Logging with its input, output, and timestamp.

---

### 4. Observability Demo
- Go to **Google Cloud Console → Logging → Logs Explorer**.  
- Query logs for your GKE workload:
  ```
  resource.type="k8s_container"
  resource.labels.cluster_name="heart-disease-cluster"
  resource.labels.container_name="heart-disease-api"
  ```
- You’ll see each prediction request logged individually.

---

## 🎯 Deliverable Checklist
- Generate random dataset ✅  
- Run per-sample predictions ✅  
- Log each request with input/output/timestamp ✅  
- Demonstrate observability in Cloud Logging ✅  

---

👉 With this setup, you’ll have **100 prediction requests**, each logged in Cloud Logging, proving observability and per‑sample tracking.  

Would you like me to also sketch a **Grafana/Prometheus integration** so you can visualize prediction counts and latencies beyond just Cloud Logging?

### Tool Name #2: CHATGPT
- Example Prompt 1:
 Model Explainability
Using explainability tools, describe in plain English the factors that have the least impact on predicting whether a patient has heart disease

    - Response Log:
  Great. Now we can proceed with **Deliverable 2: Model Explainability** using your actual dataset and the Logistic Regression model you showed.

Your dataset has these columns:

```text
sno, age, gender, cp, trestbps, chol, fbs, restecg,
thalach, exang, oldpeak, slope, ca, thal, target
```

The important point is that **`sno` should not be used as a feature**, and `target` is the output.

---

# Deliverable 2 — Model Explainability [10 Marks]

### Requirement

> Using explainability tools, describe in plain English the factors that have the least impact on predicting whether a patient has heart disease.

We'll use **SHAP** with your tuned Logistic Regression model.

The workflow is:

```text
Training data
     ↓
Logistic Regression
     ↓
SHAP
     ↓
Feature importance
     ↓
Identify least-impact features
     ↓
Plain-English explanation
```

---

## Step 1 — Install SHAP

In your notebook:

```python
!pip install shap -q
```

Then:

```python
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

---

# Step 2 — Check your trained model

You already have:

```python
from sklearn.model_selection import RandomizedSearchCV

log_reg_grid = {
    "C": np.logspace(-4, 4, 20),
    "solver": ["liblinear"]
}

rs_log_reg = RandomizedSearchCV(
    LogisticRegression(),
    param_distributions=log_reg_grid,
    cv=5,
    n_iter=20,
    verbose=True
)

rs_log_reg.fit(x_train, y_train)
```

You can get the best model using:

```python
best_model = rs_log_reg.best_estimator_

print(best_model)
print(rs_log_reg.best_params_)
```

You should get something similar to:

```text
LogisticRegression(C=..., solver='liblinear')
```

---

# Step 3 — Check your feature names

This is important.

Run:

```python
print(x_train.columns)
```

If `x_train` is a pandas DataFrame, you should see something like:

```text
Index(['age', 'gender', 'cp', 'trestbps', 'chol', 'fbs',
       'restecg', 'thalach', 'exang', 'oldpeak',
       'slope', 'ca', 'thal'],
      dtype='object')
```

If `sno` appears, **remove it**.

Your feature set should conceptually be:

```text
age
gender
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal
```

Not:

```text
sno
target
```

---

# Step 4 — Create the SHAP explainer

For Logistic Regression, we can use SHAP's linear explainer:

```python
explainer = shap.LinearExplainer(
    best_model,
    x_train
)

shap_values = explainer(x_train)
```

Now SHAP has calculated the contribution of every feature to the model's predictions.

---

# Step 5 — Generate the SHAP summary plot

Run:

```python
shap.summary_plot(
    shap_values,
    x_train
)
```

You'll get a plot similar to:

```text
Feature        importance
-------------------------
cp             █████████████
thalach        ███████████
ca             █████████
oldpeak        ███████
age            █████
...
fbs            ██
chol           █
```

The **exact ordering will depend on your trained model**.

---

# Step 6 — Calculate the actual feature importance

For your assignment, I recommend calculating the numbers explicitly rather than relying only on the graph.

Run:

```python
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

feature_importance = pd.DataFrame({
    "feature": x_train.columns,
    "mean_abs_shap": mean_abs_shap
})

feature_importance = feature_importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

print(feature_importance)
```

You'll get something like:

```text
      feature    mean_abs_shap
0           cp        0.52
1       thalach        0.41
2            ca        0.35
3       oldpeak        0.29
...
10          fbs        0.04
11         chol        0.03
12       slope        0.02
```

**Do not use these example numbers.**

Use the numbers produced by your notebook.

---

# Step 7 — Find the least impactful features

To directly answer the question:

```python
least_impact = feature_importance.tail(5)

print("Least impactful features:")
print(least_impact)
```

Or, if you want only the **single least impactful feature**:

```python
print(
    "Least impactful feature:",
    feature_importance.iloc[-1]["feature"]
)

print(
    "Mean absolute SHAP value:",
    feature_importance.iloc[-1]["mean_abs_shap"]
)
```

---

# Step 8 — Make a clean table for your submission

I recommend producing this:

```python
feature_importance_display = feature_importance.copy()

feature_importance_display["mean_abs_shap"] = (
    feature_importance_display["mean_abs_shap"].round(4)
)

print(feature_importance_display.to_string(index=False))
```

This gives you evidence such as:

```text
 feature    mean_abs_shap
 cp             0.XXXX
 thalach        0.XXXX
 ca             0.XXXX
 ...
 fbs            0.XXXX
 chol           0.XXXX
```

Take a screenshot of this output for your OPPE evidence.

---

# Step 9 — Plot the feature importance

For a clean visualization:

```python
plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["feature"],
    feature_importance["mean_abs_shap"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("SHAP Feature Importance - Heart Disease Model")

plt.tight_layout()
plt.show()
```

The important interpretation is:

> **Higher mean absolute SHAP value = greater influence on the model.**

and

> **Lower mean absolute SHAP value = lesser influence on the model.**

---

# Step 10 — Understand what SHAP is telling you

Suppose your result says:

```text
Feature      Mean |SHAP|
cp              0.51
thalach         0.42
ca              0.36
oldpeak         0.28
age             0.18
...
fbs             0.03
chol            0.02
```

Then:

* `cp` has a relatively large influence.
* `thalach` has a relatively large influence.
* `fbs` has a relatively small influence.
* `chol` has a relatively small influence.

So if `chol` has the smallest SHAP value, you can say:

> **Cholesterol has the least impact on the model's predictions among the features considered.**

But remember: **we need your actual SHAP output before making this statement.**

---

# Important distinction

Don't say:

> "Cholesterol does not cause heart disease."

That is **not** what SHAP tells us.

Instead say:

> "Cholesterol has relatively low influence on the predictions made by this particular trained model."

That's much more accurate.

SHAP explains **the model**, not medical causality.

---

# Step 11 — Generate the exact answer automatically

Once you've calculated `feature_importance`, you can run:

```python
least_feature = feature_importance.iloc[-1]

print(
    f"The feature with the least impact on the model's predictions "
    f"is '{least_feature['feature']}', with a mean absolute SHAP "
    f"value of {least_feature['mean_abs_shap']:.4f}."
)
```

You can also identify the bottom 3:

```python
print("Three least impactful features:")

for _, row in feature_importance.tail(3).iterrows():
    print(
        f"{row['feature']}: "
        f"{row['mean_abs_shap']:.4f}"
    )
```

---

# What to write for Deliverable 2

After you run the code, your final answer should follow this structure:

> **Model Explainability using SHAP**
>
> SHAP was used to measure the contribution of each input feature to the Logistic Regression model's heart-disease predictions. Feature importance was calculated using the mean absolute SHAP value across the training samples.
>
> The features with the lowest mean absolute SHAP values have the least influence on the model's predictions. According to the analysis, **[FEATURE 1]**, **[FEATURE 2]**, and **[FEATURE 3]** have the lowest impact.
>
> Therefore, these features contribute relatively little to the model's prediction compared with features having larger SHAP values. This interpretation describes the behavior of the trained model and should not be interpreted as evidence that these factors are medically unimportant.

### One thing before we move to Deliverable 3

**Run Steps 2–7 and send me the output of:**

```python
print(feature_importance)
```

Then I'll tell you **exactly which features are the least impactful** and help you formulate the 10-mark answer from your actual results.

After that we'll do **Deliverable 3 — Fairlearn using `age` as the sensitive attribute**, including the exact fairness metrics and code.
