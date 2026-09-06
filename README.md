# MLOPS - OPPE 2 — Heart Disease Prediction & MLOps Deployment

## Overview

This project implements an end-to-end **Machine Learning Operations (MLOps) pipeline** for a Heart Disease Prediction model.

The project covers the complete lifecycle of a machine learning model:

* Model training and evaluation
* Model explainability using **SHAP**
* Fairness analysis using **Fairlearn**
* Dockerized REST API using **Flask**
* Deployment on **Google Kubernetes Engine (GKE)**
* Kubernetes service and autoscaling
* CI/CD using **GitHub Actions**
* Per-sample prediction logging
* Observability using **GCP Cloud Logging**
* API performance and stress testing using **wrk**
* Input drift detection using statistical tests

The model used for prediction is a **Logistic Regression** model optimized using `RandomizedSearchCV`.

---

# Project Architecture

```text
                         ┌─────────────────────┐
                         │   Heart Disease     │
                         │      Dataset        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Model Training      │
                         │ Logistic Regression │
                         │ RandomizedSearchCV  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   SHAP Explainability│
                         │   & Fairness Test   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Save Model          │
                         │ .joblib             │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Flask REST API      │
                         │ /health             │
                         │ /predict            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Docker Container    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Google Artifact     │
                         │ Registry            │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Google Kubernetes   │
                         │ Engine (GKE)        │
                         └──────────┬──────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  ┌──────────────┐      ┌──────────────┐
                  │ Kubernetes   │      │ HPA          │
                  │ Service      │      │ 1 → 3 Pods   │
                  └──────────────┘      └──────────────┘
                         │
                         ▼
                  ┌──────────────┐
                  │ Cloud        │
                  │ Logging      │
                  └──────────────┘
```

---

# Dataset

The project uses a Heart Disease dataset containing **303 training samples**.

The input features used by the model are:

| Feature    | Description                       |
| ---------- | --------------------------------- |
| `sno`      | Sample/index identifier           |
| `age`      | Patient age                       |
| `gender`   | Patient gender                    |
| `cp`       | Chest pain type                   |
| `trestbps` | Resting blood pressure            |
| `chol`     | Serum cholesterol                 |
| `fbs`      | Fasting blood sugar               |
| `restecg`  | Resting ECG result                |
| `thalach`  | Maximum heart rate achieved       |
| `exang`    | Exercise-induced angina           |
| `oldpeak`  | ST depression                     |
| `slope`    | Slope of peak exercise ST segment |
| `ca`       | Number of major vessels           |
| `thal`     | Thalassemia-related feature       |

The target variable is:

```text
target
```

with the classes:

```text
yes
no
```

---

# Deliverable 1 — Private Git Repository

A private GitHub repository was created using the required naming convention:

```text
22F1001871_IITMBS_MLOPS_OPPE2_MAY_2026
```

The repository contains the complete implementation of the MLOps pipeline.

The required course collaborator access was also configured for the repository.

---

# Deliverable 2 — Model Explainability using SHAP

## Objective

SHAP (SHapley Additive exPlanations) was used to understand how different features influence the model's predictions.

The mean absolute SHAP value was calculated for every feature.

### SHAP results

| Feature    | Mean Absolute SHAP |
| ---------- | -----------------: |
| `sno`      |          27.165201 |
| `thalach`  |           3.509515 |
| `trestbps` |           2.405063 |
| `age`      |           1.929628 |
| `oldpeak`  |           0.972467 |
| `chol`     |           0.953702 |
| `cp`       |           0.342298 |
| `slope`    |           0.269709 |
| `restecg`  |           0.252841 |
| `ca`       |           0.125861 |
| `fbs`      |           0.013764 |
| `gender`   |           0.012138 |
| `exang`    |           0.007325 |
| `thal`     |           0.007012 |

## Least impactful features

The three features with the lowest mean absolute SHAP values were:

```text
gender : 0.012138
exang  : 0.007325
thal   : 0.007012
```

Therefore, **`thal` had the least influence on the model's predictions**, with a mean absolute SHAP value of approximately **0.0070**.

In plain English, this means that, among the features considered by the model, changes in `thal` contributed the least to changes in the model's prediction compared with the other features.

> Note: `sno` is an identifier-like feature. Although it produced the highest SHAP value, it is not a meaningful medical characteristic. It was retained because it was part of the provided team's notebook and model pipeline.

---

# Deliverable 3 — Fairness Testing using Fairlearn

## Objective

Fairness testing was performed using **Fairlearn**, with **age** as the sensitive attribute.

The age variable was divided into the following groups:

```text
<40
40-49
50-59
60+
```

The model's selection rates were:

| Age Group | Selection Rate |
| --------- | -------------: |
| `<40`     |         1.0000 |
| `40-49`   |         0.6316 |
| `50-59`   |         0.5909 |
| `60+`     |         0.4667 |

### Fairness metrics

```text
Demographic Parity Difference = 0.5333
Equalized Odds Difference     = 0.1000
```

### Interpretation

The **Demographic Parity Difference of 0.5333** indicates a noticeable difference in positive prediction rates between the age groups.

The `<40` group had the highest positive prediction rate at **100%**, while the `60+` group had the lowest at approximately **46.67%**.

The **Equalized Odds Difference of 0.1** indicates a comparatively smaller difference in the model's true-positive and false-positive behavior across the age groups.

Overall, the fairness analysis indicates that the model has **noticeable disparity in positive prediction rates across age groups**, although the equalized-odds difference is relatively small.

---

# Deliverable 4 — Dockerized API Deployment on GKE

## Objective

The trained heart disease prediction model was converted into a REST API using **Flask**, containerized using **Docker**, and deployed on **Google Kubernetes Engine (GKE)**.

## Flask API

The API provides two endpoints.

### Health Check

```text
GET /health
```

Example response:

```json
{
    "status": "healthy"
}
```

### Prediction

```text
POST /predict
```

The API accepts the 14 model input features as JSON and returns:

* Prediction
* Prediction probability
* Timestamp

Example:

```json
{
    "sno": 0,
    "age": 63,
    "gender": 1,
    "cp": 3,
    "trestbps": 145.0,
    "chol": 233.0,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150.0,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
}
```

The model is loaded from:

```text
api/model/heart_disease_model.joblib
```

## Docker

The Flask application and model were packaged into a Docker container.

The Docker image was built and pushed to **Google Artifact Registry**.

## GKE

The Dockerized API was deployed to a GKE cluster:

```text
heart-disease-cluster
```

The Kubernetes deployment uses:

```text
Deployment
Service
Horizontal Pod Autoscaler (HPA)
```

The API is exposed through a Kubernetes `LoadBalancer` service.

## Autoscaling

The Horizontal Pod Autoscaler was configured as:

```text
Minimum replicas: 1
Maximum replicas: 3
Target CPU utilization: 70%
```

Therefore, the deployment can automatically scale from **1 pod up to a maximum of 3 pods** depending on CPU utilization.

## CI/CD

GitHub Actions was configured to automate deployment.

The CI/CD pipeline performs the following:

```text
Git push
   ↓
GitHub Actions
   ↓
Build Docker image
   ↓
Push image to Artifact Registry
   ↓
Authenticate with GKE
   ↓
Deploy/update Kubernetes deployment
```

This allows new versions of the application to be deployed automatically when changes are pushed to the repository.

---

# Deliverable 5 — Per-Sample Prediction, Logging & Observability

## Objective

A **100-row random dataset** was generated and used to test the deployed API.

Each row was sent to the API **individually**, rather than sending the complete dataset as a single request.

The workflow was:

```text
100-row dataset
      ↓
Individual API request
      ↓
Prediction
      ↓
Prediction + probability + timestamp
      ↓
Structured container log
      ↓
GCP Cloud Logging
```

## Prediction Logging

Each API request generates a structured log containing:

```json
{
    "timestamp": "...",
    "input_features": {
        "...": "..."
    },
    "prediction": "yes",
    "probability": 0.91
}
```

The logs are written to container stdout and collected by GKE's logging infrastructure.

## Observability

The prediction logs can be inspected through:

```text
Google Cloud Console
        ↓
Logging
        ↓
Logs Explorer
```

This provides centralized visibility into individual prediction requests made to the deployed API.

The generated prediction results are stored in:

```text
prediction_results_100.csv
```

---

# Deliverable 6 — Performance Monitoring & Stress Testing

## Objective

The deployed Heart Disease Prediction API was stress tested using **wrk** to evaluate its performance under a high-concurrency workload.

The assignment required more than 2,000 concurrent connections. The test was performed using **2,000 concurrent connections**, which satisfies the high-concurrency requirement.

### Stress Test Command

```bash
wrk -t12 -c2000 -d30s -s post.lua http://8.231.86.107/predict
```

The parameters used were:

| Parameter |      Value | Description                               |
| --------- | ---------: | ----------------------------------------- |
| `-t`      |         12 | Number of wrk threads                     |
| `-c`      |       2000 | Number of concurrent connections          |
| `-d`      |        30s | Test duration                             |
| `-s`      | `post.lua` | Lua script used to generate POST requests |
| Endpoint  | `/predict` | Heart disease prediction API              |

### Stress Test Results

```text
Threads       : 12
Connections   : 2000
Duration      : 30 seconds

Average Latency : 1.35 seconds
Latency Std Dev : 403 ms
Maximum Latency : 2.00 seconds

Requests/sec    : 78.12
Total Requests  : 2350

Timeouts        : 1108
```

### Performance Analysis

The API processed approximately **78.12 requests per second** during the 30-second stress test, with **2,350 requests** completed during the test period.

The average latency was approximately **1.35 seconds**, while the latency standard deviation was **403 ms**, indicating noticeable variation in response times under heavy concurrent load. The maximum observed latency was approximately **2 seconds**.

A total of **1,108 requests timed out** during the test. This indicates that the API experienced significant performance degradation when subjected to 2,000 simultaneous connections.

The results demonstrate that while the API was able to continue processing requests under high concurrency, the deployed service was **resource-constrained under the tested workload**, resulting in increased latency and a substantial number of timeouts.

The stress test therefore provided evidence of the API's behavior under extreme load and can also be used to evaluate the effectiveness of the Kubernetes autoscaling configuration.

### Summary

| Metric                     |                 Result |
| -------------------------- | ---------------------: |
| Concurrent connections     |              **2,000** |
| Threads                    |                 **12** |
| Test duration              |         **30 seconds** |
| Total requests             |              **2,350** |
| Throughput                 | **78.12 requests/sec** |
| Average latency            |           **1.35 sec** |
| Latency standard deviation |             **403 ms** |
| Maximum latency            |           **2.00 sec** |
| Timeouts                   |              **1,108** |

### Conclusion

The stress test shows that the deployed API can handle requests under a high-concurrency workload, but performance deteriorates considerably at 2,000 concurrent connections. The relatively high latency and **1,108 timeouts** indicate that additional resources or further performance optimization would be required to reliably handle workloads of this magnitude.


# Deliverable 7 — Input Drift Detection

## Objective

Input drift was detected by comparing the original training dataset with the **same 100-row dataset used for Deliverable 5**.

Dataset sizes:

```text
Training samples : 303
Incoming samples : 100
Features tested  : 14
```

### Statistical tests

Two statistical tests were used:

### Numerical features

The **Kolmogorov-Smirnov (KS) test** was used for:

```text
sno
age
trestbps
chol
thalach
oldpeak
```

### Categorical features

The **Chi-square test** was used for:

```text
gender
cp
fbs
restecg
exang
slope
ca
thal
```

A significance level of:

```text
α = 0.05
```

was used.

Therefore:

```text
p-value < 0.05  → Drift detected
p-value >= 0.05 → No significant drift
```

## Results

| Feature    | Test       | Statistic |  p-value | Drift |
| ---------- | ---------- | --------: | -------: | :---: |
| `sno`      | KS         |  0.666667 | 1.33e-32 |  Yes  |
| `age`      | KS         |  0.203894 | 3.16e-03 |  Yes  |
| `trestbps` | KS         |  0.182742 | 1.09e-02 |  Yes  |
| `chol`     | KS         |  0.172119 | 2.02e-02 |  Yes  |
| `thalach`  | KS         |  0.191477 | 6.84e-03 |  Yes  |
| `oldpeak`  | KS         |  0.345776 | 1.70e-08 |  Yes  |
| `gender`   | Chi-square |  3.775515 | 5.20e-02 |   No  |
| `cp`       | Chi-square | 32.749833 | 3.64e-07 |  Yes  |
| `fbs`      | Chi-square | 65.610814 | 5.49e-16 |  Yes  |
| `restecg`  | Chi-square |  4.881607 | 8.71e-02 |   No  |
| `exang`    | Chi-square |  1.070333 | 3.01e-01 |   No  |
| `slope`    | Chi-square | 39.311266 | 2.91e-09 |  Yes  |
| `ca`       | Chi-square | 46.856151 | 1.63e-09 |  Yes  |
| `thal`     | Chi-square | 54.400236 | 9.22e-12 |  Yes  |

### Drift summary

```text
Training samples      : 303
Incoming samples      : 100
Features tested       : 14
Features with drift   : 11
```

### Conclusion

Input drift was detected in **11 of the 14 tested features**.

Statistically significant drift was observed in:

```text
sno
age
trestbps
chol
thalach
oldpeak
cp
fbs
slope
ca
thal
```

No significant drift was detected for:

```text
gender
restecg
exang
```

The `sno` feature is an identifier/index-like variable, so its statistical drift should not be interpreted as a medical distribution shift.

---

# Technologies Used

| Technology               | Purpose                           |
| ------------------------ | --------------------------------- |
| Python                   | Model development and analysis    |
| Pandas                   | Data processing                   |
| NumPy                    | Numerical operations              |
| Scikit-learn             | Model training and evaluation     |
| Logistic Regression      | Classification model              |
| RandomizedSearchCV       | Hyperparameter optimization       |
| SHAP                     | Model explainability              |
| Fairlearn                | Fairness analysis                 |
| Flask                    | REST API                          |
| Joblib                   | Model serialization               |
| Docker                   | Containerization                  |
| Google Artifact Registry | Docker image storage              |
| Google Kubernetes Engine | Model deployment                  |
| Kubernetes               | Deployment and service management |
| HPA                      | Autoscaling                       |
| GitHub Actions           | CI/CD                             |
| GCP Cloud Logging        | Observability                     |
| wrk                      | Load/stress testing               |
| SciPy                    | Statistical drift detection       |

---

# Project File Structure

```text
22F1001871_IITMBS_MLOPS_OPPE2_MAY_2026/
│
├── api/
│   ├── prediction.py
│   └── requirements.txt
│   └── Dockerfile
│   └──model/
│           └── heart_disease_model.joblib
│
├── k8s/
│   ├── deployment.yaml
│   ├── service.yaml
│   └── hpa.yaml
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── notebook/
│   └── heart_disease_mlops.ipynb
│
├── DELIVERABLE-5
│        └──deliverable5_dataset.csv
│        └── predict_and_log.py
│        └── predictions_output.csv
│        └── predictions_output.json
│
├── DELIVERABLE-6
│        └──post.lua
│ 
├── prediction_results_100.csv
│
├── input_drift_results.csv
│
│
└── README.md
```

> The exact notebook filename may differ depending on the original notebook supplied for the assignment.

---

# End-to-End Workflow

The complete implementation can be summarized as:

```text
                    DATA
                     │
                     ▼
              Model Training
                     │
                     ▼
           Logistic Regression
                     │
                     ▼
       ┌─────────────┴─────────────┐
       │                           │
       ▼                           ▼
     SHAP                     Fairlearn
 Explainability               Fairness
       │                           │
       └─────────────┬─────────────┘
                     │
                     ▼
              Save Model
                     │
                     ▼
              Flask REST API
                     │
                     ▼
                  Docker
                     │
                     ▼
            Artifact Registry
                     │
                     ▼
                   GKE
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
         Kubernetes       HPA
          Service         1 → 3
              │
              ▼
          API Requests
              │
              ▼
      100 Individual Samples
              │
       ┌──────┴───────┐
       ▼              ▼
   Cloud Logging   Predictions
       │              │
       └──────┬───────┘
              ▼
        Drift Analysis
              │
              ▼
       Training vs Incoming
```

---

# Final Summary

This project implements an end-to-end MLOps solution for a Heart Disease Prediction model.

The implementation covers:

* **Explainability:** SHAP identified the least influential features, with `thal` having the lowest mean absolute SHAP value of approximately **0.0070**.
* **Fairness:** Fairlearn identified a **Demographic Parity Difference of 0.5333** and an **Equalized Odds Difference of 0.1** across age groups.
* **Deployment:** The model was converted into a Flask API, Dockerized, and deployed on **GKE**.
* **Autoscaling:** Kubernetes HPA was configured with a maximum of **3 pods**.
* **CI/CD:** GitHub Actions automates Docker image building and GKE deployment.
* **Observability:** 100 individual prediction requests are logged with their input features, predictions, probabilities, and timestamps and are observable through **GCP Cloud Logging**.
* **Performance:** The API was stress tested using `wrk` with high concurrency.
* **Drift Detection:** Statistical testing detected input drift in **11 of 14 features** when comparing the training data with the 100 incoming samples.

This provides a complete MLOps pipeline covering **model development, responsible AI, containerization, deployment, automation, monitoring, performance testing, and data drift detection**.
