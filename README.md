# 🚕 UrbanRide Fare Prediction — MLOps Pipeline

This project implements a complete MLOps pipeline for predicting ride fare (`fare_amount`) using machine learning.

---

## 📁 Project Structure

---

## ⚙️ Tasks Implemented   👈 👉 ADD YOUR CONTENT HERE

### 🔹 Task 1 — Model Training & Comparison
- Models: LinearRegression, Ridge
- Metrics: MAE, RMSE, R², MAPE
- Best model selected based on RMSE

---

### 🔹 Task 2 — Hyperparameter Tuning
- RandomizedSearchCV
- 5-fold Cross Validation
- Best parameters selected using RMSE

---

### 🔹 Task 3 — FastAPI Deployment
- `/heartbeat` → health check
- `/infer` → prediction endpoint
- Runs on port 8888

---

### 🔹 Task 4 — Retraining Pipeline
- Combines old + new data
- Retrains model
- Promotes only if MAE improves ≥ 0.3

---

## 🚀 How to Run
