# 🚗 Vehicle Predictive Maintenance using ML + RAG + LLM

## 📌 Project Overview

This project implements an intelligent vehicle predictive maintenance system that combines:

- Machine Learning (Random Forest) for failure probability prediction
- Retrieval-Augmented Generation (RAG) for domain knowledge grounding
- Large Language Models (LLM) for contextual risk interpretation and explanation

The system is fully data-driven, non-rule-based, and designed to reflect real-world predictive maintenance workflows.

---

## 🎯 Problem Statement

In vehicle and industrial systems, failures are rare but costly.
Traditional rule-based systems struggle to capture complex interactions between sensor variables.

This project aims to:
- Predict failure probability from historical sensor data
- Interpret risk contextually (no fixed thresholds)
- Generate human-readable maintenance explanations
- Avoid hard-coded rules entirely

---

## 🧠 System Architecture

Historical Sensor Data
↓
Machine Learning Model (Random Forest)
↓
Failure Probability
↓
RAG (Maintenance Knowledge Base)
↓
LLM Reasoning Engine
↓
Risk Level (LOW / MEDIUM / HIGH)

Explanation


---

## 🔍 Key Design Principles

- ❌ No rule-based thresholds (e.g., no “if torque > X”)
- ✅ Risk emerges from probability + context
- ✅ LLM performs interpretation, not prediction
- ✅ Knowledge grounding via RAG
- ✅ Hardware-aware design

---

## 📂 Project Structure

vehicle_failure_prediction/
│
├── data/
│ └── predictive_maintenance.csv
│
├── models/
│ └── random_forest.py
│
├── rag/
│ ├── knowledge_base.txt
│ └── vector_store.py
│
├── llm/
│ ├── llm_client.py
│ └── decision_engine.py
│
├── pipeline/
│ └── inference_pipeline.py
│
├── main.py
└── README.md


---

## 📊 Dataset

- Predictive maintenance sensor dataset
- Features include:
  - Air temperature [K]
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
- Target: Machine failure (rare event)

The dataset is naturally imbalanced, reflecting real operational conditions.

---

## ⚙️ How the System Works

1. Random Forest predicts failure probability
2. RAG retrieves relevant maintenance knowledge
3. LLM interprets probability + context
4. Final risk and explanation are generated

No hard thresholds are used.

---

## 🧪 Example Output

FINAL RISK: MEDIUM

FINAL EXPLANATION:
The failure probability suggests an elevated risk compared to normal operation.
While no immediate failure is indicated, the combination of increased tool wear
and thermal stress may accelerate component degradation. Preventive inspection
is recommended.


---

## 🚀 How to Run

### Install dependencies
```bash

pip install -r requirements.txt

Start Ollama

ollama serve

Pull a lightweight model:



ollama pull phi3:mini

Run the project

python main.py

🧩 Why This Is Not Rule-Based

Component	Role

ML Model	Learns from data

Probability	Continuous uncertainty

RAG	Knowledge grounding

LLM	Contextual reasoning

Code	No decision rules

⚠️ Performance Note

Local LLM inference may take 20–40 seconds on CPU



This is a hardware limitation



In production, LLM explanations are typically on-demand or API-based



📈 Future Improvements



Time-series degradation modeling



Batch reporting



FastAPI backend



Streamlit dashboard



Confidence visualization