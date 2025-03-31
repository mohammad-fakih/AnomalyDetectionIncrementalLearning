# **Anomaly Detection with Incremental Learning**

This repository contains an implementation of a **Bidirectional LSTM Autoencoder** trained incrementally on the **SWaT dataset** for anomaly detection. The goal is to evaluate its performance in detecting anomalies while adapting to new data.

This work is part of a larger project exploring **incremental learning** scenarios, including:
- **Mitigating catastrophic forgetting**
- **Experimenting with different models** such as **Interfusion** and **USSAD**

## **Getting Started**

### **1. Data Visualization**
Begin with the **data visualization notebook**, which provides a basic exploration of the dataset.

### **2. Main Training & Evaluation**
Move on to the **`final` notebook**, which contains the core implementation.

### **3. Understanding the Code**
- The **`final` notebook** references various functions—navigate to the corresponding Python files to understand their implementation.

## **Repository Structure**
- `notebooks/` → Jupyter notebooks for training & evaluation
- `data/` → SWaT dataset
- `src/` → Python scripts for model training, evaluation, and utilities

## **Requirements**
Ensure you have the necessary dependencies installed before running the code:
```bash
pip install -r requirements.txt
```

## **Usage**
Run the main notebook:
```bash
jupyter notebook notebooks/final.ipynb
```

---
This README provides clear guidance and structure for users exploring your project. 🚀
