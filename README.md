# Anomaly Detector in Network Traffic (UNSW-NB15)

## Overview
This project demonstrates how machine learning can be used to detect network anomalies in cybersecurity scenarios. It utilizes labeled network traffic data from the UNSW-NB15 dataset to classify connections as normal or malicious. The core of the project is a supervised Random Forest model trained to identify patterns associated with attacks.

## Dataset
The dataset used is the training portion of the [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) dataset. It includes both normal and attack traffic with a wide variety of features such as:
- Packet counts (source/destination)
- Bytes transferred (source/destination)
- Flow rates and durations
- Protocol, state, service types
- Derived TCP/IP flow features

**Label:**
- `0` – Normal traffic
- `1` – Malicious traffic (e.g. DoS, Exploits, Reconnaissance, etc.)

## Project Steps
1. **Data Cleaning and Preprocessing**
   - Removed irrelevant columns such as `id`, `attack_cat`
   - Encoded categorical variables (`proto`, `state`, `service`) using Label Encoding
   - Scaled numerical features using StandardScaler

2. **Model Training**
   - The dataset was split into training and testing sets (70/30)
   - A Random Forest Classifier with 100 trees was trained on the preprocessed data

3. **Evaluation**
   - Achieved **97.77% accuracy** on the test set
   - Confusion matrix showed high precision and recall on both classes
   - Feature importance analysis highlighted which metrics were most predictive (e.g. `dload`, `sload`, `dbytes`, `sbytes`)

## Key Technologies Used
- Python (pandas, scikit-learn, matplotlib)
- Jupyter Notebook / VS Code for experimentation

## Potential Improvements
- Add real-time detection via streaming or packet sniffing
- Incorporate explainable AI (XAI) methods to justify detections
- Visualize attack patterns over time or by category

## Usage
To run this project:
1. Clone the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run `anomaly_detector.py` to train or test the model
4. Use the saved model to predict anomalies on new traffic data

---
Created by: *Your Name*
Date: March 2025
