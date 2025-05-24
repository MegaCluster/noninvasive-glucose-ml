# A Noninvasive Blood Glucose Monitoring System using Machine Learning

## 🔬 Project Overview
This project implements deep learning models to estimate resonant frequency and transmission magnitude (S21) from a FC-CSRR sensor for noninvasive blood glucose monitoring.

## Models Implemented
- Single-Step Model (S-FM)
- Double-Step Model (D-F and D-M)
- DNNs with Batch Normalization, Dropout, and ResNet

## Dataset
- `C1024.xlsx`: Experimental Network Analyzer measurements with 1024 sensor configurations.

## How to Run
```bash
# Create environment
pip install -r requirements.txt

# Run notebooks
jupyter notebook notebooks/

# Launch Streamlit App
streamlit run app/streamlit_app.py
