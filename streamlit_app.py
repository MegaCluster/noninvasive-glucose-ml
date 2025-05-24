import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# === Load the model ===
model = tf.keras.models.load_model("models/single_step_model.keras")

# === Scaler stats from training ===
SCALER_MEAN = [2.026, 4.155, 5.737, 8.484]
SCALER_STD = [0.395, 0.462, 0.413, 0.382]

def scale_input(r_values):
    return [(val - mean) / std for val, mean, std in zip(r_values, SCALER_MEAN, SCALER_STD)]

# === Streamlit UI ===
st.set_page_config(page_title="FC-CSRR Sensor Predictor", layout="centered")
st.title("🔬 FC-CSRR Sensor Predictor")
st.markdown("Estimate **Resonant Frequency** and **S21 Magnitude (dB)** using input parameters.")

# Input sliders
r1 = st.slider("Ring 1 Radius (r1)", 1.0, 3.0, 2.0, 0.05)
r2 = st.slider("Ring 2 Radius (r2)", 3.0, 5.0, 4.0, 0.05)
r3 = st.slider("Ring 3 Radius (r3)", 5.0, 7.0, 6.0, 0.05)
r4 = st.slider("Ring 4 Radius (r4)", 7.5, 9.5, 8.5, 0.05)

if st.button("📈 Predict and Visualize"):
    input_values = [r1, r2, r3, r4]
    scaled_input = np.array([scale_input(input_values)])
    prediction = model.predict(scaled_input)
    freq, s21 = prediction[0]

    st.success(f"📡 Predicted Resonant Frequency: **{freq:.3f} GHz**")
    st.success(f"📉 Predicted S21 Magnitude: **{s21:.2f} dB**")

    # Plot prediction
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["Resonant Freq (GHz)", "S21 (dB)"], [freq, s21], color=["#1f77b4", "#ff7f0e"])
    ax.set_title("Predicted Sensor Output")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)

    # Additional scatter
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.scatter(freq, s21, color="purple", s=100)
    ax2.set_xlabel("Frequency (GHz)")
    ax2.set_ylabel("S21 (dB)")
    ax2.set_title("Sensor Output Position")
    ax2.grid(True)
    st.pyplot(fig2)

    st.caption("Visualized from trained DNN model. Results vary based on input design parameters.")
