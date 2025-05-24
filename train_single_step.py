import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# Load data
data_path = 'data/cleaned_valleys.csv'
df = pd.read_csv(data_path)

# Features (X) and Targets (Y)
feature_cols = ['r1', 'r2', 'r3', 'r4']  # Add more if needed
target_cols = ['Resonant_Freq_GHz', 'S21_dB']

X = df[feature_cols].values
y = df[target_cols].values

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build DNN model
def build_model(input_dim, output_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model(X.shape[1], y.shape[1])

# Train with early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"\n✅ RMSE: {rmse:.4f}")
print(f"✅ R²: {r2:.4f}")
print(f"✅ MAPE: {mape * 100:.2f}%")

# Save model (modern format)
os.makedirs("models", exist_ok=True)
model.save("models/single_step_model.keras")
print("✅ Model saved to models/single_step_model.keras")

# Plot loss curves
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("models/loss_curve.png")
print("📈 Loss curve saved to models/loss_curve.png")

# Plot predicted vs actual
plt.figure(figsize=(12, 5))
for i, target in enumerate(target_cols):
    plt.subplot(1, 2, i+1)
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"Actual {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"{target}: Actual vs Predicted")
    plt.grid(True)
plt.tight_layout()
plt.savefig("models/pred_vs_actual.png")
print("📈 Prediction scatter plot saved to models/pred_vs_actual.png")
