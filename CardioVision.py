# cad_cnn_predictor.py

"""
🫀 CAD-CNN Predictor

A Convolutional Neural Network (Conv1D) model to predict Coronary Artery Disease (CAD)
using structured tabular health data.

Developed in: Google Colab (CSV upload required)
Libraries: pandas, scikit-learn, tensorflow, numpy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from google.colab import files


def load_and_preprocess_data():
    """Loads CSV from user upload and prepares data for CNN model."""
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]

    df = pd.read_csv(filename)

    X = df.drop('target', axis=1).values
    y = df['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for Conv1D input: (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    return X_train, X_test, y_train, y_test, scaler


def build_cnn_model(input_shape):
    """Constructs and compiles the CNN model."""
    model = Sequential([
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def predict_cad(new_data, model, scaler):
    """
    Predicts Coronary Artery Disease (CAD) from input features.

    Args:
        new_data (list or np.array): 1D or 2D array of shape (13,) or (1, 13)
        model (tf.keras.Model): Trained CNN model
        scaler (StandardScaler): Trained scaler

    Returns:
        str: Prediction result
    """
    new_data = np.array(new_data)

    if new_data.ndim == 1:
        new_data = new_data.reshape(1, -1)

    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = new_data_scaled.reshape(new_data_scaled.shape[0], new_data_scaled.shape[1], 1)

    prediction = model.predict(new_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return "Coronary Artery Disease Present" if predicted_class == 1 else "No Disease"


if __name__ == "__main__":
    # === STEP 1: Load and preprocess the data ===
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()

    # === STEP 2: Build, compile, and train the model ===
    model = build_cnn_model(input_shape=(X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

    # === STEP 3: Evaluate the model ===
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n🧪 Test Accuracy: {accuracy * 100:.2f}%")

    # === STEP 4: Make predictions on new patients ===
    sample_patient_1 = [57, 1, 2, 130, 236, 0, 1, 174, 0, 0.0, 1, 0, 2]
    sample_patient_2 = [58, 1, 3, 146, 371, 0, 2, 157, 1, 4.7, 2, 1, 2]

    print("\n📍 Prediction for Patient 1:", predict_cad(sample_patient_1, model, scaler))
    print("📍 Prediction for Patient 2:", predict_cad(sample_patient_2, model, scaler))
