import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ✅ Step 1: Load Dataset
df = pd.read_csv("heart.csv")

# ✅ Step 2: Preprocess Data
X = df.drop("HeartDisease", axis=1)  # Features
y = df["HeartDisease"]  # Target variable

# Encode categorical columns
label_encoders = {}
for col in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Standardize numerical features
scaler = StandardScaler()
X[X.select_dtypes(include=["number"]).columns] = scaler.fit_transform(X[X.select_dtypes(include=["number"]).columns])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 3: Build Neural Network Model
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")  # Output layer (Binary Classification)
])

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Step 4: Train the Model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# ✅ Step 5: Save the Model
model.save("heart_disease_model.h5")

print("✅ Model training complete. Model saved as 'heart_disease_model.h5'")

