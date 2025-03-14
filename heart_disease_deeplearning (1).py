# -*- coding: utf-8 -*-
"""heart_disease-DeepLearning.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-J3Hdc0rYE13SacbEvt1XlfAzkY7DPI2
"""

# !git clone https://github.com/parisazeynaly/heart_disease-DeepLearning.git

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load Dataset
df = pd.read_csv("heart.csv")

# Display Basic Info
print(df.info())  # Dataset summary
print(df.describe().T.round(2))  # Statistical summary
print(df.head())  # First few rows

"""**EDA**

Visualizing RestingBP vs Cholesterol
"""

plt.figure(figsize=(6, 4))
df.plot(kind='scatter', x='RestingBP', y='Cholesterol', s=32, alpha=.8)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.title("RestingBP vs Cholesterol")
plt.show()

plt.figure(figsize=(6,4))
sns.countplot(x=df["HeartDisease"])
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

df["HeartDisease"].value_counts().plot(kind='pie', colormap="tab10", explode=[0.02, 0.02]) # Correct the explode list length
plt.show()

"""# Preprocessing"""

# Encode Categorical Variables (One-Hot Encoding)
df = pd.get_dummies(df, columns=["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"], drop_first=True)

# Normalize Numerical Features
scaler = StandardScaler()
num_features = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
df[num_features] = scaler.fit_transform(df[num_features])


# ***********

# Correlation Heatmap

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlations with Heart Disease")
plt.show()

#  Define Features & Target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

#  Split Dataset (Stratified to Preserve Class Balance)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45, stratify=y)

# Convert to NumPy arrays
x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)

#  Compute Class Weights (Fix Class Imbalance)
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

print(f"X_train shape: {x_train.shape}")
print(f"X_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Ensure all data is numeric
x_train = pd.DataFrame(x_train).apply(pd.to_numeric, errors='coerce')
x_test = pd.DataFrame(x_test).apply(pd.to_numeric, errors='coerce')

"""Model defenition"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization


model = Sequential([
    Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test), class_weight=class_weights_dict)

# Plot Training Performance
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# Evaluate Model on Test Data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print(f"\n Test Accuracy: {test_acc:.4f}")

from sklearn.metrics import classification_report
#  Generate Classification Report
y_pred = (model.predict(x_test) > 0.5).astype("int32")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

!pip install mlflow

import mlflow
import mlflow.sklearn

mlflow.set_experiment("Your_Project_Experiment")

model.save("model/heart_disease_model.h5")

# Set up the optimizer
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=5e-5)

# Compile the model with the new optimizer if needed
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

with mlflow.start_run() as run:
    # Training loop using Keras fit method
    history = model.fit(x_train, y_train, epochs=20, batch_size=32,
                        validation_data=(x_test, y_test), class_weight=class_weights_dict)

    # Log metrics after each epoch
    for epoch in range(20):
        mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch + 1)
        mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch + 1)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch + 1)
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch + 1)

    # Log model using mlflow.keras
    mlflow.keras.log_model(model, "model")

    # Print run_id
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")

#Serve the model locally using MLflow
!mlflow models serve -m runs:/{value from run_id}/model -p 5000 --no-conda


#zip the folder and download it:

!zip -r mlruns.zip /content/mlruns


# Download the zipped file
from google.colab import files
files.download('mlruns.zip')