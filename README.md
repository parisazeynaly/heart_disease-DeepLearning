# heart_disease-DeepLearning
Project Name: Heart Disease Prediction using Deep Learning
Description:
This project aims to develop a deep learning model for predicting heart disease based on patient data. It leverages TensorFlow/Keras for training the model, Docker for containerization, and MLflow for experiment tracking.
Clone the Repository:git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

Install Dependencies:pip install -r requirements.txt
docker run -d -p 5000:5000 --name mlflow-server \
    -v "$(pwd)/mlruns:/mlruns" \
    ghcr.io/mlflow/mlflow:v2.10.2 server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root /mlruns
Model Training:
MLflow Experiment Tracking:
Running the Model in Docker
Usage: Model Inference
