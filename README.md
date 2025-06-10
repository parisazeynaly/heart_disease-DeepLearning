This project, Heart Disease Prediction using Deep Learning, leverages advanced deep learning techniques to predict heart disease from patient data. It integrates TensorFlow/Keras for model development, Docker for seamless deployment, and MLflow for comprehensive experiment tracking.

Getting Started
To begin, clone the repository and navigate into the project directory:

git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

Installation
Install Python Dependencies:

pip install -r requirements.txt

Start MLflow Tracking Server (Docker):
This command launches an MLflow server in a Docker container, making it accessible on port 5000 and persisting experiment data in the mlruns directory.

docker run -d -p 5000:5000 --name mlflow-server \
  -v "$(pwd)/mlruns:/mlruns" \
  ghcr.co/mlflow/mlflow:v2.10.2 server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root /mlruns


Usage & FeaturesThis project streamlines the deep learning workflow with the following key features:

Model Training: Develop and train deep learning models for heart disease prediction using TensorFlow/Keras.

MLflow Experiment Tracking: Automatically log parameters, metrics, and artifacts for each training run, enabling easy comparison and reproducibility.

Containerized Deployment (Docker): Package your model and its dependencies into a Docker image for consistent and portable execution across environments.

Model Inference: Easily perform predictions using the trained deep learning model..
