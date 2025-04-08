# from fastapi import FastAPI
# import uvicorn
#
# app = FastAPI()
#
# @app.get("/")
# def home():
#     """Check if API is running."""
#     return {"message": "FastAPI is running!"}
#
# # Run FastAPI when executing app.py (Without `reload=True`)
# if __name__ == "__main__":
#     uvicorn.run("app:app", host="127.0.0.1", port=8000)
# from flask import Flask, request, render_template
# import numpy as np
# import joblib  # or keras.models.load_model if it's a Keras model
# import shap
# from tensorflow.keras.models import load_model
#
# app = Flask(__name__)
# model = load_model("heart_disease_model.h5")
#
# @app.route('/')
# def home():
#     return render_template('index.html')
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     user_input = [float(x) for x in request.form.values()]
#     data = np.array(user_input).reshape(1, -1)
#     prediction = model.predict(data)[0]
#
#     # Optional XAI explanation (simple example)
#     explainer = shap.Explainer(model)
#     shap_values = explainer(data)
#     top_features = np.argsort(-np.abs(shap_values.values[0]))[:3]
#
#     return render_template('result.html', prediction=prediction, explanation=top_features)
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
import shap

app = Flask(__name__)

# Load the trained Keras model
model = load_model("heart_disease_model.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # گرفتن ورودی‌های فرم (باید 11 عدد باشن)
        user_input = [float(x) for x in request.form.values()]

        if len(user_input) != 11:
            return f"❌ باید دقیقاً 11 ویژگی وارد کنی. الان {len(user_input)} تا دادی!"

        data = np.array(user_input).reshape(1, -1)

        # پیش‌بینی
        prediction = model.predict(data)[0][0]  # چون مدل Keras خروجی 2D می‌ده

        # SHAP برای XAI
        explainer = shap.Explainer(model, data)
        shap_values = explainer(data)
        top_features = np.argsort(-np.abs(shap_values.values[0]))[:3]

        return render_template('result.html', prediction=round(prediction, 2), explanation=top_features)

    except Exception as e:
        return f"❌ خطا: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
