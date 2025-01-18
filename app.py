from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from scripts.data_preprocessing import preprocess_data
from scripts.feature_engineering import feature_engineering

app = Flask(__name__)

# Load models
rf_model = joblib.load('models/random_forest_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
lgb_model = joblib.load('models/lightgbm_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.json

        # Convert data to DataFrame
        df = pd.DataFrame([data])
        print("DataFrame created from JSON:", df)

        # Preprocess and feature engineer
        X, _ = preprocess_data(df)
        print("Data after preprocessing:", X)

        X = feature_engineering(X, X, None)[0]  # Feature engineering for prediction
        print("Data after feature engineering:", X)

        # Make predictions
        rf_pred = rf_model.predict(X)[0]
        xgb_pred = xgb_model.predict(X)[0]
        lgb_pred = lgb_model.predict(X)[0]

        return jsonify({
            'Random Forest Prediction': rf_pred,
            'XGBoost Prediction': xgb_pred,
            'LightGBM Prediction': lgb_pred
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
