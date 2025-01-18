import traceback
from data_preprocessing import preprocess_data, feature_engineering

def evaluate_models():
    try:
        # Load data
        X, y = preprocess_data('data/electronics_supply_chain_data_500.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature engineering
        X_train_selected, X_test_selected = feature_engineering(X_train, X_test, y_train)

        # Load models
        rf_model = joblib.load('models/random_forest_model.pkl')
        xgb_model = joblib.load('models/xgboost_model.pkl')
        lgb_model = joblib.load('models/lightgbm_model.pkl')

        # Evaluate models
        rf_preds = rf_model.predict(X_test_selected)
        xgb_preds = xgb_model.predict(X_test_selected)
        lgb_preds = lgb_model.predict(X_test_selected)

        rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
        xgb_rmse = mean_squared_error(y_test, xgb_preds, squared=False)
        lgb_rmse = mean_squared_error(y_test, lgb_preds, squared=False)

        rf_r2 = r2_score(y_test, rf_preds)
        xgb_r2 = r2_score(y_test, xgb_preds)
        lgb_r2 = r2_score(y_test, lgb_preds)

        print(f"Random Forest RMSE: {rf_rmse}")
        print(f"XGBoost RMSE: {xgb_rmse}")
        print(f"LightGBM RMSE: {lgb_rmse}")

        print(f"Random Forest R²: {rf_r2}")
        print(f"XGBoost R²: {xgb_r2}")
        print(f"LightGBM R²: {lgb_r2}")

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
