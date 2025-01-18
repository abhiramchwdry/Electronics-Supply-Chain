import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scripts.data_preprocessing import preprocess_data
from scripts.feature_engineering import feature_engineering

def train_models():
    try:
        # Load and preprocess data
        X, y = preprocess_data('data/electronics_supply_chain_data_500.csv')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Feature engineering
        X_train_selected, X_test_selected = feature_engineering(X_train, X_test, y_train)

        # Initialize and train models
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)

        rf_model.fit(X_train_selected, y_train)
        xgb_model.fit(X_train_selected, y_train)
        lgb_model.fit(X_train_selected, y_train)

        # Save models
        joblib.dump(rf_model, 'models/random_forest_model.pkl')
        joblib.dump(xgb_model, 'models/xgboost_model.pkl')
        joblib.dump(lgb_model, 'models/lightgbm_model.pkl')

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

if __name__ == "__main__":
    train_models()
