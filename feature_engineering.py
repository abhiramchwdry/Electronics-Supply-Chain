from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

def feature_engineering(X_train, X_test, y_train):
    # Example scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Feature Selection
    selector = SelectKBest(score_func=f_regression, k='all')
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected
