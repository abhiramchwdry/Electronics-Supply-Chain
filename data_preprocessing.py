import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handle missing values with SimpleImputer
    imputer = SimpleImputer(strategy='median')
    df.iloc[:, :] = imputer.fit_transform(df)

    # Convert categorical variables to numeric
    df = pd.get_dummies(df, drop_first=True)
    
    target_column = 'target_column'  # Replace with your actual target column name
    if target_column not in df.columns:
        raise ValueError(f"The dataset must contain '{target_column}'")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    return X, y
