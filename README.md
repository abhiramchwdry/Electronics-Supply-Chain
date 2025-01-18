# Electronics Supply Chain Project

## Overview

This project aims to predict the availability dates of electronic products using machine learning algorithms. It involves preprocessing data, feature engineering, training various models, and evaluating their performance.

## Project Structure

- data/: Contains the dataset file.
- scripts/: Contains Python scripts for data preprocessing, feature engineering, model training, and evaluation.
- 
otebooks/: (Optional) Jupyter notebooks for exploratory data analysis and model experimentation.
- equirements.txt: List of dependencies.
- README.md: Project documentation.

## Setup

1. **Create and activate a virtual environment:**

`ash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
` 

2. **Install dependencies:**

`ash
pip install -r requirements.txt
` 

3. **Run the scripts:**

- **Preprocess Data:**

`ash
python scripts/data_preprocessing.py
` 
- **Feature Engineering:**
`ash
python scripts/feature_engineering.py
` 
- **Train Models:**
`ash
python scripts/model_training.py
` 
- **Evaluate Models:**
`ash
python scripts/model_evaluation.py
` 

## Results

The models' performance is evaluated based on the RMSE metric. The best model will be selected based on the lowest RMSE.

## License

This project is licensed under the MIT License.
