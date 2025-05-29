import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from utils import logging, CustomException

try:
    # Load model parameters from params.yaml
    params = yaml.safe_load(open('params.yaml', 'r'))['model_trainer']
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    min_samples_split = params['min_samples_split']
    min_samples_leaf = params['min_samples_leaf']

    # Load processed data
    train_data = pd.read_csv(os.path.join("data", "processed", "train_processed.csv"))
    test_data = pd.read_csv(os.path.join("data", "processed", "test_processed.csv"))
    logging.info("✅ Loaded processed train and test data")

    # Split features and target
    X_train = train_data.drop(columns=["Price"])
    y_train = train_data["Price"]

    # Identify numeric and categorical features
    numeric_features = ['Ram', 'Weight', 'ppi', 'HDD', 'SSD']
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing + model pipeline
    transformer = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
        remainder='passthrough'
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    pipe = Pipeline([
        ("transformer", transformer),
        ("model", model)
    ])

    # Train the model
    pipe.fit(X_train, y_train)
    logging.info("✅ Model training completed using RandomForestRegressor")

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    logging.info("✅ Model saved to models/model.pkl")

except Exception as e:
    logging.exception("❌ Error occurred in model training")
    raise CustomException(e, sys)
