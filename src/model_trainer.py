import os
import sys
import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from utils import logging, CustomException

try:
    # Load model parameters from params.yaml
    n_estimators = yaml.safe_load(open('params.yaml','r'))['model_trainer']['n_estimators']
    max_depth = yaml.safe_load(open('params.yaml','r'))['model_trainer']['max_depth']
    learning_rate = yaml.safe_load(open('params.yaml','r'))['model_trainer']['learning_rate']

    # Load processed data
    train_data = pd.read_csv(os.path.join("data", "processed", "train_processed.csv"))
    test_data = pd.read_csv(os.path.join("data", "processed", "test_processed.csv"))
    logging.info("✅ Loaded processed train and test data")

    # Split features and target
    X_train = train_data.drop(columns=["Price"])
    y_train = np.log(train_data["Price"])

    # Identify categorical columns
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Build preprocessing + model pipeline
    transformer = make_column_transformer(
        (OneHotEncoder(sparse=False, drop='first'), categorical_features),
        remainder='passthrough'
    )

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    pipe = Pipeline([
        ("transformer", transformer),
        ("model", model)
    ])

    # Train the model
    pipe.fit(X_train, y_train)
    logging.info("✅ Model training completed")

    # Save pipeline
    os.makedirs("models", exist_ok=True)
    pickle.dump(pipe, open("models/model.pkl", "wb"))
    logging.info("✅ Model saved to models/model.pkl")

except Exception as e:
    logging.exception("❌ Error occurred in model training")
    raise CustomException(e, sys)
