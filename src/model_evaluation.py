import yaml
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from utils import logging, CustomException

import mlflow
import mlflow.sklearn
import dagshub

# Initialize Dagshub tracking
# dagshub.init(repo_owner='iamprashantjain', repo_name='laptop_price_predictor_mlops', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/laptop_price_predictor_mlops.mlflow")


# Fetch DAGSHUB_PAT from environment
dagshub_token = os.getenv("DAGSHUB_PAT")

# Error if not set
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable not set")

# Set MLflow credentials for DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set MLflow tracking URI for DagsHub
dagshub_url = "https://dagshub.com"
repo_owner = "iamprashantjain"
repo_name = "laptop_price_predictor_mlops"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


mlflow.set_experiment("dvc-pipeline")

try:
    # Load test data
    df_test = pd.read_csv("data/processed/test_processed.csv")
    logging.info("✅ Loaded processed test data")

    X_test = df_test.drop(columns=["Price"])
    y_test = df_test["Price"]

    # Load the trained pipeline
    pipe = pickle.load(open("models/model.pkl", "rb"))
    logging.info("✅ Loaded trained model")

    # Start MLflow run
    with mlflow.start_run():
        # Make predictions
        y_pred = pipe.predict(X_test)

        # Evaluate
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"✅ R2 Score: {r2:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        logging.info(f"✅ R2 Score: {r2:.4f}, MAE: {mae:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("mae", mae)

        # Log model to MLflow (key for later registration)
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        # Save metrics to local YAML file
        os.makedirs("metrics", exist_ok=True)
        with open("metrics/evaluation.yaml", "w") as f:
            yaml.dump({"r2_score": float(r2), "mae": float(mae)}, f)

        logging.info("✅ Evaluation metrics saved and model logged to MLflow")

except Exception as e:
    logging.exception("❌ Exception occurred during model evaluation")
    raise CustomException(e, sys)
