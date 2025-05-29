import yaml
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from utils import logging, CustomException

try:
    # Load test data
    df_test = pd.read_csv("data/processed/test_processed.csv")
    logging.info("✅ Loaded processed test data")

    X_test = df_test.drop(columns=["Price"])
    y_test = df_test["Price"]  # No log transformation

    # Load the trained pipeline
    pipe = pickle.load(open("models/model.pkl", "rb"))
    logging.info("✅ Loaded trained model")

    # Make predictions
    y_pred = pipe.predict(X_test)

    # Evaluate
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ R2 Score: {r2:.4f}")
    print(f"✅ MAE: {mae:.4f}")
    logging.info(f"✅ R2 Score: {r2:.4f}, MAE: {mae:.4f}")
    
    # Save metrics to file
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/evaluation.yaml", "w") as f:
        yaml.dump({"r2_score": float(r2), "mae": float(mae)}, f)
    
    logging.info("✅ Evaluation metrics saved to metrics/evaluation.yaml")

except Exception as e:
    logging.exception("❌ Exception occurred during model evaluation")
    raise CustomException(e, sys)
