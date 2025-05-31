import os
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

# Initialize Flask app
app = Flask(__name__)

# ------------------ MLflow Setup ------------------
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable not set.")

# Set MLflow authentication environment variables
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Configure MLflow to use DagsHub as tracking URI
dagshub_url = "https://dagshub.com"
repo_owner = "iamprashantjain"
repo_name = "laptop_price_predictor_mlops"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


# Load model from MLflow Model Registry
model_name = "laptop_price_predictor"
model_stage = "Production"  # Use "Staging" or specific version if needed

model_uri = f"models:/{model_name}/{model_stage}"
model: Pipeline = mlflow.sklearn.load_model(model_uri)
print(f"✅ Loaded model from MLflow Model Registry: {model_uri}")

# ------------------ Prediction Route ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_data = {
                'Company': request.form['Company'],
                'TypeName': request.form['TypeName'],
                'Ram': int(request.form['Ram']),
                'Weight': float(request.form['Weight']),
                'Touchscreen': int(request.form.get('Touchscreen', 0)),
                'Ips': int(request.form.get('Ips', 0)),
                'ppi': float(request.form['ppi']),
                'Cpu brand': request.form['CpuBrand'],
                'HDD': int(request.form['HDD']),
                'SSD': int(request.form['SSD']),
                'Gpu brand': request.form['GpuBrand'],
                'os': request.form['os']
            }

            df_input = pd.DataFrame([input_data])
            prediction = model.predict(df_input)[0]
            return render_template("index.html", prediction=f"₹ {prediction:,.2f}")

        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}")

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
