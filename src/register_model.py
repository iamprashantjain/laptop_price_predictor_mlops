from mlflow.tracking import MlflowClient
import mlflow
import dagshub
import time

dagshub.init(repo_owner='iamprashantjain', repo_name='laptop_price_predictor_mlops', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/iamprashantjain/laptop_price_predictor_mlops.mlflow")
client = MlflowClient()

try:
    # Get experiment details
    experiment = client.get_experiment_by_name("dvc-pipeline")
    if experiment is None:
        raise ValueError("Experiment 'dvc-pipeline' not found.")

    experiment_id = experiment.experiment_id

    # Get latest successful run
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError("No runs found in the experiment.")

    run_id = runs[0].info.run_id
    print(f"Using Run ID: {run_id}")

    # Path to model inside run's artifacts (update if different)
    model_path = "model"  # Check Dagshub UI if it's different

    # Build model URI
    model_uri = f"runs:/{run_id}/{model_path}"
    model_name = "laptop_price_predictor"

    # Register model
    result = mlflow.register_model(model_uri, model_name)
    print(f"Model registered: version {result.version}")

    # Wait to ensure registration completes
    time.sleep(5)

    # Add description
    client.update_model_version(
        name=model_name,
        version=result.version,
        description="This is a random forest model trained on laptop price"
    )

    # Add tags
    client.set_model_version_tag(
        name=model_name,
        version=result.version,
        key="author",
        value="prashantj"
    )

    print("Model registration and tagging completed successfully.")

    # Promote model to "Staging"
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="None",
        archive_existing_versions=True
    )

    print(f"Model {model_name} version {result.version} moved to Staging.")


except Exception as e:
    print(f"Error during model registration: {e}")
