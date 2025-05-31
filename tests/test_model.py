import unittest
import mlflow
import os


class TestModelLoading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get DagsHub token from environment
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("❌ DAGSHUB_PAT environment variable not set.")

        # Set MLflow authentication environment variables
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Configure MLflow to use DagsHub as tracking URI
        dagshub_url = "https://dagshub.com"
        repo_owner = "iamprashantjain"
        repo_name = "laptop_price_predictor_mlops"

        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Set model name and fetch latest model version
        cls.model_name = "my_model"
        cls.model_version = cls.get_latest_model_version(cls.model_name, stage="Staging")

        if cls.model_version is None:
            raise ValueError(f"❌ No model found in 'Staging' stage for model '{cls.model_name}'")

        cls.model_uri = f"models:/{cls.model_name}/{cls.model_version}"

        try:
            cls.model = mlflow.pyfunc.load_model(cls.model_uri)
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model from URI {cls.model_uri}: {str(e)}")

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        """Get the latest model version in a given stage (default: 'Staging')"""
        client = mlflow.MlflowClient()
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        """Test that the MLflow model is not None after loading."""
        self.assertIsNotNone(self.model, "❌ Model failed to load, model is None.")


if __name__ == "__main__":
    unittest.main()
