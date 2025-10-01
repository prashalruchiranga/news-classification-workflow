from types import SimpleNamespace
import mlflow
from utils.misc import load_config

def get_gcs_artifacts_path(reg_model_name: str, alias: str) -> str:
    # Download the MLmodel metadata file for the alias
    model_uri = f"models:/{reg_model_name}@{alias}"
    model_metadata_uri = f"{model_uri}/MLmodel"
    dest = mlflow.artifacts.download_artifacts(artifact_uri=model_metadata_uri, dst_path="./artifacts")
    # Extract the artifacts path in the GCS
    metadata = SimpleNamespace(**load_config(dest))
    return metadata.artfact_path

def deploy():
    config_dict = load_config("./configs/serving.yaml")
    configs = SimpleNamespace(**config_dict)
    mlflow.set_tracking_uri(uri=configs.mlflow_endpoint)
    model_name = configs.registered_model
    gcs_artifacts_path = get_gcs_artifacts_path(model_name, alias="challenger")

if __name__ == "__main__":
    deploy()
