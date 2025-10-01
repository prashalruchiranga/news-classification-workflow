from types import SimpleNamespace
import mlflow
from google.cloud import aiplatform
from utils.misc import load_config
from serving.import_model import upload_model_to_vertexai
from google.api_core.exceptions import AlreadyExists

def get_gcs_artifacts_path(reg_model_name: str, alias: str) -> str:
    # Fetch artifacts path from MLflow metadata
    model_uri = f"models:/{reg_model_name}@{alias}"
    model_metadata_uri = f"{model_uri}/MLmodel"
    dest = mlflow.artifacts.download_artifacts(
        artifact_uri=model_metadata_uri, dst_path="./artifacts")
    metadata = SimpleNamespace(**load_config(dest))
    return metadata.artifact_path, metadata.model_id

def deploy(configs: SimpleNamespace):
    mlflow.set_tracking_uri(uri=configs.mlflow_endpoint)
    aiplatform.init(project=configs.project_id, location=configs.location)
    model_name = configs.registered_model
    gcs_artifacts_path, model_id = get_gcs_artifacts_path(model_name, alias="challenger")
    try:
        model = upload_model_to_vertexai(
            project=configs.project_id,
            location=configs.location,
            display_name=configs.registered_model,
            model_id=model_id,
            serving_container_image_uri=configs.container_image_uri,
            artifact_uri=gcs_artifacts_path,
            serving_container_predict_route="/predict",
            serving_container_health_route="/"
        )
        print(model)
    except AlreadyExists:
        print(f"Model ID {model_id} already exists in the registry. Terminating the process.")
    except Exception as e:
        print(f"Error occured: {e}")

if __name__ == "__main__":
    config_dict = load_config("./configs/serving.yaml")
    configs = SimpleNamespace(**config_dict)
    deploy(configs)
