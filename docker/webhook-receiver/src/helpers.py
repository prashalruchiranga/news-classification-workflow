import yaml
from typing import Optional
from types import SimpleNamespace
import mlflow
from google.cloud import aiplatform, aiplatform_v1


def fetch_mlmodel_metadata(
    registered_model_name: str,
    version: Optional[str] = None,
    alias: Optional[str] = None
):
    if version and alias:
        raise ValueError("Both version and alias cannot be provided simultaneously")
    if version:
        model_uri = f"models:/{registered_model_name}/{version}"
    elif alias:
        model_uri = f"models:/{registered_model_name}@{alias}"
    else:
        raise ValueError("Either version or alias must be provided")
    
    model_metadata_uri = f"{model_uri}/MLmodel"
    metadata_text = mlflow.artifacts.load_text(model_metadata_uri)
    metadata_dict = yaml.safe_load(metadata_text)
    return SimpleNamespace(**metadata_dict)


def handle_model_version_created(
    payload_data: dict,
    container_image_uri: str
):
    try:
        model_name = payload_data.get("name")
        version = payload_data.get("version")

        # Fetch MLflow model metadata using MLmodel.txt
        metadata = fetch_mlmodel_metadata(
            registered_model_name=model_name,
            version=version
        )
        gcs_artifact_path = metadata.artifact_path
        mlflow_model_id = metadata.model_id

        # Create a parent model if registered model does not exist in Vertex AI Model Registry
        models = aiplatform.Model.list(
            filter=f'display_name="{model_name}"',
            order_by="create_time"
        )
        if models:
            parent_model = models[0].resource_name
        else:
            parent_model = None

        # Import model from the storage bucket to Vertex AI Model Registry
        aiplatform.Model.upload(
            serving_container_image_uri=container_image_uri,
            artifact_uri=gcs_artifact_path,
            parent_model=parent_model,
            display_name=model_name,
            serving_container_predict_route="/predict",
            serving_container_health_route="/",
            labels = {"mlflow_model_id": mlflow_model_id, "mlflow_model_version": version}
        )
    except Exception as e:
        print(f"Model import failed: {e}")


def handle_model_version_alias_created(
    payload_data: dict,
    location: str
):
    try:
        model_name = payload_data.get("name")
        alias = payload_data.get("alias")
        version = payload_data.get("version")

        if alias == "champion":
            # Fetch MLflow model metadata using MLmodel.txt
            metadata = fetch_mlmodel_metadata(
                registered_model_name=model_name,
                alias=alias
            )
            mlflow_model_id = metadata.model_id

            # Get the champion model from imported models
            models = aiplatform.Model.list(
                filter=f'display_name="{model_name}"',
                order_by="create_time"
            )
            parent = models[0].resource_name

            client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
            client = aiplatform_v1.ModelServiceClient(
                client_options=client_options
            )
            mv_request = aiplatform_v1.ListModelVersionsRequest(name=parent)
            mv_response = client.list_model_versions(request=mv_request)
            for model in mv_response.models:
                label = model.labels
                if label["mlflow_model_id"] == mlflow_model_id and label["mlflow_model_version"] == version:
                    champion = model
                    break

            # Assign champion alias
            name = f"{parent}@{champion.version_id}"
            mva_request = aiplatform_v1.MergeVersionAliasesRequest(
                name=name,
                version_aliases=["champion"]
            )
            mva_response = client.merge_version_aliases(request=mva_request)
    except Exception as e:
        print(f"Model alias update failed: {e}")
