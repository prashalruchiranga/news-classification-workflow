from types import SimpleNamespace
import yaml
import mlflow
from google.cloud import aiplatform
from utils.misc import load_config
from google.api_core.exceptions import AlreadyExists
from operations.import_model import upload_model_to_vertexai
from operations.deploy_model import deploy_model_with_dedicated_resources

def fetch_mlmodel_metadata(registered_model_name: str, alias: str):
    model_uri = f"models:/{registered_model_name}@{alias}"
    model_metadata_uri = f"{model_uri}/MLmodel"
    metadata_text = mlflow.artifacts.load_text(model_metadata_uri)
    metadata_dict = yaml.safe_load(metadata_text)
    return SimpleNamespace(**metadata_dict)

def main(configs: SimpleNamespace):
    mlflow.set_tracking_uri(uri=configs.mlflow_endpoint)
    aiplatform.init(project=configs.project_id, location=configs.location)

    metadata = fetch_mlmodel_metadata(
        registered_model_name=configs.registered_model,
        alias=configs.alias_to_deploy
    )
    model_id = metadata.model_id
    gcs_artifact_path = metadata.artifact_path
    display_name = f"{configs.registered_model} {model_id}"
    model = upload_model_to_vertexai(
        project=configs.project_id,
        location=configs.location,
        display_name=display_name,
        model_id=model_id,
        serving_container_image_uri=configs.container_image_uri,
        artifact_uri=gcs_artifact_path,
        serving_container_predict_route="/predict",
        serving_container_health_route="/"
    )
    
    endpoint_name = configs.endpoint_name
    endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
    models_to_undeploy = None
    if endpoints:
        endpoint = endpoints[0]
        models_to_undeploy = endpoint.list_models()
    else:
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_name
        )
    returned_endpoint = deploy_model_with_dedicated_resources(
        project=configs.project_id,
        location=configs.location,
        model=model,
        machine_type=configs.machine_type,
        endpoint=endpoint,
        traffic_percentage=100
    )

    if models_to_undeploy:
        prev_dep_model = models_to_undeploy[0]
        deployment_id = prev_dep_model.id
        endpoint.undeploy(deployed_model_id=deployment_id)
        # resource_name = prev_dep_model.model
        # aiplatform.Model(model_name=resource_name).delete()

    return None


if __name__ == "__main__":
    try:
        config_dict = load_config("./configs/serving.yaml")
        configs = SimpleNamespace(**config_dict)
        endpoint_resource = main(configs)
        print(f"Model deployment finished.")
    except AlreadyExists as e:
        print(f"Model already exists in the registry: {e}")
        exit(1)
    except Exception as e:
        print(f"Model deployment failed with error: {e}")
        exit(1)
