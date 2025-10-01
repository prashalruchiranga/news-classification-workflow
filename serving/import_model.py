from typing import Dict, Optional, Sequence
from google.cloud import aiplatform
from google.cloud.aiplatform import explain

def upload_model_to_vertexai(
    project: str,
    location: str,
    display_name: str,
    model_id: str,
    serving_container_image_uri: str,
    artifact_uri: Optional[str] = None,
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    description: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
    instance_schema_uri: Optional[str] = None,
    parameters_schema_uri: Optional[str] = None,
    prediction_schema_uri: Optional[str] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None, # type: ignore
    explanation_parameters: Optional[explain.ExplanationParameters] = None, # type: ignore
    sync: bool = True,
):
    aiplatform.init(project=project, location=location)
    model = aiplatform.Model.upload(
        display_name=display_name,
        model_id=model_id,
        artifact_uri=artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        instance_schema_uri=instance_schema_uri,
        parameters_schema_uri=parameters_schema_uri,
        prediction_schema_uri=prediction_schema_uri,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        sync=sync,
    )
    model.wait()
    return model
