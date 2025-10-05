from typing import Dict, Optional, Sequence, Tuple
from google.cloud import aiplatform
from google.cloud.aiplatform import explain, Model, Endpoint

def deploy_model_with_dedicated_resources(
    project: str,
    location: str,
    model: Model,
    machine_type: str,
    endpoint: Optional[Endpoint] = None,
    deployed_model_display_name: Optional[str] = None,
    traffic_percentage: Optional[int] = 0,
    traffic_split: Optional[Dict[str, int]] = None,
    min_replica_count: int = 1,
    max_replica_count: int = 1,
    accelerator_type: Optional[str] = None,
    accelerator_count: Optional[int] = None,
    explanation_metadata: Optional[explain.ExplanationMetadata] = None, #type: ignore
    explanation_parameters: Optional[explain.ExplanationParameters] = None, #type: ignore
    metadata: Optional[Sequence[Tuple[str, str]]] = (),
    sync: bool = True,
):
    aiplatform.init(project=project, location=location)
    endpoint = model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=deployed_model_display_name,
        traffic_percentage=traffic_percentage,
        traffic_split=traffic_split,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        explanation_metadata=explanation_metadata,
        explanation_parameters=explanation_parameters,
        metadata=metadata,
        sync=sync,
    )
    model.wait()
    return endpoint
