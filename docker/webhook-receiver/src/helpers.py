import yaml
from typing import Optional
from types import SimpleNamespace
import mlflow

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
