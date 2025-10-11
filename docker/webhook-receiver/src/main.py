import os
from typing import Optional
import logging
from fastapi import FastAPI, Request, HTTPException, Header
import mlflow
from google.cloud import aiplatform
from contextlib import asynccontextmanager
from verification import verify_timestamp_freshness, verify_mlflow_signature
from helpers import fetch_mlmodel_metadata

app = FastAPI()
logger = logging.getLogger(__name__)

# Maximum allowed age for webhook timestamps (in seconds)
MAX_TIMESTAMP_AGE = 300

# Read environment variables
MLFLOW_WEBHOOK_SECRET = os.getenv("MLFLOW_WEBHOOK_SECRET")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
CONTAINER_IMAGE_URI = os.getenv("CONTAINER_IMAGE_URI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
    aiplatform.init(project=PROJECT_ID, location=LOCATION)
    yield


@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_mlflow_signature: Optional[str] = Header(None),
    x_mlflow_delivery_id: Optional[str] = Header(None),
    x_mlflow_timestamp: Optional[str] = Header(None)
):
    """Handle webhook with HMAC signature verification"""

    # Get raw payload for signature verification
    payload_bytes = await request.body()
    payload = payload_bytes.decode("utf-8")

    # Verify required headers are present
    if not x_mlflow_signature:
        raise HTTPException(status_code=400, detail="Missing signature header")
    if not x_mlflow_delivery_id:
        raise HTTPException(status_code=400, detail="Missing delivery ID header")
    if not x_mlflow_timestamp:
        raise HTTPException(status_code=400, detail="Missing timestamp header")
    
    # Verify timestamp freshness to prevent replay attacks
    if not verify_timestamp_freshness(x_mlflow_timestamp, MAX_TIMESTAMP_AGE):
        raise HTTPException(
            status_code=400,
            detail="Timestamp is too old or invalid (possible replay attack)"
        )
    
    # Verify signature
    if not verify_mlflow_signature(
        payload,
        x_mlflow_signature,
        MLFLOW_WEBHOOK_SECRET,
        x_mlflow_delivery_id,
        x_mlflow_timestamp
    ):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Parse payload
    webhook_data = await request.json()

    # Extract webhook metadata
    entity = webhook_data.get("entity")
    action = webhook_data.get("action")
    timestamp = webhook_data.get("timestamp")
    payload_data = webhook_data.get("data", {})

    # Print the payload for debugging
    print(f"Received webhook: {entity}.{action}")
    print(f"Timestamp: {timestamp}")
    print(f"Delivery ID: {x_mlflow_delivery_id}")
    print(f"Payload: {payload_data}")

    if entity == "model_version" and action == "created":
        model_name = payload_data.get("name")
        version = payload_data.get("version")

        # Fetch MLflow model metadata using MLmodel.txt
        metadata = fetch_mlmodel_metadata(
            registered_model_name=model_name,
            version=version
        )
        gcs_artifact_path = metadata.artifact_path

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
            serving_container_image_uri=CONTAINER_IMAGE_URI,
            artifact_uri=gcs_artifact_path,
            parent_model=parent_model,
            display_name=model_name,
            serving_container_predict_route="/predict",
            serving_container_health_route="/"
        )

    elif entity == "model_version_alias" and action == "created":
        model_name = payload_data.get("name")
        alias = payload_data.get("alias")
        version = payload_data.get("version")
        metadata = fetch_mlmodel_metadata(
            registered_model_name=model_name,
            alias=alias
        )
        model_id = metadata.model_id
        gcs_artifact_path = metadata.artifact_path

    return {"status": "success"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
