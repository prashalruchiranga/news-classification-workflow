import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tensorflow import load_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model
    global reloaded_model
    storage_uri = os.getenv("AIP_STORAGE_URI")
    destination = "./artifacts"
    os.makedirs(destination, exist_ok=True)
    reloaded_model = load_model(model_uri=storage_uri, dst_path=destination)
    yield
    # Clean up the model and release the resources
    reloaded_model = None

class PredictRequest(BaseModel):
    instances: list[str]

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"status": "OK"}

@app.post("/predict")
async def predict(request: PredictRequest):
    if reloaded_model is None:
        return {"error": "Model is not loaded"}
    predictions = reloaded_model.predict(request.instances)
    return {"predictions": predictions.tolist()}
