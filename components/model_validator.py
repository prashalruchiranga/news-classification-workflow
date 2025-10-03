from kfp.dsl import component

@component(
    base_image="docker.io/prashalruchiranga/news-classifier:components-v1.2"
)
def validate_model(
    mlflow_run_id: str,
    experiment_name: str,
    registered_model_name: str,
    baseline: dict
):
    import os
    import mlflow

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id=mlflow_run_id)
    metrics = run.data.metrics
    accuracy = metrics["sparse_categorical_accuracy"]
    f1_score = metrics["f1_score"]

    # Register the model if accuracy and f1 score exceed the baseline
    baseline_accuracy = baseline["accuracy"]
    baseline_f1_score = baseline["f1_score"]
    if accuracy >= baseline_accuracy and f1_score >= baseline_f1_score:
        mlflow_model_uri = f"runs:/{mlflow_run_id}/keras-model"
        try:
            client.get_registered_model(registered_model_name)
            print(f"Model '{registered_model_name}' already exists.")
        except mlflow.exceptions.RestException as ex:
            if ex.error_code == "RESOURCE_DOES_NOT_EXIST":
                print(f"Model '{registered_model_name}' does not exist. Creating new model.")
                client.create_registered_model(registered_model_name)
            else:
                print(f"An error occurred while registering the model: {ex}")
                raise
        mv = client.create_model_version(registered_model_name, source=mlflow_model_uri, run_id=mlflow_run_id)
        client.set_registered_model_tag(registered_model_name, key="experiment", value=experiment_name)
        client.set_registered_model_alias(registered_model_name, alias="challenger", version=mv.version)
    else:
        message = (
            "Model from run {} did not meet the baseline ({}). "
            "It will not be registered."
        )
        print(message.format(mlflow_run_id, baseline))
