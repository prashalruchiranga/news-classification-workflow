from kfp.dsl import component

@component(
    base_image="docker.io/prashalruchiranga/news-classifier:components-v1.2"
)
def get_run_id(
    mlflow_tracking_uri: str,
    experiment_name: str
    ) -> str:
    import os
    import mlflow

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    run_id = run.info.run_id
    mlflow.end_run()
    return run_id
