from kfp.dsl import component

@component(
    base_image="docker.io/prashalruchiranga/news-classification:kfp-base-arm64"
)
def get_run_id(
    experiment_name: str
    ) -> str:
    import os
    import mlflow

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    run = mlflow.start_run()
    run_id = run.info.run_id
    mlflow.end_run()
    return run_id
