from kfp import kubernetes
from kfp.dsl import pipeline
from components import mlflow_run, data_loader, model_trainer, model_evaluator, model_validator
from typing import Optional

@pipeline(
    name="news-clf-model-trainer",
    description= "Pipeline to fine-tune a pre-trained BERT model for news classification"
)
def training_pipeline(
    experiment_name: str,
    registered_model_name: str,
    baseline: float,
    dataset_name: str,
    text_column_name: str,
    label_column_name: str,
    val_data_fraction: int = 0.2,
    dataset_split_seed: int = 42,
    bert_preset: str = "bert_tiny_en_uncased",
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    epochs: int = 20,
    early_stopping_min_delta: float = 0.001,
    early_stopping_patience: int = 5,
    debug_batch_count: Optional[int] = None
    ):
    get_run_id_op = mlflow_run.get_run_id(
        experiment_name=experiment_name
    )
    kubernetes.use_secret_as_volume(
        task=get_run_id_op,
        secret_name="gcs-credentials",
        mount_path="/etc/gcs"
    )
    kubernetes.use_config_map_as_env(
        task=get_run_id_op,
        config_map_name="mlflow-config",
        config_map_key_to_env={"mlflow_tracking_uri": "MLFLOW_TRACKING_URI"}
    )
    get_run_id_op.set_caching_options(False)
    run_id = get_run_id_op.output

    load_dataset_op = data_loader.load_hf_dataset(
        mlflow_run_id=run_id,
        name=dataset_name,
        val_fraction=val_data_fraction,
        seed=dataset_split_seed
    )
    kubernetes.use_secret_as_volume(
        task=load_dataset_op,
        secret_name="gcs-credentials",
        mount_path="/etc/gcs"
    )
    kubernetes.use_config_map_as_env(
        task=load_dataset_op,
        config_map_name="mlflow-config",
        config_map_key_to_env={"mlflow_tracking_uri": "MLFLOW_TRACKING_URI"}
    )

    finetune_bert_op = model_trainer.finetune_bert(
        mlflow_run_id=run_id,
        preset=bert_preset,
        batch_size=batch_size,
        text_col=text_column_name,
        label_col=label_column_name,
        lr=learning_rate,
        epochs=epochs,
        min_delta=early_stopping_min_delta,
        patience=early_stopping_patience,
        debug_batch_count=debug_batch_count,
        train_dataset=load_dataset_op.outputs["train_dataset"],
        val_dataset=load_dataset_op.outputs["val_dataset"]
    )
    kubernetes.use_secret_as_volume(
        task=finetune_bert_op,
        secret_name="gcs-credentials",
        mount_path="/etc/gcs"
    )
    kubernetes.use_config_map_as_env(
        task=finetune_bert_op,
        config_map_name="mlflow-config",
        config_map_key_to_env={"mlflow_tracking_uri": "MLFLOW_TRACKING_URI"}
    )

    evaluate_model_op = model_evaluator.evaluate_model(
        mlflow_run_id=run_id,
        batch_size=batch_size,
        text_col=text_column_name,
        label_col=label_column_name,
        debug_batch_count=debug_batch_count,
        test_dataset=load_dataset_op.outputs["test_dataset"],
        saved_model=finetune_bert_op.outputs["keras_model"]
    )
    kubernetes.use_secret_as_volume(
        task=evaluate_model_op,
        secret_name="gcs-credentials",
        mount_path="/etc/gcs"
    )
    kubernetes.use_config_map_as_env(
        task=evaluate_model_op,
        config_map_name="mlflow-config",
        config_map_key_to_env={"mlflow_tracking_uri": "MLFLOW_TRACKING_URI"}
    )

    validate_model_op = model_validator.validate_model(
        mlflow_run_id=run_id,
        experiment_name=experiment_name,
        registered_model_name=registered_model_name,
        baseline=baseline
    )
    kubernetes.use_secret_as_volume(
        task=validate_model_op,
        secret_name="gcs-credentials",
        mount_path="/etc/gcs"
    )
    kubernetes.use_config_map_as_env(
        task=validate_model_op,
        config_map_name="mlflow-config",
        config_map_key_to_env={"mlflow_tracking_uri": "MLFLOW_TRACKING_URI"}
    )
    validate_model_op.after(evaluate_model_op)


if __name__ == "__main__":
    from kfp.compiler import Compiler
    from kfp.client import Client
    from utils.misc import load_config
    from types import SimpleNamespace

    config_dict = load_config("./configs/training.yaml")
    configs = SimpleNamespace(**config_dict)

    compiler = Compiler()
    compiler.compile(
        pipeline_func=training_pipeline,
        package_path="./pipelines/training_pipeline.yaml"
    )

    client = Client(host=configs.kfp_endpoint)
    client.create_run_from_pipeline_func(
        pipeline_func=training_pipeline,
        arguments={
            "experiment_name": configs.experiment,
            "registered_model_name": configs.registered_model,
            "baseline": configs.baseline,
            "dataset_name": configs.dataset,
            "text_column_name": configs.text_column,
            "label_column_name": configs.label_column,
            "batch_size": configs.batch_size,
            "epochs": configs.epochs,
            "debug_batch_count": configs.debug_batch_count,
        },
        experiment_name=configs.experiment
    )
