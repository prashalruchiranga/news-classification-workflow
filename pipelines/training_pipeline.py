from kfp.dsl import pipeline
from components import data_loader, model_trainer, model_evaluator
from typing import Optional

@pipeline(
    name="news-clf-model-trainer",
    description= "Pipeline to fine-tune a pre-trained BERT model for news classification"
)
def training_pipeline(
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
    max_no_improve_epochs: int = 5,
    debug_batch_count: Optional[int] = None
    ):
    load_dataset_op = data_loader.load_hf_dataset(
        name=dataset_name, 
        val_fraction=val_data_fraction,
        seed=dataset_split_seed
    )
    finetune_bert_op = model_trainer.finetune_bert(
        preset=bert_preset,
        batch_size=batch_size,
        text_col=text_column_name,
        label_col=label_column_name,
        lr=learning_rate,
        epochs=epochs,
        min_delta=early_stopping_min_delta,
        patience=max_no_improve_epochs,
        debug_batch_count=debug_batch_count,
        train_dataset=load_dataset_op.outputs["train_dataset"],
        val_dataset=load_dataset_op.outputs["val_dataset"]
    )
    evaluate_model_op = model_evaluator.evaluate_model(
        batch_size=batch_size,
        text_col=text_column_name,
        label_col=label_column_name,
        test_dataset=load_dataset_op.outputs["test_dataset"],
        saved_model=finetune_bert_op.outputs["saved_model"]
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler
    from kfp.client import Client
    from utils.misc import load_config
    from types import SimpleNamespace

    config_dict = load_config("./config/config.yaml")
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
            "dataset_name": configs.dataset,
            "text_column_name": configs.text_column,
            "label_column_name": configs.label_column,
            "batch_size": configs.batch_size,
            "epochs": configs.epochs,
            "debug_batch_count": configs.debug_batch_count
        },
        experiment_name=configs.experiment
    )
