from kfp.dsl import pipeline
from components import data_loader, trainer

@pipeline(
    name="news-clf-model-trainer",
    description= "Pipeline to fine-tune a pretrained BERT model for news classification"
)
def training_pipeline(
    dataset_name: str,
    val_data_fraction: int = 0.2,
    dataset_split_seed: int = 42,
    bert_preset: str = "bert_tiny_en_uncased"
    ):
    load_dataset_op = data_loader.load_hf_dataset(
        name=dataset_name, 
        val_fraction=val_data_fraction,
        seed=dataset_split_seed
    )
    finetune_bert_op = trainer.finetune_bert(
        preset=bert_preset,
        train_dataset=load_dataset_op.outputs["train_dataset"],
        val_dataset=load_dataset_op.outputs["val_dataset"]
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
            "dataset_name": configs.dataset
        },
        experiment_name=configs.experiment
    )
