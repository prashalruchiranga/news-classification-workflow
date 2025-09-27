from kfp.dsl import component, Output, Dataset

@component(
    base_image="docker.io/prashalruchiranga/news-classifier:components-v1.2"
)
def load_hf_dataset(
    mlflow_run_id: str,
    name: str,
    val_fraction: int, # Fraction of the training dataset to use as the validation split
    seed: int,
    train_dataset: Output[Dataset],
    val_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preview: Output[Dataset]
    ):
    "Load a dataset from the Hugging Face Hub and save it as a DatasetDict artifact"
    import os
    import mlflow
    from datasets import load_dataset

    dataset = load_dataset(name)
    # Split train dataset into train split and val split
    split_dataset = dataset["train"].train_test_split(test_size=val_fraction, shuffle=True, seed=seed)
    train_split = split_dataset["train"]
    val_split = split_dataset["test"]
    # Just take test dataset as test split
    test_split = dataset["test"]

    train_split.save_to_disk(train_dataset.path)
    val_split.save_to_disk(val_dataset.path)
    test_split.save_to_disk(test_dataset.path)
    
    # For preview only
    df = train_split.to_pandas().head(50)
    df.to_csv(preview.path, index=False)

    # Log mlflow experiments
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_params({
            "dataset_name": name,
            "val_data_fraction": val_fraction,
            "dataset_split_seed": seed
        })
