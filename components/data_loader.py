from kfp.dsl import component, Output, Dataset

@component(
    base_image="python:3.11-slim",
    packages_to_install=["datasets>=4.0.0"],
)
def load_hf_dataset(
    name: str,
    # Fraction of the training dataset to use as the validation split. The validation set is taken from the training set, not from the entire dataset
    val_fraction: int,
    seed: int,
    train_dataset: Output[Dataset],
    val_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    preview: Output[Dataset]
    ):
    "Load a dataset from the Hugging Face Hub and save it as a DatasetDict artifact"
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
    df = train_split.to_pandas().head(20)
    df.to_csv(preview.path, index=False)
