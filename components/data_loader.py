from kfp.dsl import component, Output, Dataset

@component(
    base_image="python:3.11-slim",
    packages_to_install=["datasets>=4.0.0"],
)
def load_hf_dataset(name: str, output: Output[Dataset], preview: Output[Dataset]):
    "Load a dataset from the Hugging Face Hub and save it as a DatasetDict artifact"
    from datasets import load_dataset
    dataset = load_dataset(name)
    dataset.save_to_disk(output.path)
    #For preview only
    df = dataset["train"].to_pandas().head(20)
    df.to_csv(preview.path, index=False)
