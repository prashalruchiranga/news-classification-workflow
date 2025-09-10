from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "datasets>=4.0.0",
        "tensorflow==2.18.0"
    ]
)
def evaluate_model(
    batch_size: int,
    text_col: str,
    label_col: str,
    test_dataset: Input[Dataset],
    saved_model: Input[Model],
    metrics: Output[Metrics]
):
    from datasets import load_from_disk
    import tensorflow as tf

    # shift [1,2,3,4] -> [0,1,2,3]
    def adjust_labels(x, y):
        return x, y - 1

    test_split = load_from_disk(test_dataset.path)
    tf_test_dataset = test_split.to_tf_dataset(
        columns=[text_col],
        label_cols=[label_col],
        batch_size=batch_size,
        shuffle=True
    )
    tf_test_dataset = tf_test_dataset.map(adjust_labels)

    model_filepath = f"{saved_model.path}/model.keras"
    reloaded_model = tf.keras.models.load_model(model_filepath)
    loss, accuracy = reloaded_model.evaluate(tf_test_dataset)
    metrics.log_metric("loss", float(loss))
    metrics.log_metric("sparse_categorical_accuracy", float(accuracy))
