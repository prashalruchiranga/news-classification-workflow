from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@component(
    base_image="docker.io/prashalruchiranga/news-classifier:components-v1.2"
)
def evaluate_model(
    mlflow_run_id: str,
    batch_size: int,
    text_col: str,
    label_col: str,
    category_lookup: dict,
    debug_batch_count: int,
    test_dataset: Input[Dataset],
    saved_model: Input[Model],
    metrics: Output[Metrics],
    confusion_matrix: Output[ClassificationMetrics]
):
    import os
    from datasets import load_from_disk
    import tensorflow as tf
    import mlflow
    from sklearn.metrics import ConfusionMatrixDisplay

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

    if debug_batch_count is not None:
        tf_test_dataset = tf_test_dataset.take(debug_batch_count)

    model_filepath = f"{saved_model.path}/model.keras"
    reloaded_model = tf.keras.models.load_model(model_filepath)
    loss, accuracy = reloaded_model.evaluate(tf_test_dataset)
    metrics.log_metric("loss/test", float(loss))
    metrics.log_metric("sparse_categorical_accuracy/test", float(accuracy))

    # Design the confusion matrix
    labels = tf.convert_to_tensor(test_dataset["label"]) - 1
    logits = reloaded_model.predict(test_dataset["description"])
    probabilities = tf.nn.softmax(logits, axis=-1)
    pred_class_ids = tf.argmax(probabilities, axis=-1)
    matrix = tf.math.confusion_matrix(labels=labels, predictions=pred_class_ids)
    confusion_matrix.log_confusion_matrix(
        categories=list(category_lookup.values()),
        matrix=matrix.numpy().tolist()
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.numpy(),
        display_labels=list(category_lookup.values())
    )
    display.plot()

    # Log mlflow experiments
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            "loss/test": loss, 
            "sparse_categorical_accuracy/test": accuracy
        })
        mlflow.log_figure(display.figure_, "confusion_matrix.png")
