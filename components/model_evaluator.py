from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, ClassificationMetrics

@component(
    base_image="docker.io/prashalruchiranga/news-classification:kfp-base-arm64"
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
    import numpy as np
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
    examples, labels = [], []
    for x, y in tf_test_dataset.as_numpy_iterator():
        examples.append(x)
        labels.append(y)
    examples = np.concatenate(examples, axis=0)
    labels = np.concatenate(labels, axis=0)
    logits = reloaded_model.predict(examples)
    probabilities = tf.nn.softmax(logits, axis=-1)
    predictions = tf.argmax(probabilities, axis=-1)
    matrix = tf.math.confusion_matrix(labels=labels, predictions=predictions)
    confusion_matrix.log_confusion_matrix(
        categories=list(category_lookup.values()),
        matrix=matrix.numpy().tolist()
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix.numpy(),
        display_labels=list(category_lookup.values())
    )
    display.plot()

    # Calculate precision, recall, and f1 score
    metric = tf.keras.metrics.Precision()
    metric.update_state(y_true=labels, y_pred=predictions)
    precision = metric.result()
    metric = tf.keras.metrics.Recall()
    metric.update_state(y_true=labels, y_pred=predictions)
    recall = metric.result()
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Log mlflow experiments
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/etc/gcs/key.json"
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_id=mlflow_run_id):
        mlflow.log_metrics({
            "loss": loss, 
            "sparse_categorical_accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })
        mlflow.log_figure(display.figure_, "confusion_matrix.png")
