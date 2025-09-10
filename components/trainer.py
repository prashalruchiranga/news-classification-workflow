from kfp.dsl import component, Input, Output, Dataset, Model

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "datasets>=4.0.0",
        "tensorflow==2.18.0",
        "keras-hub>=0.22.1"
    ]
)
def finetune_bert(
    preset: str,
    batch_size: int,
    text_col: str,
    label_col: str,
    lr: float,
    epochs: int,
    min_delta: float,
    patience: int,
    debug_batch_count: int,
    train_dataset: Input[Dataset],
    val_dataset: Input[Dataset],
    saved_model: Output[Model]
    ):
    import os
    import json
    from datasets import load_from_disk
    from keras_hub.models import BertTextClassifier
    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    from tensorflow.keras.metrics import SparseCategoricalAccuracy
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # shift [1,2,3,4] -> [0,1,2,3]
    def adjust_labels(x, y):
        return x, y - 1
    
    train_split = load_from_disk(train_dataset.path)
    val_split = load_from_disk(val_dataset.path)
    tf_train_dataset = train_split.to_tf_dataset(
        columns=[text_col],
        label_cols=[label_col],
        batch_size=batch_size,
        shuffle=True
    )
    tf_val_dataset = val_split.to_tf_dataset(
        columns=[text_col],
        label_cols=[label_col],
        batch_size=batch_size,
        shuffle=True
    )
    tf_train_dataset = tf_train_dataset.map(adjust_labels)
    tf_val_dataset = tf_val_dataset.map(adjust_labels)

    if debug_batch_count is not None:
        tf_train_dataset = tf_train_dataset.take(debug_batch_count)
        tf_val_dataset = tf_val_dataset.take(debug_batch_count)

    number_of_classes = len(set(train_split[label_col]))
    classifier = BertTextClassifier.from_preset(
        preset=preset,
        num_classes=number_of_classes,
    )
    classifier.compile(
        loss=SparseCategoricalCrossentropy(name="sparse_categorical_loss", from_logits=True),
        metrics = [SparseCategoricalAccuracy(name="sparse_categorical_accuracy")],
        optimizer=Adam(lr)
    )
    assert classifier.backbone.trainable == True

    early_stopping = EarlyStopping(
        min_delta=min_delta,
        patience=patience,
        verbose=True,
        restore_best_weights=True,
    )
    history = classifier.fit(
        x=tf_train_dataset,
        epochs=epochs,
        validation_data=tf_val_dataset,
        callbacks=[early_stopping]
    )

    # Save fine-tuned model
    os.makedirs(saved_model.path, exist_ok=True)
    model_filepath = os.path.join(saved_model.path, "news_clf_model.keras")
    classifier.save(model_filepath)
