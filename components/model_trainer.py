from kfp.dsl import component, Input, Output, Dataset, Model, Markdown

@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "datasets>=4.0.0",
        "tensorflow==2.18.0",
        "keras-hub>=0.22.1",
        "matplotlib>=3.10.6"
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
    learning_curves: Output[Markdown],
    saved_model: Output[Model]
    ):
    import os
    import json
    import matplotlib.pyplot as plt
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

    # Save training curves as an artifact
    history_dict = history.history
    acc = history_dict["sparse_categorical_accuracy"]
    val_acc = history_dict["val_sparse_categorical_accuracy"]
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 10))
    fig.tight_layout()

    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')

    local_png_path = "learning_curves.png"
    plt.savefig(local_png_path, format="png")
    plt.close()

    md_content = f"## Training Curves\n\n![Training Curves]({local_png_path})"
    with open(learning_curves.path, "w") as f:
        f.write(md_content)

    # Save fine-tuned model
    os.makedirs(saved_model.path, exist_ok=True)
    model_filepath = os.path.join(saved_model.path, "model.keras")
    classifier.save(model_filepath)
