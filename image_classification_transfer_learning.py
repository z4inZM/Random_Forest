import os
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3, ResNet50
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess

VAL_SPLIT = 0.1
SHUFFLE_BUFFER = 10000
MIN_EPOCHS = 10
MAX_EPOCHS = 20


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()
    split_index = int((1 - VAL_SPLIT) * len(x_train))
    x_val, y_val = x_train[split_index:], y_train[split_index:]
    x_train, y_train = x_train[:split_index], y_train[:split_index]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_dataset(images, labels, image_size, preprocess_fn, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(min(len(images), SHUFFLE_BUFFER))

    def _preprocess(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, image_size)
        image = preprocess_fn(image)
        return image, label

    return (
        dataset.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_resnet50(image_size, num_classes):
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="ResNet50_transfer")


def build_inception_v3(image_size, num_classes):
    base_model = InceptionV3(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3),
        pooling="avg",
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="InceptionV3_transfer")


def train_model(model, train_ds, val_ds, epochs):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    start_time = time.perf_counter()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    training_time = time.perf_counter() - start_time
    return history, training_time


def model_size_mb(model):
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model.keras")
        model.save(model_path, include_optimizer=False)
        size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 * 1024)


def plot_metric(histories, metric, filename):
    plt.figure(figsize=(9, 6))
    for name, history in histories.items():
        values = history.history.get(metric, [])
        epochs = range(1, len(values) + 1)
        plt.plot(epochs, values, label=f"{name} train")
        val_values = history.history.get(f"val_{metric}")
        if val_values is not None:
            plt.plot(epochs, val_values, linestyle="--", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} vs Epoch")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def print_summary(results):
    header = f"{'Model':<14}{'Test Acc':>12}{'Test Loss':>12}{'Time (s)':>12}{'Size (MB)':>12}"
    print("\nModel Comparison")
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<14}"
            f"{metrics['test_accuracy']:>12.4f}"
            f"{metrics['test_loss']:>12.4f}"
            f"{metrics['training_time_sec']:>12.1f}"
            f"{metrics['model_size_mb']:>12.2f}"
        )


def generate_observations(results):
    resnet = results["ResNet50"]
    inception = results["InceptionV3"]
    if resnet["test_accuracy"] >= inception["test_accuracy"]:
        accuracy_line = (
            "ResNet50 achieved higher accuracy, likely because residual connections make optimization "
            "easier and help deeper features transfer effectively."
        )
    else:
        accuracy_line = (
            "InceptionV3 achieved higher accuracy, likely because its multi-scale filters capture "
            "features at different receptive fields that suit CIFAR-10 images."
        )
    if resnet["training_time_sec"] >= inception["training_time_sec"]:
        time_line = "ResNet50 took longer to train, which is expected with its deeper architecture."
    else:
        time_line = "InceptionV3 took longer to train, likely due to its larger input resolution."
    if resnet["model_size_mb"] >= inception["model_size_mb"]:
        size_line = "ResNet50 has a larger model size, reflecting more parameters."
    else:
        size_line = "InceptionV3 has a larger model size, reflecting its wider inception blocks."
    return "\n".join([accuracy_line, time_line, size_line])


def run_experiment(name, model_builder, preprocess_fn, image_size, datasets, epochs, batch_size, num_classes):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = datasets
    train_ds = build_dataset(x_train, y_train, image_size, preprocess_fn, batch_size, training=True)
    val_ds = build_dataset(x_val, y_val, image_size, preprocess_fn, batch_size, training=False)
    test_ds = build_dataset(x_test, y_test, image_size, preprocess_fn, batch_size, training=False)

    model = model_builder(image_size, num_classes)
    history, training_time = train_model(model, train_ds, val_ds, epochs)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    size_mb = model_size_mb(model)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "training_time_sec": float(training_time),
        "model_size_mb": float(size_mb),
    }
    return metrics, history


def main():
    tf.keras.utils.set_random_seed(42)
    epochs = int(os.environ.get("EPOCHS", "10"))
    if epochs < MIN_EPOCHS or epochs > MAX_EPOCHS:
        raise ValueError(f"EPOCHS must be between {MIN_EPOCHS} and {MAX_EPOCHS}.")
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))

    datasets = load_cifar10()
    num_classes = len(np.unique(datasets[0][1]))

    results = {}
    histories = {}

    resnet_metrics, resnet_history = run_experiment(
        "ResNet50",
        build_resnet50,
        resnet_preprocess,
        (224, 224),
        datasets,
        epochs,
        batch_size,
        num_classes,
    )
    results["ResNet50"] = resnet_metrics
    histories["ResNet50"] = resnet_history

    inception_metrics, inception_history = run_experiment(
        "InceptionV3",
        build_inception_v3,
        inception_preprocess,
        (299, 299),
        datasets,
        epochs,
        batch_size,
        num_classes,
    )
    results["InceptionV3"] = inception_metrics
    histories["InceptionV3"] = inception_history

    plot_metric(histories, "accuracy", "accuracy_vs_epoch.png")
    plot_metric(histories, "loss", "loss_vs_epoch.png")

    print_summary(results)
    print("\nObservations")
    print(generate_observations(results))


if __name__ == "__main__":
    main()
