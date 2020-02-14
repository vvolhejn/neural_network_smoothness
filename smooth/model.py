import re
import os
import datetime

import numpy as np
import tensorflow as tf

assert tf.__version__[0] == "2"

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

from smooth.datasets import ClassificationDataset
from smooth import measures, callbacks


def get_measures(model, dataset, max_gradient_norm_samples=1000):
    gradient_norm = measures.gradient_norm(
        model, dataset.x_test[:max_gradient_norm_samples]
    )
    l2 = measures.average_l2(model)
    history = model.history.history

    return dict(
        gradient_norm=gradient_norm,
        l2=l2,
        loss=history["loss"][-1],
        accuracy=history["accuracy"][-1],
        val_loss=history.get("val_loss", [None])[-1],
        val_accuracy=history.get("val_accuracy", [None])[-1],
        actual_epochs=len(history["loss"]),
    )


def split_dataset(x, y, first_part=0.9):
    split = int(len(x) * first_part)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    return x_train, y_train, x_val, y_val


def get_model_id(**kwargs):
    return "{}".format(
        # "exp0211-1",
        # datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "_".join(
            (
                "{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                for key, value in sorted(kwargs.items())
            )
        )
    )


def train(
    dataset: ClassificationDataset,
    learning_rate,
    init_scale,
    log_dir,
    hidden_size=200,
    epochs=1000,
    batch_size=512,
    iteration=None,
    verbose=0,
):
    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=dataset.x_train[0].shape))
    model.add(
        layers.Dense(
            hidden_size,
            activation="relu",
            kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
            bias_initializer=VarianceScaling(scale=init_scale, mode="fan_out"),
        )
    )
    model.add(
        layers.Dense(
            dataset.n_classes,
            kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            bias_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            activation="softmax",
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    x_train, y_train, x_val, y_val = split_dataset(
        dataset.x_train, dataset.y_train, 0.9
    )

    measures_cb = callbacks.Measures(x_val, y_val)

    # How many times do we want to validate and call and the Tensorboard callback
    n_updates = 100
    validation_freq = int(epochs / n_updates)

    model.validation_freq = validation_freq
    model.id = get_model_id(
        learning_rate=learning_rate,
        init_scale=init_scale,
        hidden_size=hidden_size,
        epochs=epochs,
        batch_size=batch_size,
        iteration=iteration,
    )

    model.log_dir = os.path.join(log_dir, model.id)
    # Warning: update_freq is in samples in TF 2.0.0, but in batches in TF 2.1
    tensorboard_callback = callbacks.TensorBoard(model.log_dir, validation_freq)

    file_writer = tf.summary.create_file_writer(os.path.join(model.log_dir, "train"))
    file_writer.set_as_default()

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[  # tqdm_cb,
            measures_cb,
            tf.keras.callbacks.EarlyStopping("loss", min_delta=1e-5, patience=500),
            tensorboard_callback,
            callbacks.Stopping(0.01),
            # checkpoint_callback,
        ],
        validation_data=(x_val, y_val),
        validation_freq=validation_freq,  # evaluate once every <validation_freq> epochs
    )

    model.plot_title = "lr={}, init_scale={:.2f}, ".format(
        learning_rate, init_scale
    ) + "h={}, ep={}".format(hidden_size, epochs)

    return model
