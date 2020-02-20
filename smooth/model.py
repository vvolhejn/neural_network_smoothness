import re
import os
import datetime
from typing import List

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import VarianceScaling

import smooth.datasets
import smooth.measures
import smooth.callbacks

assert tf.__version__[0] == "2"


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


def train_shallow_relu(
    dataset: smooth.datasets.Dataset,
    learning_rate,
    init_scale,
    hidden_size=200,
    epochs=1000,
    batch_size=512,
    iteration=None,
    verbose=0,
    loss_threshold=0.0,
    log_dir=None,
    callbacks: List[tf.keras.callbacks.Callback] = [],
    train_val_split=0.9
):
    """
    Trains a single-layer ReLU network.

    :param dataset:
    :param learning_rate:
    :param init_scale:
    :param hidden_size:
    :param epochs:
    :param batch_size:
    :param iteration: Used only as an additional identifier
        (for training multiple models with the same hyperparams)
    :param verbose: Verbosity level passed to `model.fit`
    :param loss_threshold: Stop training when this loss is reached.
    :param log_dir: Where to save Tensorboard logs. If None, Tensorboard is not used.
    :param callbacks: list of tf.keras.Callback functions
    :param train_val_split: 0.9 = use 90% for training, 10% for validation
    :return: A trained model.
    """
    # Classification or regression?
    classification = "Classification" in type(dataset).__name__

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
            dataset.n_classes if classification else 1,
            kernel_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            bias_initializer=VarianceScaling(scale=init_scale, mode="fan_in"),
            activation="softmax" if classification else None,
        )
    )
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate),
        loss="sparse_categorical_crossentropy" if classification else "mse",
        metrics=["accuracy"] if classification else ["mae"],
    )

    x_train, y_train, x_val, y_val = split_dataset(
        dataset.x_train, dataset.y_train, train_val_split
    )

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

    callbacks += [
        smooth.callbacks.Stopping(loss_threshold),
    ]

    if log_dir is not None:
        # Use TensorBoard.
        measures_cb = smooth.callbacks.Measures(x_val, y_val)
        tensorboard_cb = smooth.callbacks.TensorBoard(model.log_dir, validation_freq)

        model.log_dir = os.path.join(log_dir, model.id)
        file_writer = tf.summary.create_file_writer(
            os.path.join(model.log_dir, "train")
        )
        file_writer.set_as_default()

        callbacks += [tensorboard_cb, measures_cb]

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks,
        # The case `len(x_val) == 0` occurs if `validation_split == 1.0`
        validation_data=(x_val, y_val) if len(x_val) > 0 else None,
        validation_freq=validation_freq,  # evaluate once every <validation_freq> epochs
    )

    model.plot_title = "lr={:.2f}, init_scale={:.2f}, ".format(
        learning_rate, init_scale
    ) + "h={}, ep={}".format(hidden_size, epochs)

    return model
