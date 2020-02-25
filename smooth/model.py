import re
import os
import datetime
from typing import List
import warnings

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


def get_shallow(
    dataset: smooth.datasets.Dataset,
    learning_rate: float,
    init_scale: float,
    hidden_size: int,
    activation: str,  # "relu", "tanh" etc.
):
    # Classification or regression?
    classification = "Classification" in type(dataset).__name__

    model = tf.keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=dataset.x_train[0].shape))
    model.add(
        layers.Dense(
            hidden_size,
            activation=activation,
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

    return model


def train_shallow(
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
    train_val_split=0.9,
    activation="relu",
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
    :param activation: activation function for the hidden layer
    :return: A trained model.
    """
    model = get_shallow(dataset, learning_rate, init_scale, hidden_size, activation)

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
        dataset=dataset.name,
    )

    callbacks += [
        smooth.callbacks.Stopping(loss_threshold),
    ]

    if log_dir is not None:
        # Use TensorBoard.
        model.log_dir = os.path.join(log_dir, model.id)
        print("Log dir: ", model.log_dir)

        file_writer = tf.summary.create_file_writer(
            os.path.join(model.log_dir, "train")
        )
        file_writer.set_as_default()

        measures_cb = smooth.callbacks.Measures(dataset.x_test, dataset.y_test)
        tensorboard_cb = smooth.callbacks.TensorBoard(model.log_dir, validation_freq)

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


def interpolate_relu_network(dataset: smooth.datasets.Dataset, use_test_set=False):
    """
    Creates a shallow ReLU network which interpolates the 1D training data.
    Returns an error when the data is multidimensional.
    If `use_test_set` is True, interpolates the test set rather than the training set.
    """
    if use_test_set:
        x = dataset.x_test.squeeze()
        y = dataset.y_test.squeeze()
    else:
        x = dataset.x_train.squeeze()
        y = dataset.y_train.squeeze()

    assert len(x.shape) == 1
    model = get_shallow(
        dataset,
        learning_rate=0.0,  # Not used, but needed in model compilation
        init_scale=1.0,  # Not used, but needed in model compilation
        hidden_size=len(x),
        activation="relu",
    )
    weights = model.get_weights()
    weights[0] = np.ones_like(weights[0])
    weights[1] = -x
    weights[2] = np.zeros_like(weights[2])
    weights[3] = np.reshape(y[0], (1,))

    # We need to go through the array ordered by increasing x
    p = np.argsort(x)
    slope = 0.
    for i in range(0, len(x) - 1):
        target_slope = (y[p[i + 1]] - y[p[i]]) / (x[p[i + 1]] - x[p[i]])
        weights[2][p[i]] = target_slope - slope
        slope = target_slope

    model.set_weights(weights)

    return model

def interpolate_polynomial(dataset: smooth.datasets.Dataset, deg=None):
    """
    Fits a polynomial to the training data. If the degree is not given, it is chosen
    as the lowest degree which can interpolate the data, so `len(x_train)-1`.
    Returns a tf.keras.Model representation of the polynomial.
    """
    assert len(dataset.x_train.squeeze().shape) == 1
    if deg is None:
        deg = len(dataset.x_train) - 1

    with warnings.catch_warnings():
        # The fit() function raises warnings about the polynomial being ill-conditioned
        # but we don't care about that
        warnings.simplefilter("ignore")

        poly = np.polynomial.polynomial.Polynomial.fit(
            x=np.squeeze(dataset.x_train),
            y=np.squeeze(dataset.y_train),
            deg=deg,
        )

    y_pred = poly(dataset.x_test)
    # .linspace(n=200, domain=(-1, 1))
    poly_dataset = smooth.datasets.Dataset(dataset.x_test, y_pred, [], [])
    model = smooth.model.interpolate_relu_network(poly_dataset)
    return model
