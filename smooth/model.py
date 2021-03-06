"""
Function which create models either by training or by directly computing their parameters.
"""

import os
from typing import List, Optional
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
import GPy
import sklearn.kernel_ridge

import smooth.datasets
import smooth.callbacks
import smooth.measures
import smooth.util

assert tf.__version__[0] == "2"


def split_dataset(x, y, first_part=0.9):
    """Creates a validation set."""
    split = int(len(x) * first_part)
    x_train = x[:split]
    y_train = y[:split]
    x_val = x[split:]
    y_val = y[split:]
    return x_train, y_train, x_val, y_val


class PCALayer(tf.keras.layers.Dense):
    """
    Applies PCA as a dense layer.
    """

    def __init__(self, dims, xs: np.ndarray):
        pca = sklearn.decomposition.PCA(n_components=dims)
        pca.fit(xs.reshape(len(xs), -1))

        super().__init__(
            dims,
            trainable=False,
            weights=[pca.components_.T, -np.dot(pca.components_, pca.mean_)],
        )


def get_shallow(
    dataset: smooth.datasets.Dataset,
    init_scale: float,
    hidden_size: int,
    activation: str,  # "relu", "tanh" etc.
    pca_dims: Optional[int] = None,
) -> tf.keras.Model:
    """
    Create a two-layer neural network without training it.
    """
    # Classification or regression?
    classification = "Classification" in type(dataset).__name__

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=dataset.x_shape()))

    if pca_dims is not None:
        model.add(PCALayer(pca_dims, dataset.x_train))

    model.add(
        tf.keras.layers.Dense(
            hidden_size,
            activation=activation,
            kernel_initializer=VarianceScaling(
                scale=init_scale, mode="fan_avg", distribution="uniform"
            ),
        )
    )
    model.add(
        tf.keras.layers.Dense(
            dataset.n_classes if classification else 1,
            kernel_initializer=VarianceScaling(
                scale=init_scale, mode="fan_avg", distribution="uniform"
            ),
            activation=None,
        )
    )

    return model


def train_shallow(
    dataset: smooth.datasets.Dataset,
    learning_rate: float,
    init_scale: float,
    hidden_size=200,
    epochs=1000,
    batch_size=None,
    iteration=None,
    verbose=0,
    error_threshold=0.0,
    log_dir=None,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    activation="relu",
    gradient_norm_reg_coef=0.0,
    gradient_norm_squared_reg_coef=0.0,
    weights_product_reg_coef=0.0,
    path_length_f_reg_coef=0.0,
    path_length_d_reg_coef=0.0,
    early_stopping_patience=None,
    early_stopping_min_delta=None,
    model_id=None,
    pca_dims=None,
):
    """
    Train a two-layer neural network.

    :param dataset: the Dataset to use
    :param learning_rate: Learning rate for SGD
    :param init_scale: Model initialization scale
    :param hidden_size: Size of the hidden layer
    :param epochs: How many epochs to train for
    :param batch_size: If None, will perform GD rather than SGD.
    :param iteration: Used only as an additional identifier
        (for training multiple models with the same hyperparams)
    :param verbose: Verbosity level passed to `model.fit`
    :param error_threshold: Stop training when this loss is reached.
    :param log_dir: Where to save Tensorboard logs. If None, Tensorboard is not used.
    :param callbacks: List of tf.keras.Callback functions
    :param activation: Activation function for the hidden layer
    :param gradient_norm_reg_coef: Gradient norm regularization
    :param gradient_norm_squared_reg_coef: Squared gradient norm regularization
    :param weights_product_reg_coef: Weights product regularization
    :param path_length_f_reg_coef: Function path length regularization
    :param path_length_d_reg_coef: Gradient path length regularization
    :param early_stopping_patience: How many epochs to wait for loss to improve
    :param early_stopping_min_delta: Minimum loss difference to count as improvement
    :param model_id: Override the generated model id
    :param pca_dims: Reduce the dataset using PCA to this many dimensions

    :return: A trained model.
    """
    if callbacks is None:
        callbacks = []

    model = get_shallow(dataset, init_scale, hidden_size, activation, pca_dims)

    model = add_explicit_regularization(
        model,
        gradient_norm_reg_coef,
        gradient_norm_squared_reg_coef,
        weights_product_reg_coef,
        path_length_f_reg_coef,
        path_length_d_reg_coef,
    )

    classification = "Classification" in type(dataset).__name__
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate),
        loss=(
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            if classification
            else "mse"
        ),
        metrics=["accuracy"] if classification else ["mae", "mse"],
    )

    x_train, y_train = dataset.x_train, dataset.y_train

    # How many times do we want to validate and call and the Tensorboard callback
    n_updates = 100
    validation_freq = max(10, int(epochs / n_updates))
    model.validation_freq = validation_freq
    if model_id is not None:
        model.id = model_id
    else:
        model.id = smooth.util.dict_to_short_string(
            dict(
                learning_rate=learning_rate,
                init_scale=init_scale,
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=batch_size,
                iteration=iteration,
                dataset=dataset.name,
                gradient_norm_reg_coef=gradient_norm_reg_coef,
                weights_product_reg_coef=weights_product_reg_coef,
            )
        )

    # Add early stopping if desired.
    assert (early_stopping_min_delta is None) == (early_stopping_patience is None)
    if early_stopping_patience is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta,
            )
        )

    callbacks += [
        smooth.callbacks.Stopping(error_threshold),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    if log_dir is not None:
        # Use TensorBoard.
        model.log_dir = os.path.join(log_dir, model.id)
        print("Log dir: ", model.log_dir)

        file_writer = tf.summary.create_file_writer(
            os.path.join(model.log_dir, "train")
        )
        file_writer.set_as_default()

        measures_cb = smooth.callbacks.Measures(dataset)

        callbacks += [measures_cb]

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size or len(x_train),  # None -> use GD, not SGD
        verbose=verbose,
        callbacks=callbacks,
        validation_data=(dataset.x_test, dataset.y_test),
        validation_freq=validation_freq,  # evaluate once every <validation_freq> epochs
    )

    model.plot_title = "lr={:.2f}, init_scale={:.2f}, ".format(
        learning_rate, init_scale
    ) + "h={}, ep={}".format(hidden_size, epochs)

    return model


def add_explicit_regularization(
    model: tf.keras.Model,
    gradient_norm_reg_coef: float,
    gradient_norm_squared_reg_coef: float,
    weights_product_reg_coef: float,
    path_length_f_reg_coef: float,
    path_length_d_reg_coef: float,
):
    """Wraps the model to add explicit regularization."""
    if gradient_norm_reg_coef > 0:
        model = RegularizedGradientModel(model, coef=gradient_norm_reg_coef,)

    if gradient_norm_squared_reg_coef > 0:
        model = RegularizedGradientModel(
            model, coef=gradient_norm_squared_reg_coef, power=2,
        )

    if weights_product_reg_coef > 0:
        model = RegularizedWeightsProductModel(model, coef=weights_product_reg_coef,)

    if path_length_f_reg_coef > 0:
        model = RegularizedPathLengthModel(
            model, coef=path_length_f_reg_coef, derivative=False
        )

    if path_length_d_reg_coef > 0:
        model = RegularizedPathLengthModel(
            model, coef=path_length_d_reg_coef, derivative=True
        )

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
    slope = 0.0
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
            x=np.squeeze(dataset.x_train), y=np.squeeze(dataset.y_train), deg=deg,
        )

    y_pred = poly(dataset.x_test)
    # .linspace(n=200, domain=(-1, 1))
    poly_dataset = smooth.datasets.Dataset(dataset.x_test, y_pred, [], [])
    model = smooth.model.interpolate_relu_network(poly_dataset)
    return model


class SklearnModel:
    """
    Wraps a scikit-learn model and pretends it's a Keras model. This is useful
    for evaluating our measures.
    """

    def __init__(self, clf):
        self.clf = clf
        self.differentiable = False

    def predict(self, x, **kwargs):
        # Silently ignores other arguments meant for Keras models
        return self.clf.predict(x)


class GPModel:
    """
    Wraps a GPy model and pretends it's a Keras model. This is useful
    for evaluating our measures.
    """

    def __init__(self, model: GPy.models.GPRegression):
        self.model = model
        self.differentiable = False

    def predict(self, x, **kwargs):
        # Silently ignores other arguments meant for Keras models
        return self.model.predict_noiseless(x)[0]


def is_differentiable(model: tf.keras.Model):
    """
    We are duck-typing non-Keras models, such as sklearn models. This means that we
    might not be able to take the model's gradient.
    """
    if not hasattr(model, "differentiable"):
        return True
    else:
        return model.differentiable


def train_model(name, dataset, **kwargs):
    """
    Train a neural network or a kernel ridge regression model.
    """
    if name == "krr":
        krr = sklearn.kernel_ridge.KernelRidge(kernel="poly", coef0=1, **kwargs)
        krr.fit(dataset.x_train, dataset.y_train)
        return SklearnModel(krr), {}
    elif name == "shallow":
        if kwargs["batch_size"] is None:
            # Batch size == None -> use GD (batch size is the training set's size)
            kwargs["batch_size"] = len(dataset.x_train)

        model = smooth.model.train_shallow(dataset=dataset, **kwargs,)

        model_to_save = model
        while not isinstance(model_to_save, tf.keras.models.Sequential):
            # The wrapper is not serializable.
            model_to_save = model_to_save.model

        model_to_save.save(os.path.join(model.log_dir, "model.h5"))
        return model, {"log_dir": model.log_dir}
    else:
        raise ValueError("Unknown model name: {}".format(name))


class RegularizedGradientModel(tf.keras.Model):
    """
    A wrapper which adds to the loss a regularization term penalizing large gradients.
    """
    def __init__(
        self, model: tf.keras.Model, coef: float, x_val: np.ndarray = None, power=1,
    ):
        super(RegularizedGradientModel, self).__init__()
        self.model = model
        self.coef = coef
        self.power = power
        if x_val is not None:
            self.x_reg = tf.constant(x_val)
        else:
            self.x_reg = None

    def call(self, x):
        reg_loss = 0.0
        if self.coef != 0:
            x_reg = self.x_reg or x
            reg_loss = self.coef * smooth.measures.gradient_norm(
                self.model, x_reg, self.power
            )

        self.add_loss(reg_loss)

        return self.model(x)


class RegularizedWeightsProductModel(tf.keras.Model):
    """
    A wrapper which adds to the loss a regularization term penalizing
    the weights product measure.
    """
    def __init__(
        self, model: tf.keras.Model, coef: float,
    ):
        super(RegularizedWeightsProductModel, self).__init__()
        self.model = model
        self.coef = coef

    def call(self, x):
        self.add_loss(self.coef * smooth.measures.weights_product(self.model))
        return self.model(x)


class RegularizedPathLengthModel(tf.keras.Model):
    """
    A wrapper which adds to the loss a regularization term penalizing
    the function or gradient path length.
    """
    def __init__(
        self, model: tf.keras.Model, derivative: bool, coef: float,
    ):
        super(RegularizedPathLengthModel, self).__init__()
        self.model = model
        self.derivative = derivative
        self.coef = coef

    def call(self, x: tf.Tensor):
        # Truncate to an even number of elements so that we can form pairs
        l = tf.shape(x)[0]
        x_pairs = tf.split(x[: l - l % 2], 2, axis=0)

        self.add_loss(
            self.coef
            * tf.reduce_sum(
                smooth.measures.total_variation_along_segment(
                    self.model,
                    x_pairs[0],
                    x_pairs[1],
                    derivative=self.derivative,
                    n_samples=10,
                )
            )
        )

        return self.model(x)
