import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import smooth.measures
import smooth.datasets
import smooth.util


def relu_net(x, y):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,), dtype=tf.float32),
            keras.layers.Dense(10, activation="relu", dtype=tf.float32),
            keras.layers.Dense(2, dtype=tf.float32),
        ]
    )
    model.compile(loss="mse", metrics=["mae"], optimizer=tf.keras.optimizers.SGD(0.01))
    model.fit(x, y, epochs=500, verbose=1)
    return model


def f_ex0(x):
    return np.array(
        [np.abs(x[0]) + np.abs(x[1]), np.abs(x[0]) - np.abs(x[1])], dtype=np.float32
    )


f_ex = np.vectorize(f_ex0, signature="(2)->(2)")
x_ex = np.reshape(np.moveaxis(np.mgrid[-5:5.1:0.5, -5:5.1:0.5], 0, -1), (-1, 2))
x_ex = x_ex.astype(np.float32)
y_ex = f_ex(x_ex)
model = relu_net(x_ex, y_ex)


def test_relu_net():
    x = np.array([[7, 6]], dtype=np.float32)
    y = model.predict(x)
    assert np.allclose(y, f_ex(x), atol=0.5)


def test_path_length_f():
    x1 = np.array([1, 1], dtype=np.float32)
    x2 = np.array([2, 1], dtype=np.float32)
    tv = smooth.measures.path_length_one_sample(
        model, x1, x2, n_samples=100, derivative=False
    )
    # x1 and x2 are selected in a way where the function between them is just linear,
    # so we can calculate the total variation directly
    assert np.isclose(tv, np.linalg.norm(f_ex(x1) - f_ex(x2)), atol=0.5)


def test_path_length_d():
    x1 = np.array([1, 1], dtype=np.float32)
    x2 = np.array([2, 1], dtype=np.float32)
    tv = smooth.measures.path_length_one_sample(
        model, x1, x2, n_samples=100, derivative=True
    )
    # Within one "segment" of this piecewise linear function, the Jacobian is constant
    assert np.isclose(tv, 0, atol=0.5)
    x2 = np.array([-1, 1], dtype=np.float32)
    tv = smooth.measures.path_length_one_sample(
        model, x1, x2, n_samples=100, derivative=True
    )
    # Here we cross a border and the Jacobian changes by 2 at two positions,
    # so by taking the Frobenius norm of the difference we have sqrt(2^2+2^2)=sqrt(8)
    assert np.isclose(tv, np.sqrt(8), atol=0.5)


def test_path_length_f_lower_bound():
    with smooth.util.NumpyRandomSeed(152):
        y = np.random.randn(10)
    dataset = smooth.datasets.Dataset(x_train=y, y_train=y, x_test=[], y_test=[])

    lb = smooth.measures.path_length_f_lower_bound(dataset, use_test_set=False)

    lb_true = 0
    for y1 in y:
        for y2 in y:
            lb_true += np.abs(y1 - y2)
    lb_true /= len(y) ** 2

    assert np.isclose(lb, lb_true, atol=1e-9)
