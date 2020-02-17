import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras

from smooth import measures


def linear_regression(x, y):
    model = keras.Sequential([keras.layers.Dense(1)])
    model.compile(
        loss="mse", metrics=["accuracy"], optimizer=tf.keras.optimizers.SGD(0.01)
    )
    model.fit(x, y, epochs=1000, verbose=0)
    return model


def f_ex(x):
    return x * 2 + 1


x_ex = np.array([1, 2, 3, 4, 5])
y_ex = f_ex(x_ex)


def test_linreg():
    model = linear_regression(x_ex, y_ex)
    x = np.array([6])
    y = model.predict(x)
    assert np.isclose(y, f_ex(x), atol=0.5)


def test_segment_total_variation():
    model = linear_regression(x_ex, y_ex)
    x1 = 1
    x2 = 5
    tv = measures._segment_total_variation(model, x1, x2, n_samples=100,
                                           derivative=False)
    assert np.isclose(tv, np.abs(f_ex(5) - f_ex(1)), atol=0.5)

def test_segment_total_variation_derivative():
    model = linear_regression(x_ex, y_ex)
    x1 = 1
    x2 = 5
    print("TODO: fix the version with the derivative")
    tv = measures._segment_total_variation(model, x1, x2, n_samples=100,
                                           derivative=True)
    assert np.isclose(tv, np.abs(f_ex(5) - f_ex(1)), atol=0.5)
