import numpy as np
import tensorflow as tf


class ClassificationDataset:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        assert np.min(y_train) >= 0
        # assert y_train.dtype == int
        self.n_classes = np.max(y_train) + 1


def _get_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    return ClassificationDataset(x_train, y_train, x_test, y_test)


mnist = _get_mnist()
