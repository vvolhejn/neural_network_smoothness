import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import smooth.model
import smooth.datasets


def test_interpolate_relu_network():
    dataset = smooth.datasets.GaussianProcessDataset(
        seed=123, samples_train=100, lengthscale=0.1
    )

    model = smooth.model.interpolate_relu_network(dataset)
    y_pred = model.predict(dataset.x_train)
    assert np.allclose(dataset.y_train, y_pred, atol=1e-3)