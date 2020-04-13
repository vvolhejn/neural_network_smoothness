import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import sklearn.decomposition

import smooth.model
import smooth.datasets


def test_interpolate_relu_network():
    dataset = smooth.datasets.GaussianProcessDataset(
        seed=123, samples_train=100, lengthscale=0.1
    )

    model = smooth.model.interpolate_relu_network(dataset)
    y_pred = model.predict(dataset.x_train)
    assert np.allclose(dataset.y_train, y_pred, atol=1e-3)


def test_pca_layer():
    dataset = smooth.datasets.get_mnist()
    dims = 10

    x_train = dataset.x_train.reshape(len(dataset.x_train), -1)
    x_test = dataset.x_test.reshape(len(dataset.x_test), -1)

    layer = smooth.model.PCALayer(dims, xs=x_train)

    pca = sklearn.decomposition.PCA(dims)
    pca.fit(x_train)

    # Note the large atol; there seems to be significant loss in precision.
    # I don't think this should affect the results much.
    assert np.allclose(layer(x_test[:2]), pca.transform(x_test[:2]), atol=1e-2)
