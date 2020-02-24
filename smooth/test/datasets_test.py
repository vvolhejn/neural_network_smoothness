import os

import numpy as np

from smooth import datasets


def test_gp_name_conversion():
    seed = 123
    lengthscale = 0.456
    samples_train = 78
    dataset = datasets.GaussianProcessDataset(
        seed=seed, samples_train=samples_train, lengthscale=lengthscale
    )

    name = dataset.name
    for x in [seed, lengthscale, samples_train]:
        assert str(x) in name

    dataset2 = datasets.GaussianProcessDataset.from_name(name)
    assert dataset2.name == name

    assert np.allclose(dataset.x_train, dataset2.x_train)
    assert np.allclose(dataset.x_test, dataset2.x_test)
    assert np.allclose(dataset.y_train, dataset2.y_train)
    assert np.allclose(dataset.y_test, dataset2.y_test)
