import logging

import numpy as np
import tensorflow as tf
import GPy
import matplotlib.pyplot as plt

import smooth.util


class Dataset:
    def __init__(self, x_train, y_train, x_test, y_test, name=None):
        # Here we cast integers to floats as well, which maybe we should avoid.
        def preprocess(arr):
            arr = np.array(arr).astype(dtype=np.float32)
            if len(arr.shape) == 1:
                return arr[:, np.newaxis]
            else:
                return arr

        x_train, y_train, x_test, y_test = [
            preprocess(arr) for arr in [x_train, y_train, x_test, y_test]
        ]
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name
        assert x_train.shape[0] == y_train.shape[0]
        assert x_test.shape[0] == y_test.shape[0]
        assert x_train.shape[1:] == x_test.shape[1:]
        assert y_train.shape[1:] == y_test.shape[1:]

    def subset(self, n_samples: int, keep_test_set=False):
        """
        Selects the first n_samples of the training set, and an appropriately scaled
        number of samples of the test set.
        The data is not copied, but returned as a view.

        :param n_samples: How many samples to select
        :param keep_test_set: If True, does not subsample the test set
        :return: A smaller dataset
        """

        if n_samples > len(self.x_train):
            raise ValueError(
                "Trying to take {} samples from a {}-sample dataset ({})".format(
                    n_samples, len(self.x_train), self
                )
            )
        assert n_samples <= len(self.x_train)
        if keep_test_set:
            n_samples_test = len(self.x_test)
        else:
            n_samples_test = int(n_samples / len(self.x_train) * len(self.x_test))

        name = self.name
        if self.name is not None:
            name = "{} ({}-sample subset)".format(self.name, n_samples)

        return Dataset(
            x_train=self.x_train[:n_samples],
            y_train=self.y_train[:n_samples],
            x_test=self.x_test[:n_samples_test],
            y_test=self.y_test[:n_samples_test],
            name=name,
        )

    def __repr__(self):
        res = "'{}'".format(self.name or type(self).__name__)

        res += " (x.shape={}, y.shape={}, training samples={}, test samples={})".format(
            self.x_shape(), self.y_shape(), len(self.x_train), len(self.x_test),
        )
        return res

    def x_shape(self):
        return self.x_train.shape[1:]

    def y_shape(self):
        return self.y_train.shape[1:]


class ClassificationDataset(Dataset):
    def __init__(self, x_train, y_train, x_test, y_test, name=None):
        super().__init__(x_train, y_train, x_test, y_test, name)
        assert np.min(y_train) >= 0
        assert np.min(y_test) >= 0
        self.n_classes = np.max(y_train) + 1


def add_label_noise(dataset: Dataset, p: float):
    """
    Adds label noise to the training set. The test set is unaffected.
    With probability p for each label, the label is changed to a random label.
    Note that the label may be the original label as well.
    Modifies the dataset in-place.
    """

    labels = np.unique(dataset.y_train)

    if len(dataset.y_train) >= 100 and len(labels) >= len(dataset.y_train) / 2:
        # Heuristic to catch obvious cases of adding label noise to non-classification
        raise ValueError(
            "Attempting to add label noise to a non-classification dataset"
            + " (the number of different labels is {})".format(len(labels))
        )

    y_random = labels[np.random.randint(0, len(labels), size=dataset.y_train.shape)]
    mask = np.random.random(size=dataset.y_train.shape) < p
    dataset.y_train = np.where(mask, y_random, dataset.y_train)


def get_keras_image_dataset(name: str):
    load_f = {
        "mnist": tf.keras.datasets.mnist.load_data,
        "cifar10": tf.keras.datasets.cifar10.load_data,
    }[name.lower()]
    (x_train, y_train), (x_test, y_test) = load_f()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return ClassificationDataset(x_train, y_train, x_test, y_test, name=name)


def get_cifar10():
    return get_keras_image_dataset("cifar10")


def get_mnist():
    return get_keras_image_dataset("mnist")


class GaussianProcessDataset(Dataset):
    DEFAULT_NOISE_VAR = 0.001

    def __init__(
        self,
        samples_train: int,
        lengthscale: float,
        seed=None,
        dim=1,
        noise_var=DEFAULT_NOISE_VAR,
        plot=False,
        disjoint=False,
    ):
        self.samples_train = samples_train
        self.lengthscale = lengthscale
        if seed is None:
            # For our purposes, one of 1e4 seeds is enough.
            seed = np.random.randint(int(1e4))
        self.seed = seed
        self.dim = dim
        self.noise_var = noise_var
        self.disjoint = disjoint
        x_min, x_max = -1, 1

        gp_model = GPy.models.GPRegression(
            # It seems the constructor needs at least 1 data point.
            np.array([[(x_min + x_max) / 2] * dim]),
            np.array([[0]]),
            noise_var=self.noise_var,
        )
        gp_model.kern.lengthscale = lengthscale
        # 0.1 * (x_max - x_min)
        self.gp_model = gp_model

        with smooth.util.NumpyRandomSeed(seed):
            # We want the seed to identify the function, so start by sampling the test
            # set (a ground truth function) before sampling the training set
            if dim == 1:
                samples_test = int(50 / lengthscale)
                # Silently truncates the number of training samples if necessary.
                # Is this a good thing?
                samples_train = min(samples_train, samples_test)
                x_test = np.linspace(x_min, x_max, samples_test).reshape(-1, 1)
                y_test = gp_model.posterior_samples_f(x_test, size=1)[:, :, 0]
            else:
                samples_test = 2000
                x_test = np.random.randn(samples_test, dim)
                y_test = gp_model.posterior_samples_f(x_test, size=1)[:, :, 0]

            gp_model.set_XY(x_test, y_test)
            if self.disjoint:
                x_train = np.random.randn(samples_train, dim)
            else:
                assert samples_train <= samples_test
                # Subsampling regularly is pointless for unordered samples,
                # but it does no harm
                indices = smooth.util.subsample_regularly(samples_test, samples_train)
                x_train = x_test[indices]

            y_train = gp_model.posterior_samples_f(x_train, size=1)[:, :, 0]

        if plot:
            plt.plot(x_test, y_test)
            plt.scatter(x_train, y_train, color="g")
            gp_model.plot(plot_limits=(x_min, x_max), levels=2)

        super().__init__(x_train, y_train, x_test, y_test, name=self.get_name())

    def get_name(self):
        res = "gp-{}-{}-{}-{}".format(
            self.dim, self.seed, self.lengthscale, self.samples_train
        )
        if self.disjoint or self.noise_var != GaussianProcessDataset.DEFAULT_NOISE_VAR:
            res += "-{}".format(self.noise_var)
        if self.disjoint:
            res += "-1"

        return res

    @staticmethod
    def from_name(name):
        # MAX_SAMPLES = 1000
        parts = name.split("-")
        gp, dim, seed, lengthscale, samples_train = parts[:5]
        assert gp == "gp"
        noise_var = GaussianProcessDataset.DEFAULT_NOISE_VAR
        disjoint = False

        if len(parts) > 5:
            noise_var = parts[5]
        if len(parts) > 6:
            assert len(parts) == 7
            disjoint = parts[6]

        return GaussianProcessDataset(
            dim=int(dim),
            seed=int(seed),
            lengthscale=float(lengthscale),
            samples_train=int(samples_train),
            noise_var=float(noise_var),
            disjoint=bool(int(disjoint)),
        )


def make_regression_dataset(base_dataset_f, output_f, name):
    class C(Dataset):
        def __init__(self, samples_train: int = None):
            nonlocal name

            dataset = base_dataset_f()

            full_name = name
            if samples_train is not None:
                dataset = dataset.subset(samples_train, keep_test_set=True)
                full_name = name + "-{}".format(samples_train)

            super().__init__(
                dataset.x_train,
                output_f(dataset.x_train, dataset.y_train),
                dataset.x_test,
                output_f(dataset.x_test, dataset.y_test),
                full_name,
            )

    return C


MnistMeanDataset = make_regression_dataset(
    get_mnist, lambda x, _: np.mean(x.reshape(len(x), -1), axis=-1), "mnistmean",
)

MnistParityDataset = make_regression_dataset(
    get_mnist, lambda _, y: y % 2 * 2 - 1, "mnistparity",
)


def from_name(name):
    parts = name.split("-")
    if parts[0] == "gp":
        return GaussianProcessDataset.from_name(name)
    else:
        ds = get_keras_image_dataset(parts[0])
        if len(parts) == 1:
            return ds
        n_samples = int(parts[1])
        if len(parts) > 2:
            raise ValueError("Invalid dataset name: {}".format(name))

        ds = ds.subset(n_samples, keep_test_set=True)
        ds.name = name
        return ds


def from_params(name, label_noise=None, **kwargs):
    if "lengthscale_coef" in kwargs:
        kwargs["lengthscale"] = kwargs["lengthscale_coef"] * kwargs["dim"]
        del kwargs["lengthscale_coef"]

    datasets = {
        "gp": GaussianProcessDataset,
        "mnistparity": MnistParityDataset,
        "mnistmean": MnistMeanDataset,
    }

    if name in datasets:
        dataset = datasets[name](**kwargs)

        if label_noise is not None:
            with smooth.util.NumpyRandomSeed(65123):
                add_label_noise(dataset, label_noise)

            dataset.name += "-{}".format(label_noise)

        return dataset
    else:
        raise ValueError("Unknown dataset name: {}".format(name))
