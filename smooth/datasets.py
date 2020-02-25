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

    def subset(self, n_samples: int):
        """
        Selects the first n_samples of the training set, and an appropriately scaled number of samples
        of the test set.
        The data is not copied, but returned as a view.
        """
        assert n_samples <= len(self.x_train)
        n_samples_test = int(n_samples / len(self.x_train) * len(self.x_test))
        return ClassificationDataset(
            x_train=self.x_train[:n_samples],
            y_train=self.y_train[:n_samples],
            x_test=self.x_test[:n_samples_test],
            y_test=self.y_test[:n_samples_test],
        )


class ClassificationDataset(Dataset):
    def __init__(self, x_train, y_train, x_test, y_test, name=None):
        super().__init__(x_train, y_train, x_test, y_test, name)
        assert np.min(y_train) >= 0
        assert np.min(y_test) >= 0
        self.n_classes = np.max(y_train) + 1

    def add_label_noise(self, p: float):
        """
        With probability p for each label, the label is changed to a random incorrect label.
        """

        def random_wrong(correct):
            """ Random ints in [0, n_classes) of shape correct.shape, where the original values are avoided """
            res = np.random.randint(self.n_classes - 1, size=correct.shape)
            res = res + (res >= correct).astype(int)
            assert not np.any(res == correct)
            return res

        def add_label_noise_to_one(y):
            # Could have been implemented in a simpler way by adjusting p to account for the possibility
            # that we generate the correct label "accidentally"
            y_wrong = random_wrong(y)
            mask = np.random.random(size=y.shape) < p
            return np.where(mask, y_wrong, y)

        return ClassificationDataset(
            x_train=self.x_train,
            y_train=add_label_noise_to_one(self.y_train),
            x_test=self.x_test,
            y_test=self.y_test,
        )


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
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # x_train = x_train.astype(np.float32) / 255.0
    # x_test = x_test.astype(np.float32) / 255.0
    # return ClassificationDataset(x_train, y_train, x_test, y_test)


def get_mnist():
    return get_keras_image_dataset("mnist")
    # (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.astype(np.float32) / 255.0
    # x_test = x_test.astype(np.float32) / 255.0
    # return ClassificationDataset(x_train, y_train, x_test, y_test)


def get_mnist_variant(n_samples, label_noise):
    rng_state = np.random.get_state()
    np.random.seed(8212)
    mnist2 = get_mnist().subset(n_samples).add_label_noise(label_noise)
    np.random.set_state(rng_state)
    return mnist2


class GaussianProcessDataset(Dataset):
    def __init__(
        self,
        samples_train: int,
        lengthscale: float,
        seed=None,
        dim=1,
        # noise_var=0.01,
        plot=False,
    ):
        self.samples_train = samples_train
        self.lengthscale = lengthscale
        if seed is None:
            # For our purposes, one of 1e4 seeds is enough.
            seed = np.random.randint(int(1e4))
        self.seed = seed
        self.dim = dim
        noise_var = 0.001
        x_min, x_max = -1, 1

        model = GPy.models.GPRegression(
            # It seems the constructor needs at least 1 data point.
            np.array([[(x_min + x_max) / 2]]),
            np.array([[0]]),
            noise_var=noise_var,
        )
        model.kern.lengthscale = lengthscale
        # 0.1 * (x_max - x_min)
        self.model = model

        with smooth.util.NumpyRandomSeed(seed):
            # We want the seed to identify the function, so start by sampling the test
            # set (a ground truth function) before sampling the training set
            samples_test = int(50 / lengthscale)
            x_test = np.linspace(x_min, x_max, samples_test).reshape(-1, 1)
            y_test = model.posterior_samples_f(x_test, full_cov=True, size=1)[:, :, 0]
            model.set_XY(x_test, y_test)

            assert samples_train <= samples_test
            indices = smooth.util.subsample_regularly(samples_test, samples_train)
            x_train = x_test[indices]
            y_train = model.posterior_samples_f(x_train, full_cov=True, size=1)[:, :, 0]

        if plot:
            plt.plot(x_test, y_test)
            plt.scatter(x_train, y_train, color="g")
            model.plot(plot_limits=(x_min, x_max), levels=2)

        super().__init__(x_train, y_train, x_test, y_test, name=self.get_name())

    def get_name(self):
        return "gp-{}-{}-{}-{}".format(
            self.dim, self.seed, self.lengthscale, self.samples_train
        )

    @staticmethod
    def from_name(name):
        parts = name.split("-")
        gp, dim, seed, lengthscale, samples_train = parts
        assert gp == "gp"
        return GaussianProcessDataset(
            dim=int(dim),
            seed=int(seed),
            lengthscale=float(lengthscale),
            samples_train=int(samples_train),
        )
